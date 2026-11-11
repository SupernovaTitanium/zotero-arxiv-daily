"""Tests for the Zotero corpus fetch: parallel paging, collection paths, retries."""

import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

import zotero_arxiv_daily.zotero as zotero_module
from zotero_arxiv_daily.config import ZoteroConfig

CONFIG = ZoteroConfig(user_id="000000", api_key="fake-zotero-key")
COLLECTIONS = [
    {"key": "TOP", "data": {"name": "2026", "parentCollection": False}},
    {"key": "SUB", "data": {"name": "survey", "parentCollection": "TOP"}},
]


def _item(i, abstract="An abstract.", collections=("SUB",), doi=""):
    return {
        "key": f"ITEM{i:04d}",
        "data": {
            "title": f"Paper {i}",
            "abstractNote": abstract,
            "dateAdded": "2026-01-02T03:04:05Z",
            "collections": list(collections),
            "DOI": doi,
        },
    }


@pytest.fixture()
def fake_zotero(monkeypatch):
    """Replace pyzotero's client with an in-memory library; returns the state
    the test fills in (``library``) and inspects (``calls``)."""
    state = SimpleNamespace(library=[], calls=[], failures=[], client_threads=[])

    class FakeZotero:
        def __init__(self, user_id, library_type, api_key):
            assert (user_id, library_type, api_key) == ("000000", "user", "fake-zotero-key")
            state.client_threads.append(threading.get_ident())
            self.request = None

        def items(self, itemType, limit, start, sort, direction):
            assert (itemType, sort, direction) == (zotero_module.ITEM_TYPES, "dateAdded", "asc")
            state.calls.append((start, limit))
            if (start, limit) in state.failures:
                state.failures.remove((start, limit))
                raise ConnectionError("zotero hiccup")
            self.request = SimpleNamespace(headers={"Total-Results": str(len(state.library))})
            return state.library[start : start + limit]

        def collections(self):
            return COLLECTIONS

        def everything(self, first_page):
            return first_page

    monkeypatch.setattr(zotero_module.zotero, "Zotero", FakeZotero)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return state


def test_fetch_corpus_fetches_every_page_once(fake_zotero):
    fake_zotero.library = [_item(i) for i in range(250)]
    corpus = zotero_module.fetch_corpus(CONFIG)
    assert sorted(c.title for c in corpus) == sorted(f"Paper {i}" for i in range(250))
    # one single-item request to learn the total, then one request per page
    assert sorted(fake_zotero.calls) == [(0, 1), (0, 100), (100, 100), (200, 100)]


def test_fetch_corpus_keeps_one_client_per_thread(fake_zotero):
    # pyzotero keeps the server's Backoff deadline on the client: a client per
    # request would never wait it out, a client shared by threads would mix up
    # their responses.
    fake_zotero.library = [_item(i) for i in range(2500)]
    assert len(zotero_module.fetch_corpus(CONFIG)) == 2500
    assert len(fake_zotero.calls) == 26  # the size probe plus 25 pages
    assert len(fake_zotero.client_threads) <= zotero_module.FETCH_WORKERS + 1
    assert len(set(fake_zotero.client_threads)) == len(fake_zotero.client_threads)


def test_fetch_corpus_builds_corpus_papers(fake_zotero):
    fake_zotero.library = [
        _item(0, doi="10.1234/abc"),
        _item(1, abstract=""),  # no abstract: nothing to embed
        _item(2, collections=("TOP", "GONE")),  # a deleted collection is skipped
    ]
    corpus = {c.title: c for c in zotero_module.fetch_corpus(CONFIG)}
    assert set(corpus) == {"Paper 0", "Paper 2"}
    assert corpus["Paper 0"].paths == ["2026/survey"]
    assert corpus["Paper 0"].doi == "10.1234/abc"
    assert corpus["Paper 0"].added_date == datetime(2026, 1, 2, 3, 4, 5)
    assert corpus["Paper 2"].paths == ["2026"]
    assert corpus["Paper 2"].doi is None


def test_fetch_corpus_retries_a_failed_page(fake_zotero):
    fake_zotero.library = [_item(i) for i in range(150)]
    fake_zotero.failures = [(100, 100)]
    corpus = zotero_module.fetch_corpus(CONFIG)
    assert len(corpus) == 150
    assert fake_zotero.calls.count((100, 100)) == 2


def test_fetch_corpus_empty_library(fake_zotero):
    assert zotero_module.fetch_corpus(CONFIG) == []
    assert fake_zotero.calls == [(0, 1)]
