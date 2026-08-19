"""Tests for ChemrxivRetriever."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests
from omegaconf import open_dict

from zotero_arxiv_daily.retriever import get_retriever_cls
from zotero_arxiv_daily.retriever.chemrxiv_retriever import ChemrxivRetriever
from tests.canned_responses import SAMPLE_CHEMRXIV_API_RESPONSE, _chemrxiv_item


FIXED_NOW = datetime(2026, 3, 2, 12, 0, 0, tzinfo=timezone.utc)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FIXED_NOW if tz is None else FIXED_NOW.astimezone(tz)


@pytest.fixture()
def fixed_now(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.chemrxiv_retriever.datetime", _FixedDatetime)


def _patch_pages(monkeypatch, pages):
    """Patch requests.get to serve ``pages`` (list of responses) keyed by ``skip``."""
    calls = []

    def _patched(url, **kwargs):
        assert "chemrxiv.org" in url
        params = kwargs.get("params", {})
        calls.append(params)
        skip = int(params.get("skip", 0))
        limit = int(params.get("limit", 50))
        page = pages[skip // limit] if skip // limit < len(pages) else {"totalCount": 0, "itemHits": []}
        resp = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        resp.json = lambda: page
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    return calls


def _configure(config, category):
    with open_dict(config.source):
        config.source.chemrxiv = {"category": category}
    return ChemrxivRetriever(config)


def test_chemrxiv_is_registered():
    assert get_retriever_cls("chemrxiv") is ChemrxivRetriever


def test_chemrxiv_retrieve_filters_by_category_and_date(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    calls = _patch_pages(monkeypatch, [SAMPLE_CHEMRXIV_API_RESPONSE])
    retriever = _configure(config, ["theoretical and computational chemistry"])
    papers = retriever.retrieve_papers()
    # Old paper is outside 24h window; Organic Chemistry paper is filtered out.
    assert [p.title for p in papers] == ["A chemrxiv paper"]
    assert calls[0]["sort"] == "PUBLISHED_DATE_DESC"
    assert calls[0]["limit"] == 50


def test_chemrxiv_null_category_keeps_all_recent(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    _patch_pages(monkeypatch, [SAMPLE_CHEMRXIV_API_RESPONSE])
    retriever = _configure(config, None)
    papers = retriever.retrieve_papers()
    assert [p.title for p in papers] == ["A chemrxiv paper", "Another chemrxiv paper"]


def test_chemrxiv_paginates_until_cutoff(config, monkeypatch, fixed_now):
    monkeypatch.setattr(ChemrxivRetriever, "page_size", 2)
    recent = [
        _chemrxiv_item(f"id{i}", f"Paper {i}", f"2026-03-02T0{9 - i}:00:00.000Z", "Organic Chemistry", [("A", "B")])
        for i in range(4)
    ]
    old = _chemrxiv_item("old", "Old", "2026-02-01T00:00:00.000Z", "Organic Chemistry", [("A", "B")])
    pages = [
        {"totalCount": 5, "itemHits": recent[:2]},
        {"totalCount": 5, "itemHits": recent[2:]},
        {"totalCount": 5, "itemHits": [old]},
        {"totalCount": 5, "itemHits": [old]},  # must never be requested
    ]
    calls = _patch_pages(monkeypatch, pages)
    retriever = _configure(config, None)
    raw = retriever._retrieve_raw_papers()
    assert [r["title"] for r in raw] == ["Paper 0", "Paper 1", "Paper 2", "Paper 3"]
    assert [c["skip"] for c in calls] == [0, 2, 4]


def test_chemrxiv_stops_on_short_page(config, monkeypatch, fixed_now):
    calls = _patch_pages(monkeypatch, [SAMPLE_CHEMRXIV_API_RESPONSE])
    retriever = _configure(config, None)
    retriever._retrieve_raw_papers()
    # Three hits < page_size of 50, so no second request.
    assert len(calls) == 1


def test_chemrxiv_empty_response(config, monkeypatch, fixed_now):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    _patch_pages(monkeypatch, [{"totalCount": 0, "itemHits": []}])
    retriever = _configure(config, ["Organic Chemistry"])
    assert retriever.retrieve_papers() == []


def test_chemrxiv_convert_to_paper(config):
    retriever = _configure(config, None)
    raw = SAMPLE_CHEMRXIV_API_RESPONSE["itemHits"][0]["item"]
    paper = retriever.convert_to_paper(raw)
    assert paper.source == "chemrxiv"
    assert paper.title == "A chemrxiv paper"
    assert paper.authors == ["Jane Smith", "Alan Doe"]
    assert paper.abstract == "An abstract."
    assert paper.url == "https://chemrxiv.org/engage/chemrxiv/article-details/aaa111"
    assert paper.pdf_url.endswith("/aaa111/original/paper.pdf")
    assert paper.full_text is None


def test_chemrxiv_convert_to_paper_without_asset(config):
    retriever = _configure(config, None)
    raw = dict(SAMPLE_CHEMRXIV_API_RESPONSE["itemHits"][0]["item"])
    raw.pop("asset")
    paper = retriever.convert_to_paper(raw)
    assert paper.pdf_url is None


def test_chemrxiv_debug_truncates(config, monkeypatch, fixed_now):
    items = [
        _chemrxiv_item(f"id{i}", f"Paper {i}", "2026-03-02T09:00:00.000Z", "Organic Chemistry", [("A", "B")])
        for i in range(15)
    ]
    _patch_pages(monkeypatch, [{"totalCount": 15, "itemHits": items}])
    config.executor.debug = True
    retriever = _configure(config, None)
    assert len(retriever._retrieve_raw_papers()) == 10
