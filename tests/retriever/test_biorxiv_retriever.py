"""Tests for BiorxivRetriever."""

import pytest
from omegaconf import open_dict

from zotero_arxiv_daily.retriever.biorxiv_retriever import BiorxivRetriever
from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE


def _install_response(monkeypatch, pages: list[dict]):
    """Patch requests.get to serve the given pages sequentially for biorxiv URLs.

    Records every requested URL into ``pages['urls']`` if present.
    """
    import requests
    from types import SimpleNamespace

    original_get = requests.get
    remaining = list(pages)
    requested: list[str] = []

    def _patched(url, **kw):
        if "api.biorxiv.org" not in url:
            return original_get(url, **kw)
        requested.append(url)
        page = remaining.pop(0) if remaining else {"messages": [{"status": "ok"}], "collection": []}
        resp = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
        resp.json = lambda: page
        return resp

    monkeypatch.setattr(requests, "get", _patched)
    return requested


def test_biorxiv_retrieve_keeps_all_dates_in_range(config, mock_biorxiv_api, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    # Both dates in the window whose category matches (latest-date-only behavior is gone)
    assert [p.title for p in papers] == ["A biorxiv paper", "Old biorxiv paper"]


def test_biorxiv_empty_response(config, monkeypatch):
    empty = {"messages": [{"status": "ok"}], "collection": []}
    _install_response(monkeypatch, [empty])
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    papers = retriever.retrieve_papers()
    assert papers == []


def test_biorxiv_requests_explicit_date_range(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    requested = _install_response(monkeypatch, [SAMPLE_BIORXIV_API_RESPONSE])

    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    retriever._retrieve_raw_papers()

    assert len(requested) == 1
    assert "/details/biorxiv/" in requested[0]
    from_str, to_str = requested[0].split("/details/biorxiv/")[1].split("/")[:2]
    assert from_str < to_str  # explicit from/to dates instead of the "2d" alias


def test_biorxiv_follows_pagination_cursor(config, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    def _item(i, category="bioinformatics"):
        return {
            "doi": f"10.1101/2026.03.01.{i:06d}",
            "title": f"Paper {i}",
            "authors": "Smith, J.",
            "abstract": "Abstract.",
            "date": "2026-03-02",
            "category": category,
            "version": "1",
        }

    pages = [
        {"messages": [{"status": "ok", "total": 3, "cursor": "2"}], "collection": [_item(1), _item(2)]},
        {"messages": [{"status": "ok", "total": 3, "cursor": ""}], "collection": [_item(3)]},
    ]
    _install_response(monkeypatch, pages)

    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()
    assert [c["title"] for c in raw_papers] == ["Paper 1", "Paper 2", "Paper 3"]


def test_biorxiv_convert_to_paper(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    raw = SAMPLE_BIORXIV_API_RESPONSE["collection"][0]
    paper = retriever.convert_to_paper(raw)
    assert paper.title == "A biorxiv paper"
    assert paper.source == "biorxiv"
    assert "biorxiv.org" in paper.pdf_url
    assert paper.authors == ["Smith, J.", "Doe, A.", "Lee, K."]
    assert paper.doi == raw["doi"]
    assert paper.source_id == raw["doi"]


def test_biorxiv_raw_keys_use_doi_and_title(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": ["bioinformatics"]}
    retriever = BiorxivRetriever(config)
    raw = SAMPLE_BIORXIV_API_RESPONSE["collection"][0]
    keys = retriever._raw_keys(raw)
    assert "doi:10.1101/2026.03.01.000001" in keys
    assert "title:abiorxivpaper" in keys


def test_biorxiv_requires_category(config):
    with open_dict(config.source):
        config.source.biorxiv = {"category": None}
    with pytest.raises(ValueError, match="category must be specified"):
        BiorxivRetriever(config)
