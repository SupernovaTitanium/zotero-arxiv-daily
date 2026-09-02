"""Tests for RecommendedHistory."""

import json
from datetime import date, timedelta

from zotero_arxiv_daily.state import RecommendedHistory


def test_load_missing_file_returns_empty(tmp_path):
    history = RecommendedHistory.load(tmp_path / "state.json")
    assert history.seen_keys() == set()


def test_load_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not json at all {", encoding="utf-8")
    history = RecommendedHistory.load(path)
    assert history.seen_keys() == set()


def test_load_non_object_json_returns_empty(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    history = RecommendedHistory.load(path)
    assert history.seen_keys() == set()


def test_record_and_seen():
    history = RecommendedHistory("ignored.json")
    history.record(["title:foo", "doi:10.1/x"], day=date(2026, 9, 1))
    assert history.seen_keys() == {"title:foo", "doi:10.1/x"}


def test_prune_drops_old_entries():
    history = RecommendedHistory("ignored.json")
    today = date(2026, 9, 2)
    history.record(["title:new"], day=today)
    history.record(["title:old"], day=today - timedelta(days=31))
    history.prune(30, today=today)
    assert history.seen_keys() == {"title:new"}


def test_prune_keeps_boundary_entries():
    history = RecommendedHistory("ignored.json")
    today = date(2026, 9, 2)
    history.record(["title:edge"], day=today - timedelta(days=30))
    history.prune(30, today=today)
    assert "title:edge" in history.seen_keys()


def test_save_and_reload_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    history = RecommendedHistory(path)
    history.record(["title:foo", "sid:arxiv:2609.0001"], day=date(2026, 9, 1))
    history.save()

    reloaded = RecommendedHistory.load(path)
    assert reloaded.seen_keys() == {"title:foo", "sid:arxiv:2609.0001"}
    assert reloaded.entries["title:foo"] == "2026-09-01"
