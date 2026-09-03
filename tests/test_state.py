"""Tests for RecommendedHistory (v2 format with paper metadata + v1 migration)."""

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


def test_record_paper_and_seen_keys():
    history = RecommendedHistory("ignored.json")
    history.record_paper(
        ["doi:10.1/x", "title:foo", "sid:arxiv:1"],
        day=date(2026, 9, 1),
        title="Foo Paper",
        abstract="An abstract.",
        presented=True,
    )
    assert history.seen_keys() == {"doi:10.1/x", "title:foo", "sid:arxiv:1"}
    entry = history.papers["doi:10.1/x"]
    assert entry["title"] == "Foo Paper"
    assert entry["presented"] is True
    assert entry["date"] == "2026-09-01"


def test_record_paper_merges_keys():
    history = RecommendedHistory("ignored.json")
    history.record_paper(["title:foo"], day=date(2026, 9, 1))
    history.record_paper(["doi:10.1/x", "title:foo"], day=date(2026, 9, 2), presented=True)
    assert history.seen_keys() == {"title:foo", "doi:10.1/x"}
    assert len(history.papers) == 1
    assert history.papers["title:foo"]["presented"] is True
    assert history.papers["title:foo"]["date"] == "2026-09-02"


def test_record_paper_empty_keys_is_noop():
    history = RecommendedHistory("ignored.json")
    history.record_paper([], day=date(2026, 9, 1))
    assert history.papers == {}


def test_load_v1_flat_format_migrates(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"title:old": "2026-08-01"}), encoding="utf-8")
    history = RecommendedHistory.load(path)
    assert history.seen_keys() == {"title:old"}
    assert history.papers["title:old"]["presented"] is None  # never counts as presented
    assert history.papers["title:old"]["title"] == ""


def test_presented_not_saved_split():
    history = RecommendedHistory("ignored.json")
    history.record_paper(["title:saved"], day=date(2026, 9, 1), title="Saved", presented=True)
    history.record_paper(["title:ignored"], day=date(2026, 9, 1), title="Ignored", presented=True)
    history.record_paper(["title:unseen"], day=date(2026, 9, 1), title="Unseen", presented=False)
    history.record_paper(["title:v1legacy"], day=date(2026, 9, 1), title="Legacy")

    corpus_keys = {"title:savedpaper"}  # user saved a differently-titled variant
    saved, ignored = history.presented_not_saved(corpus_keys, cutoff=date(2026, 9, 2))
    # "saved" entry: keys don't intersect corpus_keys here, but it's within grace -> ignored
    assert [e["title"] for e in ignored] == ["Saved", "Ignored"]
    assert saved == []


def test_presented_not_saved_with_grace_cutoff():
    history = RecommendedHistory("ignored.json")
    history.record_paper(["title:fresh-ignored"], day=date(2026, 9, 10), title="Fresh", presented=True)
    history.record_paper(["title:old-ignored"], day=date(2026, 9, 1), title="Old", presented=True)
    history.record_paper(["title:saved"], day=date(2026, 9, 1), title="Saved", presented=True)

    corpus_keys = {"title:saved"}
    saved, ignored = history.presented_not_saved(corpus_keys, cutoff=date(2026, 9, 5))
    assert [e["title"] for e in saved] == ["Saved"]
    # fresh ignored is inside the grace window, so only the old one counts
    assert [e["title"] for e in ignored] == ["Old"]


def test_prune_drops_old_entries():
    history = RecommendedHistory("ignored.json")
    today = date(2026, 9, 2)
    history.record_paper(["title:new"], day=today)
    history.record_paper(["title:old"], day=today - timedelta(days=31))
    history.prune(30, today=today)
    assert history.seen_keys() == {"title:new"}


def test_prune_keeps_boundary_entries():
    history = RecommendedHistory("ignored.json")
    today = date(2026, 9, 2)
    history.record_paper(["title:edge"], day=today - timedelta(days=30))
    history.prune(30, today=today)
    assert "title:edge" in history.seen_keys()


def test_save_and_reload_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    history = RecommendedHistory(path)
    history.record_paper(
        ["title:foo", "sid:arxiv:2609.0001"],
        day=date(2026, 9, 1),
        title="Foo",
        abstract="Abs",
        presented=True,
    )
    history.save()

    reloaded = RecommendedHistory.load(path)
    assert reloaded.seen_keys() == {"title:foo", "sid:arxiv:2609.0001"}
    assert reloaded.papers["title:foo"]["presented"] is True
    assert reloaded.papers["title:foo"]["title"] == "Foo"
