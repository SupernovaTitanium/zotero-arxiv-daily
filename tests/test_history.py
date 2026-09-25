"""Tests for RecommendedHistory: roundtrip, merge, review split, pruning."""

from datetime import date, timedelta

from zotero_arxiv_daily.history import RecommendedHistory


def test_load_missing_file_gives_empty_history(tmp_path):
    history = RecommendedHistory.load(tmp_path / "recommended.json")
    assert history.papers == {}


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "recommended.json"
    history = RecommendedHistory(path)
    history.record_paper(
        ["doi:10.1/a", "title:foo"], title="Foo", abstract="An abstract.", presented=True
    )
    history.save()
    loaded = RecommendedHistory.load(path)
    entry = next(iter(loaded.papers.values()))
    assert entry["title"] == "Foo"
    assert entry["presented"] is True
    assert set(entry["keys"]) == {"doi:10.1/a", "title:foo"}


def test_record_merges_entries_sharing_a_key(tmp_path):
    history = RecommendedHistory(tmp_path / "recommended.json")
    history.record_paper(["title:foo"], title="Foo", presented=True)
    history.record_paper(["doi:10.1/a", "title:foo"], title="Foo", abstract="abs")
    assert len(history.papers) == 1
    entry = next(iter(history.papers.values()))
    assert set(entry["keys"]) == {"title:foo", "doi:10.1/a"}
    assert entry["presented"] is True


def test_seen_keys_aggregates_all_entries(tmp_path):
    history = RecommendedHistory(tmp_path / "recommended.json")
    history.record_paper(["title:foo"])
    history.record_paper(["title:bar"])
    assert history.seen_keys() == {"title:foo", "title:bar"}


def test_presented_not_saved_splits_by_corpus_and_grace(tmp_path):
    today = date.today()
    history = RecommendedHistory(tmp_path / "recommended.json")
    history.record_paper(["title:saved"], day=today - timedelta(days=10), presented=True)
    history.record_paper(["title:ignored"], day=today - timedelta(days=10), presented=True)
    history.record_paper(["title:fresh"], day=today - timedelta(days=1), presented=True)
    history.record_paper(["title:neverpresented"], day=today - timedelta(days=10))

    saved, ignored = history.presented_not_saved({"title:saved"}, cutoff=today - timedelta(days=5))
    assert [e["keys"] for e in saved] == [["title:saved"]]
    assert [e["keys"] for e in ignored] == [["title:ignored"]]  # fresh one is within grace


def test_prune_drops_old_entries(tmp_path):
    today = date.today()
    history = RecommendedHistory(tmp_path / "recommended.json")
    history.record_paper(["title:old"], day=today - timedelta(days=40))
    history.record_paper(["title:new"], day=today)
    history.prune(30, today=today)
    assert set(history.papers) == {"title:new"}


def test_unreadable_file_is_ignored(tmp_path):
    path = tmp_path / "recommended.json"
    path.write_text("not json{", encoding="utf-8")
    history = RecommendedHistory.load(path)
    assert history.papers == {}
