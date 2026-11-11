"""Tests for RecommendedHistory: roundtrip, merge, review split, pruning."""

from datetime import date, timedelta
from pathlib import Path

import pytest

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


def test_crash_in_mid_write_leaves_the_saved_history_readable(tmp_path, monkeypatch):
    path = tmp_path / "recommended.json"
    history = RecommendedHistory(path)
    history.record_paper(["sid:arxiv:1"], title="First")
    history.save()

    def half_write(self, text, encoding=None):
        with open(self, "w", encoding=encoding) as f:
            f.write(text[: len(text) // 2])
        raise OSError("power lost")

    history.record_paper(["sid:arxiv:2"], title="Second")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", half_write)
        with pytest.raises(OSError):
            history.save()
    # written in place, the truncated file would load as an empty history
    assert RecommendedHistory.load(path).seen_keys() == {"sid:arxiv:1"}


def test_save_waits_for_a_program_holding_the_file_open(tmp_path, monkeypatch):
    """On Windows the swap fails while a reader has the file open; it must be
    tried again, not fail a run whose email has already gone out."""
    import zotero_arxiv_daily.history as history_module

    path = tmp_path / "recommended.json"
    history = RecommendedHistory(path)
    history.record_paper(["sid:arxiv:1"], title="First")
    real_replace = history_module.os.replace
    calls = []

    def held_twice(source, target):
        calls.append(1)
        if len(calls) <= 2:
            raise PermissionError(13, "The process cannot access the file")
        real_replace(source, target)

    monkeypatch.setattr(history_module.os, "replace", held_twice)
    monkeypatch.setattr(history_module, "sleep", lambda seconds: None)
    history.save()
    assert len(calls) == 3
    assert RecommendedHistory.load(path).seen_keys() == {"sid:arxiv:1"}
    assert [p.name for p in tmp_path.iterdir()] == ["recommended.json"]


def test_save_that_cannot_swap_leaves_no_temporary_file(tmp_path, monkeypatch):
    import zotero_arxiv_daily.history as history_module

    def always_held(source, target):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(history_module.os, "replace", always_held)
    monkeypatch.setattr(history_module, "sleep", lambda seconds: None)
    history = RecommendedHistory(tmp_path / "recommended.json")
    history.record_paper(["sid:arxiv:1"])
    with pytest.raises(PermissionError):
        history.save()
    assert list(tmp_path.iterdir()) == []


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
