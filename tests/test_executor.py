"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_zotero_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, normalize_path_patterns, rate_limit_chat_client
from zotero_arxiv_daily.rate_limit import rate_limit_openai_client
from zotero_arxiv_daily.protocol import CorpusPaper



def test_rate_limit_chat_client_waits_between_chat_requests():
    from types import SimpleNamespace

    now = [0.0]
    calls = []
    sleeps = []

    def monotonic():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def create(**kwargs):
        calls.append(now[0])
        return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    limited = rate_limit_chat_client(client, 10, sleep=sleep, monotonic=monotonic)

    assert limited.chat.completions.create() == "ok"
    assert limited.chat.completions.create() == "ok"
    assert sleeps == [6.0]
    assert calls == [0.0, 6.0]


def test_rate_limit_chat_client_retries_and_slows_down_after_429():
    from types import SimpleNamespace

    class RateLimitError(Exception):
        status_code = 429

    now = [0.0]
    calls = []
    sleeps = []

    def monotonic():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def create(**kwargs):
        calls.append(now[0])
        if len(calls) == 1:
            raise RateLimitError("too many requests")
        return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    limited = rate_limit_openai_client(
        client,
        60,
        max_retries=2,
        backoff_seconds=2,
        max_interval_seconds=10,
        sleep=sleep,
        monotonic=monotonic,
    )

    assert limited.chat.completions.create() == "ok"
    assert limited.chat.completions.create() == "ok"
    assert sleeps == [2.0, 2.0]
    assert calls == [0.0, 2.0, 4.0]


def test_rate_limit_openai_client_retries_embeddings_with_retry_after():
    from types import SimpleNamespace

    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"Retry-After": "7"})

    now = [0.0]
    calls = []
    sleeps = []

    def monotonic():
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def create(**kwargs):
        calls.append(now[0])
        if len(calls) == 1:
            raise RateLimitError("too many requests")
        return "ok"

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    limited = rate_limit_openai_client(
        client,
        None,
        max_retries=1,
        backoff_seconds=30,
        max_interval_seconds=60,
        sleep=sleep,
        monotonic=monotonic,
    )

    assert limited.embeddings.create(input=["a"], model="embedding-model") == "ok"
    assert sleeps == [7.0]
    assert calls == [0.0, 7.0]


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.zotero.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.zotero.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# fetch_zotero_corpus
# ---------------------------------------------------------------------------


def test_fetch_zotero_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_zotero_corpus_paper_with_zero_collections(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "No Collection Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": [],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch):
    """Full pipeline: Zotero fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub pyzotero
    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]
    monkeypatch.setattr(
        "zotero_arxiv_daily.protocol.Paper.generate_affiliations",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("teaser mode should skip affiliations")),
    )

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self, **kw: retrieved,
    )

    # two-stage pipeline: no network in tests
    monkeypatch.setattr(Executor, "_fetch_full_texts", lambda self, papers: None)

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 5. Stub sleep (reranker/retriever)

    # 6. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


@pytest.mark.parametrize("summary_mode", ["full", "legacy"])
def test_run_detail_modes_generate_affiliations(config, monkeypatch, summary_mode):
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_zotero_client,
    )

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.executor.max_paper_num = 1
        config.llm.summary.mode = summary_mode

    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self, **kw: [make_sample_paper(score=None)])
    monkeypatch.setattr(Executor, "_fetch_full_texts", lambda self, papers: None)
    calls = []
    monkeypatch.setattr("zotero_arxiv_daily.protocol.Paper.generate_affiliations", lambda self, *a: calls.append(self.title))

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    Executor(config).run()

    assert calls == ["Sample Paper Title"]
    assert len(sent) == 1


def test_run_no_papers_send_empty_false(config, monkeypatch):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self, **kw: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_zot = make_stub_zotero_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self, **kw: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


# ---------------------------------------------------------------------------
# Dedup state: seen keys, history recording
# ---------------------------------------------------------------------------


def test_build_seen_keys_merges_corpus_and_history(config, tmp_path):
    from zotero_arxiv_daily.state import RecommendedHistory

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.state_file = str(tmp_path / "state.json")
    executor.history = RecommendedHistory.load(executor.state_file)
    executor.history.record(["sid:arxiv:2609.00001"])

    corpus = [
        CorpusPaper(
            title="Stub Paper 1",
            abstract="",
            added_date=datetime(2026, 1, 1),
            paths=["survey"],
            doi="10.1101/abc",
        )
    ]
    seen = executor.build_seen_keys(corpus)
    assert "title:stubpaper1" in seen
    assert "doi:10.1101/abc" in seen
    assert "sid:arxiv:2609.00001" in seen


def test_run_records_history_and_passes_corpus_keys_to_retriever(config, monkeypatch, tmp_path):
    import json
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    state_path = tmp_path / "state.json"
    with open_dict(config):
        config.executor.state_file = str(state_path)

    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    captured = {}

    def fake_retrieve(self, seen_keys=None):
        captured["seen"] = seen_keys
        return [make_sample_paper(title="Brand New Paper", score=None)]

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", fake_retrieve)
    monkeypatch.setattr(Executor, "_fetch_full_texts", lambda self, papers: None)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    Executor(config).run()

    # corpus titles are handed to the retriever for dedup
    assert "title:stubpaper1" in captured["seen"]
    # and the new paper is recorded only after the email was sent
    assert len(sent) == 1
    entries = json.loads(state_path.read_text(encoding="utf-8"))
    assert "title:brandnewpaper" in entries


def test_run_does_not_save_history_when_email_fails(config, monkeypatch, tmp_path):
    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_zotero_client

    state_path = tmp_path / "state.json"
    with open_dict(config):
        config.executor.state_file = str(state_path)

    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self, **kw: [make_sample_paper(title="Doomed Paper", score=None)],
    )
    monkeypatch.setattr(Executor, "_fetch_full_texts", lambda self, papers: None)

    def _fail_send(config, html):
        raise RuntimeError("SMTP down")

    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", _fail_send)

    executor = Executor(config)
    with pytest.raises(RuntimeError, match="SMTP down"):
        executor.run()
    assert not state_path.exists(), "history must not be persisted when the email fails"


# ---------------------------------------------------------------------------
# Two-stage pipeline / Zotero robustness / run outputs
# ---------------------------------------------------------------------------


def test_fetch_full_texts_limited_to_top_n(config, monkeypatch):
    from tests.canned_responses import make_sample_paper
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers
    from omegaconf import open_dict

    with open_dict(config.executor):
        config.executor.fulltext_paper_num = 2
        config.executor.fulltext_workers = 2

    fetched = []
    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "fetch_full_text",
        lambda self, paper: fetched.append(paper.title) or "text",
    )

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.retrievers = {"arxiv": registered_retrievers["arxiv"](config)}

    papers = [make_sample_paper(title=f"P{i}", full_text=None) for i in range(5)]
    executor._fetch_full_texts(papers)

    assert sorted(fetched) == ["P0", "P1"]  # only the top 2 get full text
    assert papers[0].full_text == "text"
    assert papers[2].full_text is None


def test_fetch_zotero_corpus_retries(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    stub_zot = make_stub_zotero_client()
    calls = {"n": 0}

    class FlakyZotero:
        def __init__(self, *a, **kw):
            self._inner = stub_zot
        def __getattr__(self, name):
            return getattr(self._inner, name)
        def everything(self, gen):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient Zotero outage")
            return gen

    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: FlakyZotero())
    monkeypatch.setattr("zotero_arxiv_daily.executor.time.sleep", lambda s: None)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()
    assert len(corpus) == 2
    assert calls["n"] == 3  # collections ok, items failed once then succeeded


def test_fetch_zotero_corpus_tolerates_deleted_collection(config, monkeypatch):
    from tests.canned_responses import make_stub_zotero_client

    items = [
        {
            "data": {
                "title": "Orphan Paper",
                "abstractNote": "Abstract.",
                "dateAdded": "2026-03-01T00:00:00Z",
                "collections": ["GONE_COL"],
            }
        }
    ]
    stub_zot = make_stub_zotero_client(items=items)
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: stub_zot)

    executor = Executor.__new__(Executor)
    executor.config = config
    corpus = executor.fetch_zotero_corpus()
    assert len(corpus) == 1
    assert corpus[0].paths == []


def test_run_writes_outputs_before_sending(config, monkeypatch, tmp_path):
    import json
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper, make_stub_openai_client, make_stub_smtp, make_stub_zotero_client

    with open_dict(config):
        config.executor.output_dir = str(tmp_path / "out")

    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: make_stub_zotero_client())
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self, **kw: [make_sample_paper(title="Output Paper", score=None)],
    )
    monkeypatch.setattr(Executor, "_fetch_full_texts", lambda self, papers: None)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    Executor(config).run()

    files = list((tmp_path / "out").iterdir())
    assert any(f.name.startswith("email_") for f in files)
    summary_file = next(f for f in files if f.name.startswith("run_summary_"))
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["counts"]["presented"] == 1
    assert summary["papers"][0]["title"] == "Output Paper"
    assert "timings_seconds" in summary


def test_retrievers_share_seen_keys_within_run(config, monkeypatch):
    """A paper converted by the first source is skipped by the second source."""
    from tests.canned_responses import make_sample_paper
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401
    from zotero_arxiv_daily.retriever.base import registered_retrievers

    Arxiv = registered_retrievers["arxiv"]
    seen_keys: set[str] = set()

    # first retriever converts the paper and registers its keys
    first = Arxiv.__new__(Arxiv)
    first.name = "arxiv"
    first.config = config
    first.last_skipped = 0
    paper = make_sample_paper(title="Same Title Everywhere")
    seen_keys.update(paper.dedup_keys())

    # a biorxiv-style retriever using the default title-based raw keys
    Biorxiv = registered_retrievers["biorxiv"]
    second = Biorxiv.__new__(Biorxiv)
    second.name = "biorxiv"
    second.config = config
    second.retriever_config = {"category": ["bioinformatics"]}
    second.last_skipped = 0

    monkeypatch.setattr(
        Biorxiv,
        "_retrieve_raw_papers",
        lambda self: [{"title": "Same Title Everywhere!"}],
    )

    converted = []
    original = Biorxiv.convert_to_paper
    monkeypatch.setattr(Biorxiv, "convert_to_paper", lambda self, raw: converted.append(raw.title) or original(self, raw))

    papers = second.retrieve_papers(seen_keys=seen_keys)
    assert converted == []  # normalized-title key matches -> skipped before conversion
    assert papers == []
    assert second.last_skipped == 1
