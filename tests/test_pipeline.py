"""End-to-end pipeline test with stubbed network edges: Zotero corpus, arXiv
retrieval, embedding ranking, LLM teasers, and SMTP are all faked in-process.
Verifies stage order guarantees: outputs before send, history only after send."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import OpenAIError

import zotero_arxiv_daily.pipeline as pipeline_module
from zotero_arxiv_daily import pipeline
from tests.canned_responses import make_sample_corpus, make_sample_paper


@pytest.fixture(autouse=True)
def events(monkeypatch):
    """Replace the embedder for every test (the real one would start loading
    the model in the background); returns the list its preload is logged to."""
    log = []
    monkeypatch.setattr(
        pipeline_module,
        "LocalEmbedder",
        lambda model, kwargs: SimpleNamespace(preload=lambda: log.append("preload")),
    )
    return log


@pytest.fixture()
def stubbed_pipeline(monkeypatch, config, tmp_path):
    """Wire the pipeline to in-process stubs; returns (config, sent, recorded)."""
    config.executor.output_dir = str(tmp_path / "output")
    config.executor.state_file = str(tmp_path / "state" / "recommended.json")

    corpus = make_sample_corpus(2)
    monkeypatch.setattr(pipeline_module, "fetch_corpus", lambda zotero_config: corpus)

    papers = [
        make_sample_paper(title=f"New Paper {i}", abstract=f"abstract {i}", score=float(3 - i))
        for i in range(2)
    ]
    monkeypatch.setattr(pipeline_module.arxiv, "retrieve_papers", lambda cfg, seen: papers)

    class StubRanker:
        def rank(self, candidates, corpus):
            return sorted(candidates, key=lambda p: p.score or 0.0, reverse=True)

        def assign_topics(self, presented):
            presented[0].topic = "Stub Topic"

    monkeypatch.setattr(pipeline_module, "Ranker", lambda cfg, embedder: StubRanker())
    monkeypatch.setattr(pipeline_module, "make_llm_client", lambda llm: object())

    def stub_teasers(client, llm, presented):
        for p in presented:
            p.teaser = f"teaser for {p.title}"
        return len(presented)

    monkeypatch.setattr(pipeline_module, "generate_teasers_batch", stub_teasers)

    sent = []
    monkeypatch.setattr(pipeline_module, "send_email", lambda email_cfg, html: sent.append(html))
    return config, sent, papers


def test_pipeline_sends_email_and_records_history(stubbed_pipeline):
    config, sent, papers = stubbed_pipeline
    pipeline.run(config)

    assert len(sent) == 1
    html = sent[0]
    assert "今日超級速覽" in html
    assert "New Paper 0" in html  # highest score first

    # outputs written before sending
    out_dir = Path(config.executor.output_dir)
    email_files = list(out_dir.glob("email_*.html"))
    assert len(email_files) == 1
    summary = json.loads((out_dir / f"run_summary_{email_files[0].stem.split('_')[1]}.json").read_text(encoding="utf-8"))
    assert summary["counts"]["retrieved"] == 2
    assert summary["counts"]["presented"] == 2

    # history recorded only for the retrieved papers
    history = json.loads((Path(config.executor.state_file)).read_text(encoding="utf-8"))
    assert len(history["papers"]) == 2


def test_pipeline_records_stage_timings(stubbed_pipeline):
    config, sent, papers = stubbed_pipeline
    pipeline.run(config)
    (summary_file,) = Path(config.executor.output_dir).glob("run_summary_*.json")
    timings = json.loads(summary_file.read_text(encoding="utf-8"))["timings_seconds"]
    assert set(timings) == {"zotero", "retrieve", "rerank", "topics", "tldr", "total"}


def test_pipeline_starts_loading_the_model_before_the_network_stages(monkeypatch, config, events):
    monkeypatch.setattr(
        pipeline_module, "fetch_corpus", lambda zotero_config: events.append("zotero") or make_sample_corpus(1)
    )
    monkeypatch.setattr(
        pipeline_module.arxiv, "retrieve_papers", lambda cfg, seen: events.append("retrieve") or []
    )
    pipeline.run(config)
    assert events == ["preload", "zotero", "retrieve"]


def test_pipeline_skips_email_when_no_papers(monkeypatch, config):
    monkeypatch.setattr(pipeline_module, "fetch_corpus", lambda zotero_config: make_sample_corpus(1))
    monkeypatch.setattr(pipeline_module.arxiv, "retrieve_papers", lambda cfg, seen: [])
    sent = []
    monkeypatch.setattr(pipeline_module, "send_email", lambda email_cfg, html: sent.append(html))

    pipeline.run(config)
    assert sent == []


def test_pipeline_sends_empty_email_when_configured(monkeypatch, config, tmp_path):
    config.executor.send_empty = True
    config.executor.output_dir = str(tmp_path / "out")
    monkeypatch.setattr(pipeline_module, "fetch_corpus", lambda zotero_config: make_sample_corpus(1))
    monkeypatch.setattr(pipeline_module.arxiv, "retrieve_papers", lambda cfg, seen: [])
    sent = []
    monkeypatch.setattr(pipeline_module, "send_email", lambda email_cfg, html: sent.append(html))

    pipeline.run(config)
    assert len(sent) == 1
    assert "No Papers Today" in sent[0]


def test_empty_corpus_error_does_not_expose_zotero_config(monkeypatch, config):
    config.zotero.user_id = "private-user-id"
    config.zotero.api_key = "private-api-key"
    config.zotero.include_path = ["private-collection/**"]
    monkeypatch.setattr(pipeline_module, "fetch_corpus", lambda zotero_config: [])
    with pytest.raises(RuntimeError, match="No zotero papers found") as failure:
        pipeline.run(config)
    message = str(failure.value)
    for value in (config.zotero.user_id, config.zotero.api_key, config.zotero.include_path[0]):
        assert value not in message
    assert "ZoteroConfig(" not in message


def test_pipeline_empty_corpus_fails_before_retrieval(monkeypatch, config):
    """An empty library is a failed run (exit code, failure notice), not a
    quiet day: it used to end like a run that found no new papers."""
    monkeypatch.setattr(pipeline_module, "fetch_corpus", lambda zotero_config: [])
    retrieve_calls = []
    monkeypatch.setattr(
        pipeline_module.arxiv, "retrieve_papers", lambda cfg, seen: retrieve_calls.append(1)
    )
    with pytest.raises(RuntimeError, match="No zotero papers found"):
        pipeline.run(config)
    assert retrieve_calls == []  # retrieval never ran


def test_llm_failure_is_retried_for_the_papers_that_lack_a_teaser(stubbed_pipeline, monkeypatch):
    config, sent, papers = stubbed_pipeline
    monkeypatch.setattr(pipeline_module, "TEASER_RETRY_WAITS_SECONDS", (0, 0))
    calls = []

    def flaky_teasers(client, llm, batch):
        calls.append([p.title for p in batch])
        if len(calls) == 1:
            batch[0].teaser = "teaser from the first attempt"
            raise OpenAIError("endpoint down")
        for p in batch:
            p.teaser = "teaser from the second attempt"
        return len(batch)

    monkeypatch.setattr(pipeline_module, "generate_teasers_batch", flaky_teasers)
    pipeline.run(config)
    # the second attempt asks only for the paper the first one left without a teaser
    assert calls == [["New Paper 0", "New Paper 1"], ["New Paper 1"]]
    assert "teaser from the first attempt" in sent[0]
    assert "teaser from the second attempt" in sent[0]
    assert "速覽沒有產生" not in sent[0]


def test_llm_that_stays_down_sends_the_papers_with_a_notice_and_records_them(stubbed_pipeline, monkeypatch):
    config, sent, papers = stubbed_pipeline
    config.llm.api_key = "sk-secret-key-123"
    monkeypatch.setattr(pipeline_module, "TEASER_RETRY_WAITS_SECONDS", (0, 0))
    attempts = []

    def down(client, llm, batch):
        attempts.append(len(batch))
        raise OpenAIError("Error code: 401 - account pool has no usable credential for sk-secret-key-123")

    monkeypatch.setattr(pipeline_module, "generate_teasers_batch", down)
    pipeline.run(config)
    assert attempts == [2, 2, 2]  # the first attempt, then one after each wait
    assert "今天有 2 篇論文的速覽沒有產生" in sent[0]
    assert "account pool has no usable credential" in sent[0]
    assert "abstract 0" in sent[0]  # the abstract stands in for the teaser
    history = json.loads(Path(config.executor.state_file).read_text(encoding="utf-8"))
    assert len(history["papers"]) == 2
    (summary_file,) = Path(config.executor.output_dir).glob("run_summary_*.json")
    summary = summary_file.read_text(encoding="utf-8")
    assert "401" in json.loads(summary)["teaser_error"]
    for text in (sent[0], summary):
        assert "sk-secret-key-123" not in text
        assert "[REDACTED]" in text


def test_response_that_is_not_json_counts_as_an_endpoint_failure(stubbed_pipeline, monkeypatch):
    """A 200 response whose body is not JSON raises a ValueError from the SDK,
    not an OpenAIError; the papers must still go out."""
    config, sent, papers = stubbed_pipeline
    monkeypatch.setattr(pipeline_module, "TEASER_RETRY_WAITS_SECONDS", (0,))

    def garbled(client, llm, batch):
        json.loads("<html>Bad gateway</html>")

    monkeypatch.setattr(pipeline_module, "generate_teasers_batch", garbled)
    pipeline.run(config)
    assert "JSONDecodeError" in sent[0]
    assert "今天有 2 篇論文的速覽沒有產生" in sent[0]


def test_bug_in_teaser_code_is_not_hidden_as_an_endpoint_failure(stubbed_pipeline, monkeypatch):
    config, sent, papers = stubbed_pipeline
    monkeypatch.setattr(pipeline_module, "TEASER_RETRY_WAITS_SECONDS", (0,))

    def buggy(client, llm, batch):
        raise AttributeError("'NoneType' object has no attribute 'choices'")

    monkeypatch.setattr(pipeline_module, "generate_teasers_batch", buggy)
    with pytest.raises(AttributeError):
        pipeline.run(config)
    assert sent == []


def test_history_that_cannot_be_saved_says_the_email_went_out(stubbed_pipeline, monkeypatch):
    config, sent, papers = stubbed_pipeline

    def locked(self):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(pipeline_module.RecommendedHistory, "save", locked)
    with pytest.raises(RuntimeError, match="The email was sent, but the dedup history could not be saved"):
        pipeline.run(config)
    assert len(sent) == 1
