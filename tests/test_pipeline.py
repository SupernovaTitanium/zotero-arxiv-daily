"""End-to-end pipeline test with stubbed network edges: Zotero corpus, arXiv
retrieval, embedding ranking, LLM teasers, and SMTP are all faked in-process.
Verifies stage order guarantees: outputs before send, history only after send."""

import json
from pathlib import Path

import pytest

import zotero_arxiv_daily.pipeline as pipeline_module
from zotero_arxiv_daily import pipeline
from tests.canned_responses import make_sample_corpus, make_sample_paper


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

        def apply_preferences(self, ranked, prefs):
            return ranked

        def assign_topics(self, presented):
            presented[0].topic = "Stub Topic"

    monkeypatch.setattr(pipeline_module, "Ranker", lambda cfg, embedder: StubRanker())
    monkeypatch.setattr(pipeline_module, "LocalEmbedder", lambda model, kwargs: object())
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


def test_pipeline_empty_corpus_returns_early(monkeypatch, config):
    monkeypatch.setattr(pipeline_module, "fetch_corpus", lambda zotero_config: [])
    retrieve_calls = []
    monkeypatch.setattr(
        pipeline_module.arxiv, "retrieve_papers", lambda cfg, seen: retrieve_calls.append(1)
    )
    pipeline.run(config)
    assert retrieve_calls == []  # retrieval never ran
