from types import SimpleNamespace

from omegaconf import open_dict

from scripts import smoke_arxiv_teaser_email as smoke
from tests.canned_responses import make_sample_paper, make_stub_openai_client


def test_category_query_uses_arxiv_categories():
    assert smoke._category_query(["cs.AI", "cs.LG"]) == "cat:cs.AI OR cat:cs.LG"


def test_run_generates_teasers_and_sends_email(config, monkeypatch):
    with open_dict(config):
        config.llm.summary.mode = "teaser"
        config.llm.summary.teaser_char_limit = 150

    papers = [
        make_sample_paper(title="Smoke Paper 1", tldr=None, teaser=None),
        make_sample_paper(title="Smoke Paper 2", tldr=None, teaser=None),
    ]
    sent = []

    monkeypatch.setattr(smoke, "_load_config", lambda: config)
    monkeypatch.setattr(smoke, "_fetch_recent_arxiv_papers", lambda cfg, max_papers: papers[:max_papers])
    monkeypatch.setattr(smoke, "OpenAI", lambda **kwargs: make_stub_openai_client())
    monkeypatch.setattr(smoke, "rate_limit_chat_client", lambda client, requests_per_minute: client)
    monkeypatch.setattr(smoke, "send_email", lambda cfg, html: sent.append((cfg, html)))

    smoke.run(2)

    assert len(sent) == 1
    assert "Smoke Paper 1" in sent[0][1]
    assert "Hello! How can I assist you today?" in sent[0][1]
    assert [paper.tldr for paper in papers] == [
        "Hello! How can I assist you today?",
        "Hello! How can I assist you today?",
    ]


def test_fetch_recent_arxiv_papers_converts_results(config, monkeypatch):
    raw_result = SimpleNamespace(
        title="Recent Arxiv Paper",
        authors=[SimpleNamespace(name="Author A")],
        summary="Abstract text.",
        entry_id="https://arxiv.org/abs/2607.00001",
        pdf_url="https://arxiv.org/pdf/2607.00001",
    )

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            return iter([raw_result])

    monkeypatch.setattr(smoke.arxiv, "Client", FakeClient)

    papers = smoke._fetch_recent_arxiv_papers(config, 1)

    assert len(papers) == 1
    assert papers[0].title == "Recent Arxiv Paper"
    assert papers[0].authors == ["Author A"]
    assert papers[0].tldr is None


def test_run_fails_when_teaser_generation_fails(config, monkeypatch):
    papers = [make_sample_paper(title="Smoke Paper 1", tldr=None, teaser=None)]
    sent = []

    monkeypatch.setattr(smoke, "_load_config", lambda: config)
    monkeypatch.setattr(smoke, "_fetch_recent_arxiv_papers", lambda cfg, max_papers: papers)
    monkeypatch.setattr(smoke, "OpenAI", lambda **kwargs: make_stub_openai_client())
    monkeypatch.setattr(smoke, "rate_limit_chat_client", lambda client, requests_per_minute: client)
    monkeypatch.setattr(smoke, "generate_teaser", lambda *args: (_ for _ in ()).throw(RuntimeError("api 404")))
    monkeypatch.setattr(smoke, "send_email", lambda cfg, html: sent.append((cfg, html)))

    import pytest

    with pytest.raises(RuntimeError, match="api 404"):
        smoke.run(1)

    assert sent == []
