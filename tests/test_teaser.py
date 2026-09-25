"""Tests for batched teaser generation: request counting and fallbacks."""

import json

import pytest

import zotero_arxiv_daily.teaser as teaser_module
from tests.canned_responses import make_sample_paper


@pytest.fixture()
def llm(config):
    return config.llm


def test_batch_uses_one_request_per_batch(monkeypatch, llm):
    papers = [make_sample_paper(title=f"Paper {i}", abstract=f"abstract {i}") for i in range(4)]
    calls = []

    def stub_chat(client, llm_params, system, prompt):
        calls.append(prompt)
        items = [{"index": i, "teaser": f"teaser {i}"} for i in range(len(papers))]
        return json.dumps(items, ensure_ascii=False)

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    requests = teaser_module.generate_teasers_batch(object(), llm, papers)
    assert requests == 1
    assert all(p.teaser == f"teaser {i}" for i, p in enumerate(papers))


def test_malformed_batch_falls_back_to_per_paper(monkeypatch, llm):
    papers = [make_sample_paper(title=f"Paper {i}", abstract=f"abstract {i}") for i in range(2)]
    calls = []

    def stub_chat(client, llm_params, system, prompt):
        calls.append(prompt)
        if "JSON" in prompt:
            raise ValueError("bad json")
        return "fallback teaser"

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    requests = teaser_module.generate_teasers_batch(object(), llm, papers)
    # one batch request + one per-paper request per paper
    assert requests == 3
    assert all(p.teaser == "fallback teaser" for p in papers)


def test_batch_response_dict_format_is_accepted(monkeypatch, llm):
    paper = make_sample_paper(title="Paper", abstract="abstract")

    def stub_chat(client, llm_params, system, prompt):
        return json.dumps({"results": [{"index": 0, "teaser": "dict teaser"}]}, ensure_ascii=False)

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    requests = teaser_module.generate_teasers_batch(object(), llm, [paper])
    assert requests == 1
    assert paper.teaser == "dict teaser"


def test_teaser_is_clipped_to_char_limit(monkeypatch, llm):
    llm.teaser_char_limit = 10
    long_text = "x" * 50

    def stub_chat(client, llm_params, system, prompt):
        return long_text

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    teaser = teaser_module.generate_teaser(object(), llm, "T", "A", None)
    assert len(teaser) == 10
