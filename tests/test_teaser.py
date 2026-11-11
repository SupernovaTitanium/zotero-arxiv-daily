"""Tests for batched teaser generation: request counting and fallbacks."""

import json
import threading
from types import SimpleNamespace

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


def _count_in_flight(monkeypatch, llm, papers, barrier=None):
    """Run the batches with a stubbed batch request; returns the highest
    number of requests that were in flight together."""
    lock = threading.Lock()
    in_flight = peak = 0

    def stub_batch(client, llm_params, batch):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        if barrier is not None:
            barrier.wait(timeout=10)  # passes only when enough requests overlap
        with lock:
            in_flight -= 1
        return {i: f"teaser for {p.title}" for i, p in enumerate(batch)}

    monkeypatch.setattr(teaser_module, "_teasers_for_batch", stub_batch)
    requests = teaser_module.generate_teasers_batch(object(), llm, papers)
    return requests, peak


def test_batches_overlap_up_to_max_concurrency(monkeypatch, llm):
    llm.teaser_batch_size = 2
    llm.max_concurrency = 3
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(12)]
    requests, peak = _count_in_flight(monkeypatch, llm, papers, barrier=threading.Barrier(3))
    assert requests == 6
    assert peak == 3
    # every paper got the teaser of its own batch slot, whichever thread ran it
    assert all(p.teaser == f"teaser for {p.title}" for p in papers)


def test_max_concurrency_one_runs_one_request_at_a_time(monkeypatch, llm):
    llm.teaser_batch_size = 2
    llm.max_concurrency = 1
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(6)]
    requests, peak = _count_in_flight(monkeypatch, llm, papers)
    assert requests == 3
    assert peak == 1


def test_failed_request_fails_the_run_and_starts_no_further_batches(monkeypatch, llm):
    llm.teaser_batch_size = 1  # one request per paper, so every call is a batch of its own
    llm.max_concurrency = 1
    calls = []

    def stub_chat(client, llm_params, system, prompt):
        calls.append(prompt)
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    with pytest.raises(RuntimeError, match="401"):
        teaser_module.generate_teasers_batch(object(), llm, [make_sample_paper() for _ in range(25)])
    assert len(calls) == 1


def test_endpoint_error_is_not_retried_paper_by_paper(monkeypatch, llm):
    from openai import OpenAIError

    llm.teaser_batch_size = 2
    llm.max_concurrency = 1
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(6)]
    calls = []

    def stub_chat(client, llm_params, system, prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise OpenAIError("connection refused")
        return json.dumps([{"index": 0, "teaser": "a"}, {"index": 1, "teaser": "b"}])

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    with pytest.raises(OpenAIError):
        teaser_module.generate_teasers_batch(object(), llm, papers)
    # no per-paper requests for the failed batch, and the third batch was never sent
    assert len(calls) == 2
    # the batch that succeeded keeps its teasers, so a later attempt can skip it
    assert [p.teaser for p in papers] == ["a", "b", None, None, None, None]


def test_endpoint_error_with_batches_in_flight_keeps_the_finished_teasers(monkeypatch, llm):
    """Several batches at once: one fails, the others that already started
    finish and keep their teasers; no per-paper requests are sent."""
    from openai import OpenAIError

    llm.teaser_batch_size = 2
    llm.max_concurrency = 3
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(6)]
    all_started = threading.Barrier(3)
    prompts = []

    def stub_chat(client, llm_params, system, prompt):
        prompts.append(prompt)
        all_started.wait(timeout=5)
        if "Paper 2" in prompt:
            raise OpenAIError("connection refused")
        return json.dumps([{"index": 0, "teaser": "t0"}, {"index": 1, "teaser": "t1"}])

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    with pytest.raises(OpenAIError):
        teaser_module.generate_teasers_batch(object(), llm, papers)
    assert len(prompts) == 3
    assert all("論文清單" in p for p in prompts)  # batch prompts only
    assert [p.teaser for p in papers] == ["t0", "t1", None, None, "t0", "t1"]


def test_client_counts_failed_requests_too(llm):
    from openai import OpenAIError

    def create(**kwargs):
        raise OpenAIError("connection refused")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        token_usage=teaser_module.TokenUsage(),
    )
    with pytest.raises(OpenAIError):
        teaser_module._chat(client, llm, "system", "prompt")
    assert client.token_usage.requests == 1
    assert client.token_usage.prompt_tokens == 0


def test_prompts_ask_for_two_thirds_of_the_limit_and_no_questions(monkeypatch, llm):
    llm.teaser_char_limit = 150
    prompts = []

    def stub_chat(client, llm_params, system, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return "teaser"
        return json.dumps([{"index": 0, "teaser": "teaser"}, {"index": 1, "teaser": "teaser"}])

    monkeypatch.setattr(teaser_module, "_chat", stub_chat)
    teaser_module.generate_teaser(object(), llm, "T", "A")
    teaser_module.generate_teasers_batch(object(), llm, [make_sample_paper(), make_sample_paper()])
    assert len(prompts) == 2
    for prompt in prompts:
        assert "100 個字以內" in prompt
        assert "不要用問句" in prompt


def test_tex_escapes_copied_from_the_abstract_are_undone(monkeypatch, llm):
    replies = iter([
        json.dumps([{"index": 0, "teaser": "完成率由53.1\\%升至89.1\\%"}, {"index": 1, "teaser": "R\\&D 與 a\\_b"}]),
        "準確率 60.18\\%",
    ])
    monkeypatch.setattr(teaser_module, "_chat", lambda client, llm_params, system, prompt: next(replies))
    papers = [make_sample_paper(), make_sample_paper()]
    teaser_module.generate_teasers_batch(object(), llm, papers)
    assert [p.teaser for p in papers] == ["完成率由53.1%升至89.1%", "R&D 與 a_b"]
    assert teaser_module.generate_teaser(object(), llm, "T", "A") == "準確率 60.18%"


def test_client_waits_minutes_not_ten_minutes_for_a_response(llm):
    client = teaser_module.make_llm_client(llm)
    assert client.timeout == teaser_module.REQUEST_TIMEOUT_SECONDS == 180
    assert client.max_retries == teaser_module.REQUEST_RETRIES


def test_no_papers_makes_no_requests(llm):
    assert teaser_module.generate_teasers_batch(object(), llm, []) == 0


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
    teaser = teaser_module.generate_teaser(object(), llm, "T", "A")
    assert len(teaser) == 10


def test_client_counts_the_tokens_of_every_response(llm):
    from types import SimpleNamespace

    def create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="teaser"))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=40),
        )

    assert isinstance(teaser_module.make_llm_client(llm).token_usage, teaser_module.TokenUsage)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        token_usage=teaser_module.TokenUsage(),
    )
    threads = [
        threading.Thread(target=teaser_module.generate_teaser, args=(client, llm, "T", "A")) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert (client.token_usage.prompt_tokens, client.token_usage.completion_tokens) == (2400, 320)


def test_response_without_usage_counts_nothing(llm):
    from tests.canned_responses import make_stub_openai_client

    client = make_stub_openai_client("teaser")
    client.token_usage = teaser_module.TokenUsage()
    assert teaser_module.generate_teaser(client, llm, "T", "A") == "teaser"
    assert client.token_usage.prompt_tokens == 0


def test_generation_kwargs_passed_and_stream_stripped(monkeypatch, llm, config):
    llm.generation_kwargs = {"temperature": 1.0, "top_p": 0.95, "seed": 114514, "stream": True}
    captured = {}

    def stub_create(**kwargs):
        captured.update(kwargs)
        response = make_stub_response("teaser")
        return response

    from types import SimpleNamespace

    def make_stub_response(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=stub_create))
    )
    teaser = teaser_module.generate_teaser(client, llm, "T", "A")
    assert teaser == "teaser"
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 0.95
    assert captured["seed"] == 114514
    assert "stream" not in captured
    assert captured["model"] == llm.model
    assert captured["max_tokens"] == llm.max_tokens
