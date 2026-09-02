"""Tests for the unified LLM call layer (api_mode + streaming)."""

from types import SimpleNamespace

import pytest

from zotero_arxiv_daily.llm import request_llm

MESSAGES = [{"role": "user", "content": "hi"}]


def _chat_client(content="ok"):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: _chat_response(content))
        )
    )


def _chat_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _responses_client(content="ok"):
    return SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text=content)))


def test_chat_completion_mode():
    params = {"api_mode": "chat_completion", "generation_kwargs": {"model": "m"}}
    assert request_llm(_chat_client("hello"), params, MESSAGES) == "hello"


def test_response_mode_maps_max_tokens():
    received = {}

    def create(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(output_text="Summary")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    params = {"api_mode": "response", "generation_kwargs": {"model": "m", "max_tokens": 16384}}
    assert request_llm(client, params, MESSAGES) == "Summary"
    assert received["max_output_tokens"] == 16384
    assert "max_tokens" not in received


def test_streaming_chat_completion_is_concatenated():
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="a "))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="b"))]),
    ]
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: iter(chunks)))
    )
    params = {"api_mode": "chat_completion", "generation_kwargs": {"stream": True}}
    assert request_llm(client, params, MESSAGES) == "a b"


def test_invalid_api_mode_raises():
    params = {"api_mode": "bogus", "generation_kwargs": {}}
    with pytest.raises(ValueError, match="api_mode"):
        request_llm(_chat_client(), params, MESSAGES)


def test_default_api_mode_is_chat_completion():
    params = {"generation_kwargs": {}}
    assert request_llm(_chat_client("default"), params, MESSAGES) == "default"
