"""Tests for teaser batching and language handling in personal_summary."""

import json
from types import SimpleNamespace

from zotero_arxiv_daily.personal_summary import (
    generate_teaser,
    generate_teasers_batch,
    get_teaser_batch_size,
)


def _client_returning(content: str):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )
            )
        )
    )


def _params(**over):
    params = {
        "api_mode": "chat_completion",
        "language": "English",
        "generation_kwargs": {"model": "m", "max_tokens": 16384},
        "summary": {"mode": "teaser", "teaser_char_limit": 100, "batch_size": 10},
    }
    params.update(over)
    return params


def _paper(title):
    from tests.canned_responses import make_sample_paper
    return make_sample_paper(title=title, abstract=f"Abstract of {title}.")


def test_generate_teaser_mentions_language():
    prompts = []

    def create(**kwargs):
        prompts.append(str(kwargs.get("messages")))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="short teaser"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    text = generate_teaser(client, _params(language="Traditional Chinese"), "T", "A", None)
    assert text == "short teaser"
    assert "Traditional Chinese" in prompts[0]


def test_batch_teasers_parse_json_response():
    payload = json.dumps([
        {"index": 0, "teaser": "First teaser."},
        {"index": 1, "teaser": "Second teaser."},
    ])
    client = _client_returning(f"Here you go:\n{payload}")
    papers = [_paper("P1"), _paper("P2")]
    requests = generate_teasers_batch(client, _params(), papers)
    assert requests == 1
    assert [p.teaser for p in papers] == ["First teaser.", "Second teaser."]
    assert papers[0].tldr == "First teaser."


def test_batch_teasers_fall_back_per_paper_on_bad_json():
    client = _client_returning("not json at all")
    papers = [_paper("P1"), _paper("P2")]
    requests = generate_teasers_batch(client, _params(), papers)
    # 1 failed batch + 2 single-paper fallbacks
    assert requests == 3
    assert all(p.teaser for p in papers)


def test_batch_teasers_fall_back_for_missing_indexes():
    payload = json.dumps([{"index": 0, "teaser": "Only first."}])
    client = _client_returning(payload)
    papers = [_paper("P1"), _paper("P2")]
    requests = generate_teasers_batch(client, _params(), papers)
    assert requests == 2  # batch + 1 fallback
    assert papers[0].teaser == "Only first."
    assert papers[1].teaser  # fallback content


def test_batch_teasers_cap_max_tokens():

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    params = _params()
    generate_teasers_batch(client, params, [_paper("P1")] * 10)
    # the batch request caps max_tokens: 10 papers x 100 char limit x 4 + 1000 = 5000
    assert calls[0]["max_tokens"] == 5000


def test_batch_size_one_disables_batching():
    client = _client_returning("[]")
    papers = [_paper("P1"), _paper("P2")]
    requests = generate_teasers_batch(client, _params(summary={"mode": "teaser", "batch_size": 1}), papers)
    assert requests == 2
    assert get_teaser_batch_size(_params(summary={"batch_size": 1})) == 1
