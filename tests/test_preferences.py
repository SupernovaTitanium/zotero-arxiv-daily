"""Tests for preference keywords: load/save, score adjustment, review filtering, LLM parsing."""

import json

import pytest

from zotero_arxiv_daily.preferences import (
    Preferences,
    apply_preferences,
    build_review_messages,
    enough_evidence,
    filter_review_papers,
    load_preferences,
    parse_review_response,
    save_preferences,
)
from tests.canned_responses import make_sample_paper


def test_load_missing_file_returns_empty(tmp_path):
    assert load_preferences(tmp_path / "none.yaml").is_empty()
    assert load_preferences(None).is_empty()


def test_load_and_roundtrip(tmp_path):
    path = tmp_path / "preferences.yaml"
    save_preferences(
        path,
        Preferences(boost=["verifiable rewards"], mute=["text-to-image safety"]),
        generated_on="2026-09-09",
        stats={"saved": 3, "ignored": 7},
    )
    content = path.read_text(encoding="utf-8")
    assert "saved / 7 ignored" in content

    loaded = load_preferences(path)
    assert loaded.boost == ["verifiable rewards"]
    assert loaded.mute == ["text-to-image safety"]


def test_load_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "preferences.yaml"
    path.write_text("boost: [unclosed", encoding="utf-8")
    assert load_preferences(path).is_empty()


def _paper(title, abstract, score):
    return make_sample_paper(title=title, abstract=abstract, score=score)


def test_apply_preferences_boosts_and_mutes():
    prefs = Preferences(boost=["robotics"], mute=["quantization"])
    papers = [
        _paper("LLM Quantization", "weight quantization methods", 5.0),
        _paper("Robot Grasping", "a robotics manipulation study", 5.0),
        _paper("Unrelated Topic", "something else entirely", 9.0),
    ]
    result = apply_preferences(papers, prefs, boost_weight=1.0, mute_weight=1.5)
    titles = [p.title for p in result]
    # adjustment nudges but does not overturn a big embedding-score gap
    assert titles == ["Unrelated Topic", "Robot Grasping", "LLM Quantization"]
    assert result[1].score == pytest.approx(6.0)  # boosted
    assert result[2].score == pytest.approx(3.5)  # muted
    assert result[0].score == pytest.approx(9.0)  # untouched


def test_apply_preferences_empty_is_noop():
    papers = [_paper("A", "abs a", 1.0)]
    result = apply_preferences(papers, Preferences(), 1.0, 1.5)
    assert [p.title for p in result] == ["A"]


def test_filter_review_papers_drops_untitled():
    saved = [{"title": "Has Title", "abstract": "x"}, {"title": "", "abstract": "y"}]
    ignored = [{"title": "Ignored One"}]
    s, i = filter_review_papers(saved, ignored)
    assert len(s) == 1 and len(i) == 1


def test_enough_evidence_threshold():
    assert not enough_evidence([{"title": "a"}], [])
    assert enough_evidence([{"title": "a"}] * 3, [{"title": "b"}] * 2)


def test_build_review_messages_mentions_language_and_counts():
    saved = [{"title": "Saved Paper", "abstract": "about robotics"}]
    ignored = [{"title": "Ignored Paper", "abstract": "about quantization"}]
    messages = build_review_messages(saved, ignored, language="Traditional Chinese")
    assert messages[0]["role"] == "system"
    body = messages[1]["content"]
    assert "Traditional Chinese" in body
    assert "Saved Paper" in body and "Ignored Paper" in body
    assert "保存(1 篇)" in body and "忽略(1 篇)" in body


def test_parse_review_response_tolerates_wrapped_json():
    mutes = ["quantization", "", "x"] * 4  # 8 valid entries after cleaning
    raw = "Sure! Here is my analysis:" + chr(10) + json.dumps(
        {"boost": ["RL 5.0", " 可驗證獎勵 "], "mute": mutes}
    ) + chr(10) + "Done."
    prefs = parse_review_response(raw)
    assert prefs.boost == ["RL 5.0", "可驗證獎勵"]
    assert len(prefs.mute) == 8  # capped at MAX_KEYWORDS


def test_parse_review_response_no_json_raises():
    with pytest.raises(ValueError):
        parse_review_response("no json here")
