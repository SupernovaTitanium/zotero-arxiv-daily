"""Tests for config loading: env interpolation, deep merge, dataclass mapping."""

import pytest
import yaml

from zotero_arxiv_daily.config import ConfigError, load_config


def write_config(tmp_path, data):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return tmp_path


BASE = {
    "zotero": {"user_id": "${ZOTERO_ID}", "api_key": "${ZOTERO_KEY}"},
    "email": {
        "sender": "${SENDER}",
        "receiver": "${RECEIVER}",
        "sender_password": "${SENDER_PASSWORD}",
        "subject_prefix": "Daily Papers",
    },
    "llm": {
        "api_key": "${OPENAI_API_KEY}",
        "base_url": "${OPENAI_API_BASE}",
        "model": "gpt-4o-mini",
        "requests_per_minute": 0,
    },
    "executor": {"categories": ["cs.AI"], "lookback_days": 3},
    "embedding": {"model": "fake-model"},
}


@pytest.fixture()
def env(monkeypatch):
    values = {
        "ZOTERO_ID": "000000",
        "ZOTERO_KEY": "key",
        "SENDER": "s@example.com",
        "RECEIVER": "r@example.com",
        "SENDER_PASSWORD": "pw",
        "OPENAI_API_KEY": "sk",
        "OPENAI_API_BASE": "https://api.example.com/v1",
        "EMAIL_SMTP_SERVER": "localhost",
        "EMAIL_SMTP_PORT": "465",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def test_load_minimal_config(tmp_path, env):
    config = load_config(write_config(tmp_path, BASE))
    assert config.zotero.user_id == "000000"
    assert config.llm.model == "gpt-4o-mini"
    assert config.executor.categories == ["cs.AI"]
    assert config.executor.max_paper_num == 100
    assert config.embedding.model == "fake-model"


def test_missing_env_raises_with_names(tmp_path, monkeypatch, env):
    monkeypatch.delenv("ZOTERO_ID")
    with pytest.raises(ConfigError, match="ZOTERO_ID"):
        load_config(write_config(tmp_path, BASE))


def test_default_value_interpolation(tmp_path, monkeypatch, env):
    monkeypatch.delenv("SENDER")
    data = dict(BASE)
    data["email"] = dict(BASE["email"], sender="${SENDER:fallback@example.com}")
    config = load_config(write_config(tmp_path, data))
    assert config.email.sender == "fallback@example.com"


def test_custom_yaml_deep_merges_over_base(tmp_path, env):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(BASE), encoding="utf-8")
    (tmp_path / "custom.yaml").write_text(
        yaml.safe_dump({"llm": {"model": "custom-model"}, "executor": {"lookback_days": 7}}),
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    assert config.llm.model == "custom-model"
    assert config.executor.lookback_days == 7
    assert config.llm.api_key == "sk"  # base value preserved


def test_smtp_env_fallback_chain(tmp_path, monkeypatch, env):
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "465")
    config = load_config(write_config(tmp_path, BASE))
    assert config.email.smtp_server == "smtp.gmail.com"
    assert config.email.smtp_port == 465


def test_legacy_smtp_secrets_used_when_email_vars_absent(tmp_path, monkeypatch, env):
    monkeypatch.delenv("EMAIL_SMTP_SERVER")
    monkeypatch.delenv("EMAIL_SMTP_PORT")
    monkeypatch.setenv("SMTP_SERVER", "legacy.smtp.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    config = load_config(write_config(tmp_path, BASE))
    assert config.email.smtp_server == "legacy.smtp.com"
    assert config.email.smtp_port == 587


def test_lookback_and_max_paper_env_overrides(tmp_path, monkeypatch, env):
    monkeypatch.setenv("LOOKBACK_DAYS", "14")
    monkeypatch.setenv("MAX_PAPER_NUM", "50")
    config = load_config(write_config(tmp_path, BASE))
    assert config.executor.lookback_days == 14
    assert config.executor.max_paper_num == 50


def test_debug_env_forces_debug(tmp_path, monkeypatch, env):
    monkeypatch.setenv("DEBUG", "true")
    config = load_config(write_config(tmp_path, BASE))
    assert config.executor.debug is True


def test_missing_categories_rejected(tmp_path, env):
    data = {**BASE, "executor": {"categories": []}}
    with pytest.raises(ConfigError, match="categories"):
        load_config(write_config(tmp_path, data))


def test_include_path_must_be_list(tmp_path, env):
    data = {**BASE, "zotero": {**BASE["zotero"], "include_path": "2026/survey/**"}}
    with pytest.raises(ConfigError, match="include_path"):
        load_config(write_config(tmp_path, data))
