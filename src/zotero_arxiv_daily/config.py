"""Configuration loading: base.yaml + optional custom.yaml, env interpolation, dataclasses.

Replaces the old Hydra + OmegaConf + apply_config_env_overrides chain. Rules:

- ``config/base.yaml`` holds defaults; ``config/custom.yaml`` (written from the
  CUSTOM_CONFIG variable in Actions) is deep-merged on top.
- ``${VAR}`` interpolates an environment variable; ``${VAR:default}`` falls
  back when the variable is unset or empty. Missing required vars raise.
- SMTP server/port come from ``EMAIL_SMTP_SERVER`` / ``EMAIL_SMTP_PORT``
  (preferred) or the legacy ``SMTP_SERVER`` / ``SMTP_PORT`` secrets.
- ``LOOKBACK_DAYS`` and ``MAX_PAPER_NUM`` override ``executor`` values (this
  replaces the sed-based overrides the workflow used to apply to custom.yaml).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::((?:[^}]|\}(?!\}))*))?\}")


class ConfigError(ValueError):
    pass


def _interpolate(value, missing: list[str]):
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            env = os.environ.get(name)
            if env:
                return env
            if default is not None:
                return _interpolate(default, missing)
            missing.append(name)
            return ""
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, missing) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env_int(names: list[str]) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError as e:
                raise ConfigError(f"Environment variable {name}={value!r} is not an integer") from e
    return None


@dataclass
class ZoteroConfig:
    user_id: str
    api_key: str
    include_path: list[str] | None = None
    ignore_path: list[str] | None = None


@dataclass
class EmailConfig:
    sender: str
    receiver: str
    sender_password: str
    smtp_server: str
    smtp_port: int
    subject_prefix: str = "Daily Papers"


@dataclass
class LlmConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 16384
    language: str = "Traditional Chinese"
    requests_per_minute: int = 10
    rate_limit_max_retries: int = 5
    rate_limit_backoff_seconds: int = 30
    rate_limit_max_interval_seconds: int = 300
    teaser_char_limit: int = 150
    teaser_batch_size: int = 10
    system_prompt: str | None = None
    generation_kwargs: dict = field(default_factory=dict)


@dataclass
class ExecutorConfig:
    categories: list[str]
    include_cross_list: bool = False
    debug: bool = False
    send_empty: bool = False
    max_paper_num: int = 100
    lookback_days: int = 3
    state_file: str | None = "state/recommended.json"
    history_days: int = 30
    preferences_file: str | None = "preferences.yaml"
    preference_boost_weight: float = 1.0
    preference_mute_weight: float = 1.5
    preference_grace_days: int = 5
    topic_threshold: float = 0.5
    fulltext_paper_num: int = 30
    fulltext_workers: int = 4
    embedding_cache_file: str | None = "state/corpus_embeddings.npz"
    output_dir: str | None = "output"


@dataclass
class EmbeddingConfig:
    model: str = "jinaai/jina-embeddings-v5-text-nano-retrieval"
    encode_kwargs: dict = field(default_factory=dict)


@dataclass
class Config:
    zotero: ZoteroConfig
    email: EmailConfig
    llm: LlmConfig
    executor: ExecutorConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


def _pop_patterns(data: dict, key: str) -> list[str] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(p, str) for p in value):
        raise ConfigError(
            f"zotero.{key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )
    return list(value)


def load_config(config_dir: str | Path = "config") -> Config:
    config_dir = Path(config_dir)
    base = yaml.safe_load((config_dir / "base.yaml").read_text(encoding="utf-8")) or {}
    custom_path = config_dir / "custom.yaml"
    if custom_path.exists():
        custom = yaml.safe_load(custom_path.read_text(encoding="utf-8")) or {}
        data = _deep_merge(base, custom)
    else:
        data = base

    missing: list[str] = []
    data = _interpolate(data, missing)
    if missing:
        raise ConfigError(
            f"Missing required environment variables: {', '.join(sorted(set(missing)))}. "
            "Set them (or add a default with ${VAR:default}) and retry."
        )

    email = data.get("email") or {}
    smtp_server = os.environ.get("EMAIL_SMTP_SERVER") or os.environ.get("SMTP_SERVER") or email.get("smtp_server")
    smtp_port = _env_int(["EMAIL_SMTP_PORT", "SMTP_PORT"]) or email.get("smtp_port")
    required = {
        "zotero.user_id": (data.get("zotero") or {}).get("user_id"),
        "zotero.api_key": (data.get("zotero") or {}).get("api_key"),
        "email.sender": email.get("sender"),
        "email.receiver": email.get("receiver"),
        "email.sender_password": email.get("sender_password"),
        "email.smtp_server": smtp_server,
        "email.smtp_port": smtp_port,
        "llm.api_key": (data.get("llm") or {}).get("api_key"),
        "llm.base_url": (data.get("llm") or {}).get("base_url"),
        "llm.model": (data.get("llm") or {}).get("model"),
    }
    absent = [name for name, value in required.items() if not value]
    if absent:
        raise ConfigError(f"Missing required config values: {', '.join(absent)}")

    llm = data.get("llm") or {}
    executor = data.get("executor") or {}
    embedding = data.get("embedding") or {}

    debug_raw = executor.get("debug")
    debug = debug_raw if isinstance(debug_raw, bool) else str(debug_raw).lower() == "true"
    if os.environ.get("DEBUG"):
        debug = os.environ["DEBUG"].lower() != "false"

    lookback = _env_int(["LOOKBACK_DAYS"])
    max_paper = _env_int(["MAX_PAPER_NUM"])

    executor_cfg = ExecutorConfig(
        categories=list(executor.get("categories") or []),
        include_cross_list=bool(executor.get("include_cross_list", False)),
        debug=debug,
        send_empty=bool(executor.get("send_empty", False)),
        max_paper_num=max_paper if max_paper is not None else int(executor.get("max_paper_num", 100)),
        lookback_days=lookback if lookback is not None else int(executor.get("lookback_days", 3)),
        state_file=executor.get("state_file") or None,
        history_days=int(executor.get("history_days", 30) or 30),
        preferences_file=executor.get("preferences_file") or None,
        preference_boost_weight=float(executor.get("preference_boost_weight", 1.0) or 1.0),
        preference_mute_weight=float(executor.get("preference_mute_weight", 1.5) or 1.5),
        preference_grace_days=int(executor.get("preference_grace_days", 5) or 5),
        topic_threshold=float(executor.get("topic_threshold", 0.5) or 0.5),
        fulltext_paper_num=int(executor.get("fulltext_paper_num", 30) or 0),
        fulltext_workers=int(executor.get("fulltext_workers", 4) or 4),
        embedding_cache_file=executor.get("embedding_cache_file") or None,
        output_dir=executor.get("output_dir") or None,
    )
    if not executor_cfg.categories:
        raise ConfigError("executor.categories must list at least one arXiv category, e.g. [cs.AI, cs.LG]")

    return Config(
        zotero=ZoteroConfig(
            user_id=data["zotero"]["user_id"],
            api_key=data["zotero"]["api_key"],
            include_path=_pop_patterns(data.get("zotero") or {}, "include_path"),
            ignore_path=_pop_patterns(data.get("zotero") or {}, "ignore_path"),
        ),
        email=EmailConfig(
            sender=email["sender"],
            receiver=email["receiver"],
            sender_password=email["sender_password"],
            smtp_server=str(smtp_server),
            smtp_port=int(smtp_port),
            subject_prefix=email.get("subject_prefix") or "Daily Papers",
        ),
        llm=LlmConfig(
            api_key=llm["api_key"],
            base_url=llm["base_url"],
            model=llm["model"],
            max_tokens=int(llm.get("max_tokens", 16384) or 16384),
            language=llm.get("language") or "Traditional Chinese",
            requests_per_minute=int(llm.get("requests_per_minute", 10) or 0),
            rate_limit_max_retries=int(llm.get("rate_limit_max_retries", 5) or 5),
            rate_limit_backoff_seconds=int(llm.get("rate_limit_backoff_seconds", 30) or 30),
            rate_limit_max_interval_seconds=int(llm.get("rate_limit_max_interval_seconds", 300) or 300),
            teaser_char_limit=int(llm.get("teaser_char_limit", 150) or 150),
            teaser_batch_size=int(llm.get("teaser_batch_size", 10) or 1),
            system_prompt=llm.get("system_prompt") or None,
            generation_kwargs=dict(llm.get("generation_kwargs") or {}),
        ),
        executor=executor_cfg,
        embedding=EmbeddingConfig(
            model=embedding.get("model") or "jinaai/jina-embeddings-v5-text-nano-retrieval",
            encode_kwargs=dict(embedding.get("encode_kwargs") or {}),
        ),
    )
