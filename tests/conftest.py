"""Root conftest: a fully-populated Config dataclass fixture.

All mocking uses pytest monkeypatch + SimpleNamespace. No unittest.mock.
"""

import copy

import pytest

from zotero_arxiv_daily.config import (
    Config,
    EmailConfig,
    ExecutorConfig,
    LlmConfig,
    ZoteroConfig,
)


@pytest.fixture(scope="session")
def _base_config():
    """Session-scoped Config with all required values filled in.

    Never mutate this directly; use the function-scoped ``config`` fixture.
    """
    return Config(
        zotero=ZoteroConfig(user_id="000000", api_key="fake-zotero-key"),
        email=EmailConfig(
            sender="test@example.com",
            receiver="test@example.com",
            sender_password="test",
            smtp_server="localhost",
            smtp_port=1025,
            subject_prefix="Daily Papers",
        ),
        llm=LlmConfig(
            api_key="sk-fake",
            base_url="http://localhost:30000/v1",
            model="gpt-4o-mini",
            requests_per_minute=0,
        ),
        executor=ExecutorConfig(
            categories=["cs.AI", "cs.CV"],
            state_file=None,
            preferences_file=None,
            embedding_cache_file=None,
            output_dir=None,
        ),
    )


@pytest.fixture()
def config(_base_config):
    """Function-scoped deep copy of the session config. Safe to mutate."""
    return copy.deepcopy(_base_config)
