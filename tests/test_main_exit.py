"""main() must force-exit on every outcome, with a non-zero code on failure,
and tell the reader by email when a run failed."""

import pytest

import zotero_arxiv_daily.main as main_module


@pytest.fixture()
def notices(monkeypatch):
    """Failure notices main() tried to send, as (subject, html)."""
    sent = []
    monkeypatch.setattr(
        main_module, "send_email", lambda email, html, subject=None: sent.append((subject, html))
    )
    return sent


@pytest.fixture()
def wired(monkeypatch, config, notices):
    codes = []
    monkeypatch.setattr(main_module, "load_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "configure_logging", lambda debug: None)
    monkeypatch.setattr("zotero_arxiv_daily.main.os._exit", codes.append)
    return codes


def boom(config):
    raise RuntimeError("boom")


def test_failure_exits_nonzero_and_prints_traceback(monkeypatch, wired, capsys):
    monkeypatch.setattr(main_module, "run", boom)
    main_module.main()
    assert wired == [1]
    assert "boom" in capsys.readouterr().err


def test_success_exits_zero(monkeypatch, wired, notices):
    monkeypatch.setattr(main_module, "run", lambda config: None)
    main_module.main()
    assert wired == [0]
    assert notices == []


def test_failure_is_reported_by_email_without_secrets(monkeypatch, wired, notices, config):
    config.llm.api_key = "sk-very-secret"

    def fail_with_key(config):
        raise RuntimeError("401 for key sk-very-secret <html>")

    monkeypatch.setattr(main_module, "run", fail_with_key)
    main_module.main()
    ((subject, body),) = notices
    assert subject == f"{config.email.subject_prefix} 執行失敗"
    assert "RuntimeError: 401 for key [REDACTED] &lt;html&gt;" in body
    assert wired == [1]


def test_key_at_the_clipping_point_is_not_half_shown(monkeypatch, wired, notices, config):
    config.llm.api_key = "sk-very-secret"
    message = "x" * (500 - len("RuntimeError: ") - 4) + "sk-very-secret"

    def fail_with_key(config):
        raise RuntimeError(message)

    monkeypatch.setattr(main_module, "run", fail_with_key)
    main_module.main()
    ((subject, body),) = notices
    assert "sk-v" not in body


def test_notice_that_cannot_be_sent_leaves_the_exit_code(monkeypatch, wired):
    def smtp_down(email, html, subject=None):
        raise OSError("smtp down")

    monkeypatch.setattr(main_module, "run", boom)
    monkeypatch.setattr(main_module, "send_email", smtp_down)
    main_module.main()
    assert wired == [1]


def test_interrupted_run_sends_no_notice(monkeypatch, wired, notices):
    def interrupted(config):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "run", interrupted)
    main_module.main()
    assert wired == [1]
    assert notices == []
