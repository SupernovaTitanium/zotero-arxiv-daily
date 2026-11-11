"""Tests for SMTP delivery: SSL vs STARTTLS by port, subject, envelope."""

import smtplib
import ssl

import pytest

import zotero_arxiv_daily.mailer as mailer_module
from zotero_arxiv_daily.config import EmailConfig
from zotero_arxiv_daily.mailer import send_email
from tests.canned_responses import make_stub_smtp


def make_email_config(**overrides) -> EmailConfig:
    defaults = dict(
        sender="test@example.com",
        receiver="test@example.com",
        sender_password="test",
        smtp_server="localhost",
        smtp_port=1025,
        subject_prefix="Daily Papers",
    )
    defaults.update(overrides)
    return EmailConfig(**defaults)


def test_starttls_used_for_non_465_ports(monkeypatch):
    sent = []
    calls = []
    StubSMTP = make_stub_smtp(sent)

    class RecordingSMTP(StubSMTP):
        def __init__(self, *a, **kw):
            calls.append("smtp")
            super().__init__(*a, **kw)

        def starttls(self, **kwargs):
            calls.append("starttls")

    monkeypatch.setattr(smtplib, "SMTP", RecordingSMTP)
    send_email(make_email_config(smtp_port=587), "<html>hello</html>")
    assert calls == ["smtp", "starttls"]
    assert len(sent) == 1


def test_ssl_used_for_port_465(monkeypatch):
    sent = []
    calls = []
    StubSMTP = make_stub_smtp(sent)

    class RecordingSSL(StubSMTP):
        def __init__(self, *a, **kw):
            calls.append("ssl")
            super().__init__(*a, **kw)

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", RecordingSSL)
    send_email(make_email_config(smtp_port=465), "<html>hello</html>")
    assert calls == ["ssl"]  # no starttls on the SSL path
    assert len(sent) == 1


def test_envelope_and_subject(monkeypatch):
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    send_email(make_email_config(subject_prefix="My Digest"), "<html>x</html>")
    sender, recipients, body = sent[0]
    assert sender == "test@example.com"
    assert recipients == ["test@example.com"]
    assert "text/html" in body
    # Subject is MIME-encoded, so spaces may render as underscores
    assert "My_Digest" in body


def test_subject_can_be_replaced(monkeypatch):
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    send_email(make_email_config(), "<html>x</html>", subject="Digest failed")
    assert "Digest_failed" in sent[0][2] or "Digest failed" in sent[0][2]
    assert "Daily" not in sent[0][2]


def test_login_failure_propagates_without_another_attempt(monkeypatch):
    logins = []

    class FailingSMTP(make_stub_smtp([])):
        def login(self, user, password):
            logins.append(user)
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(smtplib, "SMTP", FailingSMTP)
    monkeypatch.setattr(mailer_module, "sleep", lambda seconds: None)
    with pytest.raises(smtplib.SMTPAuthenticationError):
        send_email(make_email_config(), "<html>x</html>")
    assert len(logins) == 1


def _smtp_that_drops_the_first(connections_dropped: int, sent: list, attempts: list):
    class FlakySMTP(make_stub_smtp(sent)):
        def __init__(self, *args, **kwargs):
            attempts.append(1)
            if len(attempts) <= connections_dropped:
                raise ConnectionResetError("connection reset by peer")

    return FlakySMTP


def test_dropped_connection_is_tried_again(monkeypatch):
    sent, attempts, waits = [], [], []
    monkeypatch.setattr(smtplib, "SMTP", _smtp_that_drops_the_first(2, sent, attempts))
    monkeypatch.setattr(mailer_module, "sleep", waits.append)
    send_email(make_email_config(), "<html>x</html>")
    assert len(attempts) == 3
    assert len(sent) == 1
    assert waits == [30, 30]


def test_login_that_is_refused_for_now_is_tried_again(monkeypatch):
    logins, waits = [], []

    class BusySMTP(make_stub_smtp([])):
        def login(self, user, password):
            logins.append(user)
            if len(logins) == 1:
                raise smtplib.SMTPAuthenticationError(454, b"temporary authentication failure")

    monkeypatch.setattr(smtplib, "SMTP", BusySMTP)
    monkeypatch.setattr(mailer_module, "sleep", waits.append)
    send_email(make_email_config(), "<html>x</html>")
    assert len(logins) == 2
    assert waits == [30]


def test_connection_lost_while_sending_is_not_tried_again(monkeypatch):
    """The server may have accepted the message before the connection
    dropped; sending it again could deliver it twice."""
    attempts = []

    class DroppingSMTP(make_stub_smtp([])):
        def sendmail(self, sender, recipients, msg):
            attempts.append(1)
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    monkeypatch.setattr(smtplib, "SMTP", DroppingSMTP)
    monkeypatch.setattr(mailer_module, "sleep", lambda seconds: None)
    with pytest.raises(smtplib.SMTPServerDisconnected):
        send_email(make_email_config(), "<html>x</html>")
    assert attempts == [1]


def test_connection_that_keeps_dropping_fails_after_three_attempts(monkeypatch):
    sent, attempts = [], []
    monkeypatch.setattr(smtplib, "SMTP", _smtp_that_drops_the_first(99, sent, attempts))
    monkeypatch.setattr(mailer_module, "sleep", lambda seconds: None)
    with pytest.raises(ConnectionResetError):
        send_email(make_email_config(), "<html>x</html>")
    assert len(attempts) == 3
    assert sent == []


def test_ssl_construction_gets_timeout_and_verified_context(monkeypatch):
    kwargs_seen = {}
    StubSMTP = make_stub_smtp([])

    class RecordingSSL(StubSMTP):
        def __init__(self, *args, **kwargs):
            kwargs_seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(smtplib, "SMTP_SSL", RecordingSSL)
    send_email(make_email_config(smtp_port=465), "<html>x</html>")
    assert kwargs_seen["timeout"] == 60
    assert kwargs_seen["context"].verify_mode == ssl.CERT_REQUIRED


def test_starttls_gets_verified_context(monkeypatch):
    kwargs_seen = {}
    starttls_kwargs = {}
    StubSMTP = make_stub_smtp([])

    class RecordingSMTP(StubSMTP):
        def __init__(self, *args, **kwargs):
            kwargs_seen.update(kwargs)
            super().__init__(*args, **kwargs)

        def starttls(self, **kwargs):
            starttls_kwargs.update(kwargs)

    monkeypatch.setattr(smtplib, "SMTP", RecordingSMTP)
    send_email(make_email_config(smtp_port=587), "<html>x</html>")
    assert kwargs_seen["timeout"] == 60
    assert starttls_kwargs["context"].check_hostname is True
