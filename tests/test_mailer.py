"""Tests for SMTP delivery: SSL vs STARTTLS by port, subject, envelope."""

import smtplib

import pytest

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

        def starttls(self):
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


def test_login_failure_propagates(monkeypatch):
    class FailingSMTP(make_stub_smtp([])):
        def login(self, user, password):
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(smtplib, "SMTP", FailingSMTP)
    with pytest.raises(smtplib.SMTPAuthenticationError):
        send_email(make_email_config(), "<html>x</html>")
