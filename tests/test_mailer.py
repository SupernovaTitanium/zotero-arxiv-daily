"""Tests for zotero_arxiv_daily.mailer: send_email with TLS/SSL/plain fallback."""

import smtplib

from omegaconf import open_dict

from zotero_arxiv_daily.mailer import send_email
from tests.canned_responses import make_stub_smtp


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------


def test_send_email_starttls_success(config, monkeypatch):
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    send_email(config, "<html>hello</html>")
    assert len(sent) == 1
    sender, recipients, body = sent[0]
    assert sender == "test@example.com"
    assert recipients == ["test@example.com"]
    # Body is a full MIME message (base64-encoded). Check the raw MIME string.
    assert "text/html" in body


def test_send_email_falls_back_to_ssl(config, monkeypatch):
    sent = []
    call_count = {"smtp": 0}

    StubOK = make_stub_smtp(sent)

    class StubSMTP_TLS_Fails:
        def __init__(self, *a, **kw):
            call_count["smtp"] += 1
        def starttls(self):
            raise OSError("TLS not supported")

    class StubSMTP_SSL(StubOK):
        pass

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP_TLS_Fails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTP_SSL)
    send_email(config, "<html>ssl</html>")
    assert len(sent) == 1


def test_send_email_falls_back_to_ssl_when_tls_login_disconnects(config, monkeypatch):
    sent = []

    StubOK = make_stub_smtp(sent)

    class StubSMTP_TLS_Login_Fails:
        def __init__(self, *a, **kw):
            pass
        def starttls(self):
            pass
        def login(self, u, p):
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
        def quit(self):
            pass

    class StubSMTP_SSL(StubOK):
        pass

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP_TLS_Login_Fails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTP_SSL)
    send_email(config, "<html>ssl after login failure</html>")
    assert len(sent) == 1


def test_send_email_uses_ssl_first_for_port_465(config, monkeypatch):
    from omegaconf import open_dict

    sent = []
    calls = []

    with open_dict(config):
        config.email.smtp_port = 465

    class StubSMTP:
        def __init__(self, *a, **kw):
            calls.append("smtp")

        def starttls(self):
            calls.append("starttls")

    StubSSL = make_stub_smtp(sent)

    class StubSMTP_SSL(StubSSL):
        def __init__(self, *a, **kw):
            calls.append("ssl")
            super().__init__(*a, **kw)

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTP_SSL)

    send_email(config, "<html>ssl first</html>")

    assert calls == ["ssl"]
    assert len(sent) == 1


def test_send_email_falls_back_to_plain(config, monkeypatch):
    sent = []
    call_count = {"smtp": 0}

    class StubSMTP_TLS_Fails:
        def __init__(self, *a, **kw):
            call_count["smtp"] += 1
            if call_count["smtp"] == 1:
                pass  # first SMTP() call succeeds, but starttls will fail
            else:
                pass  # third SMTP() call is the plain fallback
        def starttls(self):
            raise OSError("TLS not supported")
        def login(self, u, p):
            pass
        def sendmail(self, s, r, m):
            sent.append((s, r, m))
        def quit(self):
            pass

    class StubSMTP_SSL_Fails:
        def __init__(self, *a, **kw):
            raise OSError("SSL not supported")

    monkeypatch.setattr(smtplib, "SMTP", StubSMTP_TLS_Fails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", StubSMTP_SSL_Fails)
    send_email(config, "<html>plain</html>")
    assert len(sent) == 1


def test_send_email_subject_uses_prefix(config, monkeypatch):
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    with open_dict(config):
        config.email.subject_prefix = "My Digest"
    send_email(config, "<html>x</html>")
    _, _, body = sent[0]
    # Subject is MIME-encoded, so spaces may render as underscores
    assert "My_Digest" in body
