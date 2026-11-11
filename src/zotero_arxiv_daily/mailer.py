"""SMTP delivery: SSL on port 465, STARTTLS otherwise."""

from __future__ import annotations

import datetime
import smtplib
import ssl
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr
from time import sleep

from loguru import logger

from .config import EmailConfig

SMTP_TIMEOUT_SECONDS = 60
# The send is the last step of a run whose results exist nowhere else, so a
# connection that fails to open or log in is worth two more tries. A
# rejected login (5xx) is not: it fails the same way every time. Nothing is
# tried again once the message itself is being sent: if the connection drops
# after the server accepted it, a second try would deliver it twice.
CONNECT_ATTEMPTS = 3
CONNECT_RETRY_WAIT_SECONDS = 30


def send_email(email: EmailConfig, html: str, subject: str | None = None) -> None:
    """Send ``html`` to the configured receiver; ``subject`` replaces the
    usual "<prefix> <date>" subject."""
    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = formataddr((email.subject_prefix, email.sender))
    msg["To"] = formataddr(("You", email.receiver))
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    msg["Subject"] = Header(subject or f"{email.subject_prefix} {today}", "utf-8").encode()

    server = _connect(email)
    try:
        server.sendmail(email.sender, [email.receiver], msg.as_string())
    finally:
        try:
            server.quit()
        except Exception as e:
            logger.debug(f"Failed to close SMTP connection after sending: {e}")
    logger.info(f"Email sent to {email.receiver}")


def _connect(email: EmailConfig) -> smtplib.SMTP:
    """An open, logged-in connection."""
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        server = None
        try:
            context = ssl.create_default_context()
            if int(email.smtp_port) == 465:
                server = smtplib.SMTP_SSL(
                    email.smtp_server, email.smtp_port, timeout=SMTP_TIMEOUT_SECONDS, context=context
                )
            else:
                server = smtplib.SMTP(email.smtp_server, email.smtp_port, timeout=SMTP_TIMEOUT_SECONDS)
                server.starttls(context=context)
            server.login(email.sender, email.sender_password)
            return server
        except OSError as e:  # smtplib's own errors are OSErrors as well
            if server is not None:
                server.close()
            rejected = isinstance(e, smtplib.SMTPResponseException) and e.smtp_code >= 500
            if rejected or attempt == CONNECT_ATTEMPTS:
                raise
            logger.warning(
                f"Connecting to the mail server failed ({type(e).__name__}: {e}); "
                f"attempt {attempt + 1}/{CONNECT_ATTEMPTS} in {CONNECT_RETRY_WAIT_SECONDS} s"
            )
            sleep(CONNECT_RETRY_WAIT_SECONDS)
    raise AssertionError("unreachable")
