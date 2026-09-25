"""SMTP delivery: SSL on port 465, STARTTLS otherwise."""

from __future__ import annotations

import datetime
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr

from loguru import logger

from .config import EmailConfig


def send_email(email: EmailConfig, html: str) -> None:
    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = formataddr(("Github Action", email.sender))
    msg["To"] = formataddr(("You", email.receiver))
    today = datetime.datetime.now().strftime("%Y/%m/%d")
    msg["Subject"] = Header(f"{email.subject_prefix} {today}", "utf-8").encode()

    if int(email.smtp_port) == 465:
        server: smtplib.SMTP = smtplib.SMTP_SSL(email.smtp_server, email.smtp_port)
    else:
        server = smtplib.SMTP(email.smtp_server, email.smtp_port)
        server.starttls()
    try:
        server.login(email.sender, email.sender_password)
        server.sendmail(email.sender, [email.receiver], msg.as_string())
    finally:
        try:
            server.quit()
        except Exception as e:
            logger.debug(f"Failed to close SMTP connection after sending: {e}")
    logger.info(f"Email sent to {email.receiver}")
