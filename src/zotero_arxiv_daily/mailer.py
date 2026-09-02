"""SMTP delivery with TLS/SSL/plain fallback and a configurable subject."""

import datetime
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr

from loguru import logger
from omegaconf import DictConfig


def send_email(config: DictConfig, html: str):
    sender = config.email.sender
    receiver = config.email.receiver
    password = config.email.sender_password
    smtp_server = config.email.smtp_server
    smtp_port = config.email.smtp_port

    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    subject_prefix = config.email.get("subject_prefix", None) or "Daily Papers"
    msg['Subject'] = Header(f'{subject_prefix} {today}', 'utf-8').encode()

    last_error = None
    if int(smtp_port) == 465:
        attempts = (
            ("SSL", lambda: smtplib.SMTP_SSL(smtp_server, smtp_port), False),
            ("TLS", lambda: smtplib.SMTP(smtp_server, smtp_port), True),
            ("plain text", lambda: smtplib.SMTP(smtp_server, smtp_port), False),
        )
    else:
        attempts = (
        ("TLS", lambda: smtplib.SMTP(smtp_server, smtp_port), True),
        ("SSL", lambda: smtplib.SMTP_SSL(smtp_server, smtp_port), False),
        ("plain text", lambda: smtplib.SMTP(smtp_server, smtp_port), False),
        )

    for label, make_server, use_starttls in attempts:
        server = None
        try:
            server = make_server()
            if use_starttls:
                server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
            try:
                server.quit()
            except Exception as e:
                logger.debug(f"Failed to close SMTP connection after sending. {e}")
            return
        except Exception as e:
            last_error = e
            logger.debug(f"Failed to send email with {label}. {e}")
            if server is not None:
                try:
                    server.quit()
                except Exception:
                    close = getattr(server, "close", None)
                    if close is not None:
                        close()

    raise RuntimeError("Failed to send email by TLS, SSL, or plain SMTP") from last_error
