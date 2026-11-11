"""CLI entry point: load config, configure logging, run the pipeline."""

from __future__ import annotations

import html
import logging
import os
import sys
import traceback

from loguru import logger

from .config import Config, load_config
from .mailer import send_email
from .pipeline import run

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_logging(debug: bool) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG" if debug else "INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _report_failure(config: Config, error: Exception) -> None:
    """Email the reader that the run failed. Without it a failed run and a
    day without new papers look the same from the inbox: no email."""
    reason = config.redact(f"{type(error).__name__}: {error}")[:500]
    body = (
        "<p>今天的論文速覽執行中途失敗：</p>"
        f"<pre style=\"white-space: pre-wrap;\">{html.escape(reason)}</pre>"
        "<p>如果今天沒有另外收到速覽信，這批論文就還沒寄出，也沒有被記為已推薦，"
        "下一次成功的執行會補上。完整經過在當次的執行記錄裡。</p>"
    )
    try:
        send_email(config.email, body, subject=f"{config.email.subject_prefix} 執行失敗")
    except Exception as e:
        logger.error(f"Could not send the failure notice: {type(e).__name__}: {e}")


def main(config_dir: str = "config") -> None:
    config: Config = load_config(config_dir)
    configure_logging(config.executor.debug)
    if config.executor.debug:
        logger.info("Debug mode is enabled")
    exit_code = 0
    try:
        run(config)
    except BaseException as e:
        traceback.print_exc()
        exit_code = 1
        if isinstance(e, Exception):  # not for an interrupted run
            _report_failure(config, e)
    # Some runner sessions leave a lingering non-daemon thread (observed as the
    # completed pipeline hanging until the 6h watchdog kills it, with the email
    # already sent), so exit the interpreter explicitly once the work is done.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
