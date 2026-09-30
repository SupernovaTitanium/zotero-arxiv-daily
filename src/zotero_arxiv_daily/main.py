"""CLI entry point: load config, configure logging, run the pipeline."""

from __future__ import annotations

import logging
import os
import sys
import traceback

from loguru import logger

from .config import Config, load_config
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


def main(config_dir: str = "config") -> None:
    config: Config = load_config(config_dir)
    configure_logging(config.executor.debug)
    if config.executor.debug:
        logger.info("Debug mode is enabled")
    exit_code = 0
    try:
        run(config)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # Some runner sessions leave a lingering non-daemon thread (observed as the
    # completed pipeline hanging until the 6h watchdog kills it, with the email
    # already sent), so exit the interpreter explicitly once the work is done.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
