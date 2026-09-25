"""CLI entry point: load config, configure logging, run the pipeline."""

from __future__ import annotations

import logging
import os
import sys

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
    run(config)


if __name__ == "__main__":
    main()
