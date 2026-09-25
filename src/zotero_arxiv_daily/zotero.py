"""Zotero corpus fetching and collection-path filtering."""

from __future__ import annotations

import random
from datetime import datetime

from loguru import logger
from pyzotero import zotero

from .config import ZoteroConfig
from .paper import CorpusPaper
from .utils import glob_match


def _retry(fn, attempts: int = 3, delay: int = 30):
    import time

    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1:
                raise
            logger.warning(f"Request failed ({e}); retry {attempt + 1}/{attempts} in {delay}s")
            time.sleep(delay)


def fetch_corpus(zotero_config: ZoteroConfig) -> list[CorpusPaper]:
    logger.info("Fetching zotero corpus")
    zot = zotero.Zotero(zotero_config.user_id, "user", zotero_config.api_key)
    collections = _retry(lambda: zot.everything(zot.collections()))
    collections = {c["key"]: c for c in collections}
    corpus = _retry(
        lambda: zot.everything(zot.items(itemType="conferencePaper || journalArticle || preprint"))
    )
    corpus = [c for c in corpus if c["data"]["abstractNote"] != ""]

    def get_collection_path(col_key: str) -> str:
        # Deleted collections referenced by a paper are skipped, not fatal.
        parts = []
        while col_key and col_key in collections:
            parts.append(collections[col_key]["data"]["name"])
            col_key = collections[col_key]["data"]["parentCollection"] or None
        return "/".join(reversed(parts))

    papers = []
    for c in corpus:
        paths = [get_collection_path(col) for col in c["data"]["collections"]]
        papers.append(
            CorpusPaper(
                title=c["data"]["title"],
                abstract=c["data"]["abstractNote"],
                added_date=datetime.strptime(c["data"]["dateAdded"], "%Y-%m-%dT%H:%M:%SZ"),
                paths=[p for p in paths if p],
                doi=c["data"].get("DOI") or None,
            )
        )
    logger.info(f"Fetched {len(papers)} zotero papers")
    return papers


def filter_corpus(corpus: list[CorpusPaper], zotero_config: ZoteroConfig) -> list[CorpusPaper]:
    include = zotero_config.include_path
    ignore = zotero_config.ignore_path
    if include:
        logger.info(f"Selecting zotero papers matching include_path: {include}")
        corpus = [
            c
            for c in corpus
            if any(glob_match(path, pattern) for path in c.paths for pattern in include)
        ]
    if ignore:
        logger.info(f"Excluding zotero papers matching ignore_path: {ignore}")
        corpus = [
            c
            for c in corpus
            if not any(glob_match(path, pattern) for path in c.paths for pattern in ignore)
        ]
    if include or ignore:
        samples = random.sample(corpus, min(5, len(corpus)))
        samples = "\n".join(c.title + " - " + "\n".join(c.paths) for c in samples)
        logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
    return corpus
