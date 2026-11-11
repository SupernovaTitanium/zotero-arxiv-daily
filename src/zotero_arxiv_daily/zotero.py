"""Zotero corpus fetching and collection-path filtering."""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
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


ITEM_TYPES = "conferencePaper || journalArticle || preprint"
PAGE_SIZE = 100  # the Zotero API maximum
# Each page takes the API 3-5 s however small the library is, so pages are
# fetched side by side. Measured on a 1318-item library: 54 s one page at a
# time, 7 s with 8 workers, no Backoff header from the API.
FETCH_WORKERS = 8


def fetch_corpus(zotero_config: ZoteroConfig) -> list[CorpusPaper]:
    logger.info("Fetching zotero corpus")
    threads = threading.local()

    def client() -> zotero.Zotero:
        # One client per thread, kept for the whole fetch. pyzotero stores the
        # last response and the server's Backoff deadline on the instance: a
        # shared client would mix up responses, and a client per request would
        # never wait out a Backoff.
        if not hasattr(threads, "zot"):
            threads.zot = zotero.Zotero(zotero_config.user_id, "user", zotero_config.api_key)
        return threads.zot

    def fetch_page(start: int, limit: int = PAGE_SIZE) -> tuple[list[dict], int]:
        """One page of paper items plus the total number of matching items.
        Sorted by dateAdded ascending, a paper added mid-fetch lands after the
        last page instead of shifting the others. One deleted mid-fetch can
        still push a neighbour across a page boundary and out of this run's
        corpus, as it could with one-by-one paging; the next run has it again."""

        def fetch() -> tuple[list[dict], int]:
            zot = client()
            items = zot.items(itemType=ITEM_TYPES, limit=limit, start=start, sort="dateAdded", direction="asc")
            return items, int(zot.request.headers["Total-Results"])

        return _retry(fetch)

    def fetch_collections() -> list[dict]:
        def fetch() -> list[dict]:
            zot = client()
            return zot.everything(zot.collections())

        return _retry(fetch)

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        collections_future = pool.submit(fetch_collections)
        # A one-item request is the cheapest way to learn how many pages there are.
        _, total = fetch_page(0, limit=1)
        pages = pool.map(lambda start: fetch_page(start)[0], range(0, total, PAGE_SIZE))
        # Keyed by item key: a page boundary that shifts mid-fetch cannot duplicate a paper.
        items = {item["key"]: item for page in pages for item in page}
        collections = {c["key"]: c for c in collections_future.result()}
    corpus = [c for c in items.values() if c["data"]["abstractNote"] != ""]

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
