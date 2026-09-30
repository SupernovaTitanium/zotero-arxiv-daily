"""arXiv retrieval: search API primary, OAI-PMH harvest fallback.

arXiv rate-limits GitHub runner IPs hard (429 for hours since ~2026-09-10),
so this module is deliberately conservative: few, large, slow requests, empty
results treated as throttling, escalating backoff, and a separate OAI-PMH
service as fallback. Do not add polling loops against arXiv.
"""

from __future__ import annotations

import functools
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from time import sleep
from typing import Any

import requests
from arxiv import ArxivError, Client, Result as ArxivResult, Search, SortCriterion
from loguru import logger

from .config import Config
from .paper import Paper
from .utils import normalize_doi, normalize_title

MAX_XML_BYTES = 64 * 1024 * 1024

OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
OAI_PAGE_DELAY_SECONDS = 3
OAI_REQUEST_TIMEOUT = 90
_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


# ---------------------------------------------------------------------------
# OAI-PMH harvesting
# ---------------------------------------------------------------------------

def _oai_category_codes(set_specs: list[str]) -> list[str]:
    # OAI setSpec "group:archive:CATEGORY" -> arXiv category code:
    # "cs:cs:LG" -> "cs.LG", "physics:astro-ph:CO" -> "astro-ph.CO";
    # a two-part spec is a whole archive, "physics:quant-ph" -> "quant-ph".
    codes = []
    for spec in set_specs:
        parts = spec.split(":")
        if len(parts) == 3:
            codes.append(f"{parts[1]}.{parts[2]}")
        elif len(parts) == 2:
            codes.append(parts[1])
    return codes


def _parse_oai_page(xml_text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Parse one ListRecords page into record dicts plus the next resumption
    token (None when the list is exhausted or empty). DTDs are rejected before
    parsing: the OAI payload is a flat record list and never carries entities,
    and ElementTree would otherwise expand internal ones."""
    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        raise RuntimeError("OAI-PMH response contains a DOCTYPE/ENTITY declaration; refusing to parse")
    root = ET.fromstring(xml_text)
    error = root.find("oai:error", _OAI_NS)
    if error is not None:
        code = error.attrib.get("code", "unknown")
        if code == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI-PMH error {code}: {(error.text or '').strip()}")
    records = []
    for record in root.findall(".//oai:record", _OAI_NS):
        identifier = record.findtext("oai:header/oai:identifier", "", _OAI_NS) or ""
        records.append(
            {
                "id": identifier.rsplit("oai:arXiv.org:", 1)[-1],
                "datestamp": record.findtext("oai:header/oai:datestamp", "", _OAI_NS) or "",
                "set_specs": [el.text or "" for el in record.findall("oai:header/oai:setSpec", _OAI_NS)],
                "title": record.findtext(".//dc:title", "", _OAI_NS) or "",
                "abstract": record.findtext(".//dc:description", "", _OAI_NS) or "",
                "authors": [el.text or "" for el in record.findall(".//dc:creator", _OAI_NS)],
            }
        )
    token_el = root.find(".//oai:resumptionToken", _OAI_NS)
    token = (token_el.text or "").strip() if token_el is not None else ""
    return records, (token or None)


def _build_arxiv_result(record: dict[str, Any]) -> ArxivResult:
    arxiv_id = record["id"]
    codes = _oai_category_codes(record["set_specs"])
    datestamp = datetime.strptime(record["datestamp"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # OAI headers don't mark the primary category; setSpec order is treated as
    # primary-first. With include_cross_list=False this can over-include papers
    # whose primary differs from the first listed category — acceptable for a
    # fallback route, and dedup/ranking still gate what gets emailed.
    return ArxivResult(
        entry_id=f"https://arxiv.org/abs/{arxiv_id}",
        updated=datestamp,
        published=datestamp,
        title=" ".join(record["title"].split()),
        authors=[ArxivResult.Author(name) for name in record["authors"]],
        summary=" ".join(record["abstract"].split()),
        primary_category=codes[0] if codes else "",
        categories=codes,
        links=[
            ArxivResult.Link(
                href=f"https://arxiv.org/pdf/{arxiv_id}", title="pdf", content_type="application/pdf"
            )
        ],
    )


def _short_id(entry_id: str) -> str:
    # "https://arxiv.org/abs/2609.01234v2" -> "2609.01234" (version-less)
    tail = entry_id.rstrip("/").rsplit("/abs/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def _raw_keys(raw_paper: ArxivResult) -> list[str]:
    keys = []
    if raw_paper.doi:
        keys.append("doi:" + normalize_doi(raw_paper.doi))
    if raw_paper.title:
        keys.append("title:" + normalize_title(raw_paper.title))
    keys.append("sid:arxiv:" + _short_id(raw_paper.entry_id))
    return keys


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _papers_from_search_api(
    categories: list[str],
    include_cross_list: bool,
    lookback_days: int,
    start: datetime,
    now: datetime,
) -> list[ArxivResult]:
    # Fewer, larger pages: fewer requests means less exposure to arXiv's
    # rate limiter, and long backoffs in the loop below matter more than
    # many fast inner retries (which read as continued abuse).
    client = Client(num_retries=2, delay_seconds=10, page_size=500)
    # arxiv 4.x issues its requests without a timeout; against a congested
    # arXiv API a connection can hang for minutes. Bound it so hangs
    # surface as retryable errors instead of stalling the run.
    client._session.request = functools.partial(client._session.request, timeout=90)
    query = (
        f"({' OR '.join('cat:' + c for c in categories)})"
        f" AND submittedDate:[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
    )
    search = Search(query=query, sort_by=SortCriterion.SubmittedDate, max_results=None)
    # GitHub Actions often starts this job 1-2h after the 21:43 UTC cron,
    # landing it in arXiv's announcement window (~00:00 UTC) when the API
    # throttles hard: requests 429, hang, or — worst — answer HTTP 200 with
    # an empty feed, which arxiv 4.x turns into a silent, exception-free
    # zero ("Got empty first page; stopping generation"). A multi-day
    # window over several categories never legitimately returns zero, so
    # treat every failure mode alike: retry with escalating backoff, and
    # fail loudly if the window really can't be retrieved.
    max_attempts = 5
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_papers = [
                result
                for result in client.results(search)
                if include_cross_list or result.primary_category in categories
            ]
            last_error = None
        except (ArxivError, requests.exceptions.RequestException) as exc:
            raw_papers = []
            last_error = f"{type(exc).__name__}: {exc}"
        if raw_papers:
            return raw_papers
        if attempt < max_attempts:
            wait = 60 * attempt
            reason = last_error or "empty result while API is throttling"
            logger.warning(
                f"arXiv retrieval failed (attempt {attempt}/{max_attempts}): {reason}; retrying in {wait}s"
            )
            sleep(wait)
    if last_error:
        raise RuntimeError(f"arXiv retrieval failed after {max_attempts} attempts: {last_error}")
    raise RuntimeError(
        f"arXiv returned 0 papers after {max_attempts} attempts for a {lookback_days}-day "
        f"window over {categories} — API throttling returned an empty feed, not an empty "
        "window. Check https://status.arxiv.org."
    )


def _papers_from_oai(
    categories: list[str],
    include_cross_list: bool,
    from_day: date,
    until_day: date,
) -> list[ArxivResult]:
    params: dict[str, str] = {
        "verb": "ListRecords",
        "metadataPrefix": "oai_dc",
        "from": from_day.isoformat(),
        "until": until_day.isoformat(),
    }
    papers: list[ArxivResult] = []
    max_page_attempts = 3
    while True:
        xml_text: str | None = None
        last_error: str | None = None
        for attempt in range(1, max_page_attempts + 1):
            try:
                response = requests.get(OAI_BASE_URL, params=params, timeout=OAI_REQUEST_TIMEOUT)
                response.raise_for_status()
                # arXiv serves text/xml without a charset, so requests guesses
                # ISO-8859-1 and mojibakes non-ASCII author names; the OAI
                # payload is UTF-8 per its XML declaration.
                xml_text = response.content.decode("utf-8")
                if len(xml_text.encode("utf-8")) > MAX_XML_BYTES:
                    raise RuntimeError(f"OAI-PMH page larger than {MAX_XML_BYTES} bytes; refusing to parse")
                break
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_page_attempts:
                    wait = 30 * attempt
                    logger.warning(
                        f"OAI-PMH page request failed (attempt {attempt}/{max_page_attempts}): "
                        f"{last_error}; retrying in {wait}s"
                    )
                    sleep(wait)
        if xml_text is None:
            raise RuntimeError(
                f"OAI-PMH page request failed after {max_page_attempts} attempts: {last_error}"
            )
        records, token = _parse_oai_page(xml_text)
        for record in records:
            codes = _oai_category_codes(record["set_specs"])
            if not any(code in categories for code in codes):
                continue
            # setSpec order is treated as primary-first (see _build_arxiv_result).
            if not include_cross_list and (not codes or codes[0] not in categories):
                continue
            papers.append(_build_arxiv_result(record))
        if token is None:
            return papers
        params = {"verb": "ListRecords", "resumptionToken": token}
        sleep(OAI_PAGE_DELAY_SECONDS)


def retrieve_papers(config: Config, seen_keys: set[str]) -> list[Paper]:
    """Retrieve new arXiv papers for the configured categories and lookback
    window, convert to metadata-only Papers, and drop anything whose dedup
    keys intersect ``seen_keys`` (a shared, mutable set — converted papers'
    keys are added so later stages dedup within the same run)."""
    categories = config.executor.categories
    include_cross_list = config.executor.include_cross_list
    lookback_days = config.executor.lookback_days
    # A date range makes retrieval idempotent: if a scheduled run is missed or
    # fails, the next run still covers the missed days. submittedDate is a
    # timestamp, so the window is the last N*24h.
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    try:
        raw_papers = _papers_from_search_api(categories, include_cross_list, lookback_days, start, now)
    except RuntimeError as search_error:
        # The search API and the OAI-PMH interface are separate arXiv
        # services; since 2026-09-10 the search API hard-throttles GitHub
        # runner IPs for hours while OAI-PMH keeps answering. Harvesting
        # by OAI datestamp also catches papers updated (new version) in
        # the window, not only first submissions — acceptable drift for
        # a fallback, dedup still applies.
        logger.warning(
            f"Search API gave up ({search_error}); falling back to OAI-PMH harvest "
            f"for {start:%Y-%m-%d}..{now:%Y-%m-%d}"
        )
        raw_papers = _papers_from_oai(categories, include_cross_list, start.date(), now.date())
        if not raw_papers:
            raise RuntimeError(
                f"Both arXiv routes failed. Search API: {search_error} | OAI-PMH returned "
                f"0 usable records for {start:%Y-%m-%d}..{now:%Y-%m-%d}"
            ) from search_error
        logger.info(f"OAI-PMH fallback retrieved {len(raw_papers)} papers")
    if config.executor.debug:
        raw_papers = raw_papers[:10]

    papers: list[Paper] = []
    skipped = 0
    for raw in raw_papers:
        if seen_keys and set(_raw_keys(raw)) & seen_keys:
            skipped += 1
            continue
        paper = Paper(
            source="arxiv",
            title=raw.title,
            authors=[a.name for a in raw.authors],
            abstract=raw.summary,
            url=raw.entry_id,
            pdf_url=raw.pdf_url,
            doi=raw.doi,
            source_id=_short_id(raw.entry_id),
        )
        papers.append(paper)
        seen_keys.update(paper.dedup_keys())
    if skipped:
        logger.info(f"Skipped {skipped} papers already seen (recommended before or already in Zotero)")
    return papers
