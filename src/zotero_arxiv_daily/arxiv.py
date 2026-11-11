"""arXiv retrieval: OAI-PMH harvest primary, search API fallback.

The harvest is the primary route because of what 43 scheduled runs
(2026-10/11) showed about the search API:

- It throttled 15 of those runs (HTTP 429, hung connections, or HTTP 200 with
  an empty feed), costing 5-19 minutes each before the harvest ran anyway.
- It selects by submission time but only lists announced papers. Submissions
  made after a daily cutoff and before the window start of the run following
  their announcement (Thursday and Friday evenings UTC) were never retrieved:
  two holes of ~500 arXiv IDs every week.

OAI-PMH datestamps follow the announcement, so a datestamp window has no such
holes. A datestamp also moves on every later change to a record, so the window
is full of replacements of old papers; identifier order, which is announcement
order, tells the two apart (see ANNOUNCEMENT_MARGIN_DAYS).

arXiv asks for at most one request every three seconds over a single
connection: keep requests few, large and sequential. Do not add polling loops
or parallel requests against arXiv.
"""

from __future__ import annotations

import functools
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from time import sleep
from typing import Any

import requests
from arxiv import ArxivError, Client, Search, SortCriterion
from loguru import logger

from .config import Config
from .paper import Paper

MAX_XML_BYTES = 64 * 1024 * 1024

OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
OAI_PAGE_DELAY_SECONDS = 3
OAI_REQUEST_TIMEOUT = 90
OAI_PAGE_ATTEMPTS = 3
_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}
_OAI_ROOT_TAG = "{http://www.openarchives.org/OAI/2.0/}OAI-PMH"
# arXiv hands out identifiers when a paper is scheduled for announcement, one
# batch per announcement, so identifier order is announcement order. A paper
# held by moderation gets its identifier in the batch that finally announces
# it, whatever its submission date: in the harvest of 2026-11-10, 36 of the
# 1281 first announcements in cs.AI/CV/LG/CL had been submitted more than ten
# days earlier, some in June. A submission date therefore cannot tell a late
# first announcement from a replacement, and the identifier can: the lowest
# identifier among the records submitted since the window start minus this
# margin marks where recent announcements begin, and everything from there up
# is recent. The margin is generous on purpose (weekends, holiday closures, a
# few missed runs): the dedup history drops anything already processed.
ANNOUNCEMENT_MARGIN_DAYS = 7
# Archives the OAI set hierarchy files under the "physics" group; every other
# archive is its own group ("cs", "math", "stat", ...).
_PHYSICS_ARCHIVES = frozenset(
    {
        "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
        "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph",
    }
)


# ---------------------------------------------------------------------------
# Author strings
# ---------------------------------------------------------------------------

_TEX_ACCENTS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", ".": "\u0307", "=": "\u0304",
    "c": "\u0327", "v": "\u030c", "H": "\u030b", "r": "\u030a", "k": "\u0328", "u": "\u0306",
}
_TEX_ACCENT_RE = re.compile(r"""\\(['`^"~.=]|[cvHrku](?![A-Za-z]))\s*\{?\\?([A-Za-z])\}?""")
_TEX_LETTERS = {
    "ss": "ß", "ae": "æ", "AE": "Æ", "aa": "å", "AA": "Å", "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "i": "ı",
}
_TEX_LETTER_RE = re.compile(r"\\(ss|ae|AE|aa|AA|o|O|l|L|i)(?![A-Za-z])\s*")
_NAME_SUFFIXES = frozenset({"Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV"})


def _decode_tex(text: str) -> str:
    """Turn the TeX accents arXiv stores in author names (``Aur\\'elien``,
    ``{\\L}ukasz``) into the characters they stand for."""
    text = _TEX_ACCENT_RE.sub(lambda m: m.group(2) + _TEX_ACCENTS[m.group(1)], text)
    text = _TEX_LETTER_RE.sub(lambda m: _TEX_LETTERS[m.group(1)], text)
    return unicodedata.normalize("NFC", text.replace("{", "").replace("}", ""))


def _split_authors(text: str) -> list[str]:
    """arXivRaw carries the authors as one free-form string: names separated
    by commas and/or "and", with optional parenthesised affiliations."""
    text = _decode_tex(text)
    without_affiliations = text
    while True:
        stripped = re.sub(r"\([^()]*\)", "", without_affiliations)
        if stripped == without_affiliations:
            break
        without_affiliations = stripped
    # "(ATLAS Collaboration)": parentheses that hold the only names there are
    # are not affiliations.
    return _split_names(without_affiliations) or _split_names(re.sub(r"[()]", " ", text))


def _split_names(text: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r",|;|\band\b", text):
        name = " ".join(part.split())
        if not name:
            continue
        if name in _NAME_SUFFIXES and names:
            names[-1] = f"{names[-1]} {name}"
        else:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# OAI-PMH harvesting
# ---------------------------------------------------------------------------

def _oai_sets(categories: list[str]) -> list[str]:
    """OAI sets covering the configured categories, one per archive:
    "cs.LG" -> "cs", "astro-ph.CO" / "quant-ph" -> "physics:astro-ph" /
    "physics:quant-ph". A set also holds the papers cross-listed into it."""
    sets: list[str] = []
    for category in categories:
        archive = category.split(".", 1)[0]
        spec = f"physics:{archive}" if archive in _PHYSICS_ARCHIVES else archive
        if spec not in sets:
            sets.append(spec)
    return sets


def _parse_oai_page(xml_text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Parse one arXivRaw ListRecords page into record dicts plus the next
    resumption token (None when the list is exhausted or empty). DTDs are
    rejected before parsing: the OAI payload is a flat record list and never
    carries entities, and ElementTree would otherwise expand internal ones.

    Anything that is not the expected shape raises ValueError rather than
    being skipped: an answer that is not an OAI-PMH record list (a maintenance
    page served with HTTP 200), a live record without arXivRaw metadata, a v1
    date that is missing or has no time zone. If arXiv ever changes the
    format, the route must fail loudly instead of quietly reporting that
    nothing is new."""
    if "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        raise RuntimeError("OAI-PMH response contains a DOCTYPE/ENTITY declaration; refusing to parse")
    root = ET.fromstring(xml_text)
    if root.tag != _OAI_ROOT_TAG:
        raise ValueError(f"not an OAI-PMH response (root element {root.tag!r})")
    error = root.find("oai:error", _OAI_NS)
    if error is not None:
        code = error.attrib.get("code", "unknown")
        if code == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI-PMH error {code}: {(error.text or '').strip()}")
    listing = root.find("oai:ListRecords", _OAI_NS)
    if listing is None:
        raise ValueError("OAI-PMH response has neither an error nor a ListRecords element")
    records = []
    for record in listing.iterfind("oai:record", _OAI_NS):
        if record.find("oai:header[@status='deleted']", _OAI_NS) is not None:
            continue  # a deleted record is a header without metadata
        meta = record.find("oai:metadata/raw:arXivRaw", _OAI_NS)
        if meta is None:
            raise ValueError("OAI-PMH record without arXivRaw metadata")
        arxiv_id = (meta.findtext("raw:id", "", _OAI_NS) or "").strip()
        first_version = meta.findtext("raw:version[@version='v1']/raw:date", None, _OAI_NS)
        if first_version is None:
            raise ValueError(f"OAI-PMH record {arxiv_id!r} has no v1 submission date")
        submitted = parsedate_to_datetime(first_version.strip())
        if submitted.tzinfo is None:
            raise ValueError(f"OAI-PMH record {arxiv_id!r} has a v1 date without a time zone")
        records.append(
            {
                "id": arxiv_id,
                "submitted": submitted,
                # Primary category first.
                "categories": (meta.findtext("raw:categories", "", _OAI_NS) or "").split(),
                "title": meta.findtext("raw:title", "", _OAI_NS) or "",
                "abstract": meta.findtext("raw:abstract", "", _OAI_NS) or "",
                "authors": meta.findtext("raw:authors", "", _OAI_NS) or "",
                "doi": (meta.findtext("raw:doi", "", _OAI_NS) or "").strip() or None,
            }
        )
    token_el = root.find(".//oai:resumptionToken", _OAI_NS)
    token = (token_el.text or "").strip() if token_el is not None else ""
    return records, (token or None)


def _fetch_oai_page(params: dict[str, str]) -> tuple[list[dict[str, Any]], str | None]:
    last_error: str | None = None
    for attempt in range(1, OAI_PAGE_ATTEMPTS + 1):
        try:
            response = requests.get(OAI_BASE_URL, params=params, timeout=OAI_REQUEST_TIMEOUT)
            response.raise_for_status()
            if len(response.content) > MAX_XML_BYTES:
                raise RuntimeError(f"OAI-PMH page larger than {MAX_XML_BYTES} bytes; refusing to parse")
            # arXiv serves text/xml without a charset, so requests would guess
            # ISO-8859-1 and mojibake non-ASCII names; the payload is UTF-8
            # per its XML declaration.
            return _parse_oai_page(response.content.decode("utf-8"))
        except (requests.exceptions.RequestException, ET.ParseError, ValueError) as exc:
            # ValueError covers undecodable bytes and malformed records: a
            # truncated or garbled page is retried like a failed request.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < OAI_PAGE_ATTEMPTS:
                wait = 30 * attempt
                logger.warning(
                    f"OAI-PMH page request failed (attempt {attempt}/{OAI_PAGE_ATTEMPTS}): "
                    f"{last_error}; retrying in {wait}s"
                )
                sleep(wait)
    raise RuntimeError(f"OAI-PMH page request failed after {OAI_PAGE_ATTEMPTS} attempts: {last_error}")


def _harvest_oai(set_specs: list[str], from_day: date, until_day: date) -> Iterator[dict[str, Any]]:
    """Yield every record of the given sets whose datestamp lies in the
    window, one request at a time with the polite delay between requests."""
    first_request = True
    for set_spec in set_specs:
        params: dict[str, str] | None = {
            "verb": "ListRecords",
            "metadataPrefix": "arXivRaw",
            "set": set_spec,
            "from": from_day.isoformat(),
            "until": until_day.isoformat(),
        }
        while params is not None:
            if not first_request:
                sleep(OAI_PAGE_DELAY_SECONDS)
            first_request = False
            records, token = _fetch_oai_page(params)
            yield from records
            params = {"verb": "ListRecords", "resumptionToken": token} if token else None


def _id_order(arxiv_id: str) -> tuple[int, int] | None:
    """Sort key of a "YYMM.NNNNN" identifier; None for the pre-2007
    "archive/YYMMNNN" form, which no recent announcement carries."""
    match = re.fullmatch(r"(\d{4})\.(\d{4,5})", arxiv_id)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _papers_from_oai(
    categories: list[str],
    include_cross_list: bool,
    start: datetime,
    now: datetime,
) -> list[Paper]:
    """Papers of the configured categories that were announced recently and
    whose record changed in the window (see ANNOUNCEMENT_MARGIN_DAYS). An
    empty list is a legitimate answer (nothing was announced, e.g. over a
    holiday closure); a glitch that produces one heals itself, since the next
    run's window overlaps this one."""
    wanted = set(categories)
    submitted_after = start - timedelta(days=ANNOUNCEMENT_MARGIN_DAYS)
    records = list(_harvest_oai(_oai_sets(categories), start.date(), now.date()))
    orders = [_id_order(record["id"]) for record in records]
    first_recent = min(
        (order for order, record in zip(orders, records) if order and record["submitted"] >= submitted_after),
        default=None,
    )
    papers: dict[str, Paper] = {}
    in_scope = held = 0
    for order, record in zip(orders, records):
        record_categories = record["categories"]
        if include_cross_list:
            if not wanted.intersection(record_categories):
                continue
        elif not record_categories or record_categories[0] not in wanted:
            continue
        in_scope += 1
        arxiv_id = record["id"]
        if first_recent is None or order is None or order < first_recent or arxiv_id in papers:
            continue
        held += record["submitted"] < submitted_after
        papers[arxiv_id] = Paper(
            source="arxiv",
            title=" ".join(record["title"].split()),
            authors=_split_authors(record["authors"]),
            abstract=" ".join(record["abstract"].split()),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            doi=record["doi"],
            source_id=arxiv_id,
        )
    logger.info(
        f"OAI-PMH harvest {start:%Y-%m-%d}..{now:%Y-%m-%d}: {len(records)} records, "
        f"{in_scope} in the configured categories, {len(papers)} announced since {submitted_after:%Y-%m-%d} "
        f"({held} of them submitted earlier and held back)"
    )
    return list(papers.values())


# ---------------------------------------------------------------------------
# Search API fallback
# ---------------------------------------------------------------------------

def _short_id(entry_id: str) -> str:
    # "https://arxiv.org/abs/2609.01234v2" -> "2609.01234" (version-less)
    tail = entry_id.rstrip("/").rsplit("/abs/", 1)[-1]
    return re.sub(r"v\d+$", "", tail)


def _papers_from_search_api(
    categories: list[str],
    include_cross_list: bool,
    start: datetime,
    now: datetime,
) -> list[Paper]:
    """One pass over the search API, used only when the harvest failed. There
    is no backoff loop: the harvest already spent its own retries, and
    repeated requests against a throttling API read as continued abuse.

    The window is by submission time, so this route still has the holes
    described in the module docstring; the next harvest closes them, because
    papers this route never saw are not in the dedup history."""
    # Fewer, larger pages mean less exposure to arXiv's rate limiter.
    client = Client(num_retries=2, delay_seconds=10, page_size=500)
    # arxiv 4.x issues its requests without a timeout; against a congested
    # API a connection can hang for minutes. Bound it so a hang surfaces as
    # an error instead of stalling the run.
    client._session.request = functools.partial(client._session.request, timeout=90)
    query = (
        f"({' OR '.join('cat:' + c for c in categories)})"
        f" AND submittedDate:[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
    )
    search = Search(query=query, sort_by=SortCriterion.SubmittedDate, max_results=None)
    try:
        results = [
            result
            for result in client.results(search)
            if include_cross_list or result.primary_category in categories
        ]
    except (ArxivError, requests.exceptions.RequestException) as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
    if not results:
        # While throttling, the API answers HTTP 200 with an empty feed, which
        # arxiv 4.x turns into a silent, exception-free zero. A multi-day
        # window over the configured categories is not empty on its own.
        raise RuntimeError("empty feed (how the search API answers while it is throttling)")
    return [
        Paper(
            source="arxiv",
            title=raw.title,
            authors=[a.name for a in raw.authors],
            abstract=raw.summary,
            url=raw.entry_id,
            pdf_url=raw.pdf_url,
            doi=raw.doi,
            source_id=_short_id(raw.entry_id),
        )
        for raw in results
    ]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_papers(config: Config, seen_keys: set[str]) -> list[Paper]:
    """Retrieve new arXiv papers for the configured categories and lookback
    window as metadata-only Papers, and drop anything whose dedup keys
    intersect ``seen_keys`` (a shared, mutable set — returned papers' keys are
    added so later stages dedup within the same run)."""
    categories = config.executor.categories
    include_cross_list = config.executor.include_cross_list
    # A date range makes retrieval idempotent: if a scheduled run is missed or
    # fails, the next run still covers the missed days.
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=config.executor.lookback_days)
    try:
        candidates = _papers_from_oai(categories, include_cross_list, start, now)
    except RuntimeError as oai_error:
        logger.warning(
            f"OAI-PMH harvest failed ({oai_error}); falling back to the search API "
            f"for {start:%Y-%m-%d}..{now:%Y-%m-%d}"
        )
        try:
            candidates = _papers_from_search_api(categories, include_cross_list, start, now)
        except RuntimeError as search_error:
            raise RuntimeError(
                f"Both arXiv routes failed. OAI-PMH: {oai_error} | Search API: {search_error}"
            ) from search_error
        logger.info(f"Search API fallback retrieved {len(candidates)} papers")
    if config.executor.debug:
        candidates = candidates[:10]

    papers: list[Paper] = []
    skipped = 0
    for paper in candidates:
        keys = paper.dedup_keys()
        if seen_keys.intersection(keys):
            skipped += 1
            continue
        papers.append(paper)
        seen_keys.update(keys)
    if skipped:
        logger.info(f"Skipped {skipped} papers already seen (recommended before or already in Zotero)")
    return papers
