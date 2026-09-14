from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar, normalize_doi, normalize_title
from tempfile import TemporaryDirectory
import functools
import multiprocessing
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180

OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
OAI_PAGE_DELAY_SECONDS = 3
OAI_REQUEST_TIMEOUT = 90
_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


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
    token (None when the list is exhausted or empty)."""
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


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        categories = list(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        lookback_days = int(self.config.executor.get("lookback_days", 1) or 1)
        # Query papers submitted in the last N days instead of the daily RSS feed.
        # A date range makes retrieval idempotent: if a scheduled run is missed or
        # fails, the next run still covers the missed days.
        now = datetime.now(timezone.utc)
        # submittedDate is a timestamp (not a calendar date), so the window is
        # the last N*24h: lookback_days=1 covers the previous 24 hours.
        start = now - timedelta(days=lookback_days)
        try:
            papers = self._papers_from_search_api(categories, include_cross_list, lookback_days, start, now)
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
            papers = self._papers_from_oai(categories, include_cross_list, start.date(), now.date())
            if not papers:
                raise RuntimeError(
                    f"Both arXiv routes failed. Search API: {search_error} | OAI-PMH returned "
                    f"0 usable records for {start:%Y-%m-%d}..{now:%Y-%m-%d}"
                ) from search_error
            logger.info(f"OAI-PMH fallback retrieved {len(papers)} papers")
        if self.config.executor.debug:
            papers = papers[:10]
        return papers

    def _papers_from_search_api(
        self,
        categories: list[str],
        include_cross_list: bool,
        lookback_days: int,
        start: datetime,
        now: datetime,
    ) -> list[ArxivResult]:
        # Fewer, larger pages: fewer requests means less exposure to arXiv's
        # rate limiter, and long backoffs in the loop below matter more than
        # many fast inner retries (which read as continued abuse).
        client = arxiv.Client(num_retries=2, delay_seconds=10, page_size=500)
        # arxiv 4.x issues its requests without a timeout; against a congested
        # arXiv API a connection can hang for minutes. Bound it so hangs
        # surface as retryable errors instead of stalling the run.
        client._session.request = functools.partial(client._session.request, timeout=90)
        query = (
            f"({' OR '.join('cat:' + c for c in categories)})"
            f" AND submittedDate:[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
        )
        search = arxiv.Search(
            query=query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=None
        )
        # GitHub Actions often starts this job 1-2h after the 22:00 UTC cron,
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
            except (arxiv.ArxivError, requests.exceptions.RequestException) as exc:
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
            raise RuntimeError(
                f"arXiv retrieval failed after {max_attempts} attempts: {last_error}"
            )
        raise RuntimeError(
            f"arXiv returned 0 papers after {max_attempts} attempts for a {lookback_days}-day "
            f"window over {categories} — API throttling returned an empty feed, not an empty "
            "window. Check https://status.arxiv.org."
        )

    def _papers_from_oai(
        self,
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
                    xml_text = response.text
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

    @staticmethod
    def _short_id(entry_id: str) -> str:
        # "https://arxiv.org/abs/2609.01234v2" -> "2609.01234" (version-less)
        tail = entry_id.rstrip("/").rsplit("/abs/", 1)[-1]
        return re.sub(r"v\d+$", "", tail)

    def _raw_keys(self, raw_paper: ArxivResult) -> list[str]:
        keys = []
        if raw_paper.doi:
            keys.append("doi:" + normalize_doi(raw_paper.doi))
        if raw_paper.title:
            keys.append("title:" + normalize_title(raw_paper.title))
        keys.append("sid:arxiv:" + self._short_id(raw_paper.entry_id))
        return keys

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        # Metadata only: full text is fetched later, after ranking, for the
        # papers that actually make it into the email.
        return Paper(
            source=self.name,
            title=raw_paper.title,
            authors=[a.name for a in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
            doi=raw_paper.doi,
            source_id=self._short_id(raw_paper.entry_id),
        )

    def fetch_full_text(self, paper: Paper) -> str | None:
        full_text = extract_text_from_tar(paper)
        if full_text is None:
            full_text = extract_text_from_html(paper)
        if full_text is None:
            full_text = extract_text_from_pdf(paper)
        return full_text


def extract_text_from_html(paper: Paper) -> str | None:
    html_url = paper.url.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: Paper) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: Paper) -> str | None:
    if not paper.source_id:
        logger.warning(f"No source id available for {paper.title}")
        return None
    source_url = f"https://arxiv.org/e-print/{paper.source_id}"
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.url, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
