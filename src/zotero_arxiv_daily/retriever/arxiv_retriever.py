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
from datetime import datetime, timedelta, timezone
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


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


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Fewer, larger pages: fewer requests means less exposure to arXiv's
        # rate limiter, and long backoffs in the loop below matter more than
        # many fast inner retries (which read as continued abuse).
        client = arxiv.Client(num_retries=2, delay_seconds=10, page_size=500)
        # arxiv 4.x issues its requests without a timeout; against a congested
        # arXiv API a connection can hang for minutes. Bound it so hangs
        # surface as retryable errors instead of stalling the run.
        client._session.request = functools.partial(client._session.request, timeout=90)
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
                if self.config.executor.debug:
                    raw_papers = raw_papers[:10]
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
