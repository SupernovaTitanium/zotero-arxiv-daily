"""Smoke-test arXiv retrieval, teaser generation, and email delivery.

Fetches the newest arXiv papers by category (not the daily lookback window,
which should stay quiet when there are genuinely no new papers), generates
teasers, and sends one email. Run:
    uv run python scripts/smoke_arxiv_teaser_email.py --max-papers 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arxiv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from zotero_arxiv_daily.config import load_config  # noqa: E402
from zotero_arxiv_daily.email import render_email  # noqa: E402
from zotero_arxiv_daily.mailer import send_email  # noqa: E402
from zotero_arxiv_daily.paper import Paper  # noqa: E402
from zotero_arxiv_daily.teaser import generate_teaser, make_llm_client  # noqa: E402


def run(max_papers: int) -> None:
    config = load_config(REPO_ROOT / "config")
    query = " OR ".join(f"cat:{c}" for c in config.executor.categories)
    logger.info(f"Fetching {max_papers} recent arXiv papers with query: {query}")
    client = arxiv.Client(num_retries=3, delay_seconds=5)
    search = arxiv.Search(
        query=query, max_results=max_papers, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers: list[Paper] = []
    for index, result in enumerate(client.results(search), start=1):
        papers.append(
            Paper(
                source="arxiv",
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                score=float(max_papers - index + 1),
            )
        )
    if not papers:
        raise RuntimeError("Smoke test found no arXiv papers")

    llm_client = make_llm_client(config.llm)
    for paper in papers:
        logger.info(f"Generating teaser: {paper.title}")
        teaser = generate_teaser(llm_client, config.llm, paper.title, paper.abstract, None)
        if not teaser:
            raise RuntimeError(f"Failed to generate teaser for {paper.url}")
        paper.teaser = teaser

    logger.info("Rendering and sending smoke-test email")
    send_email(config.email, render_email(papers, config.llm.teaser_char_limit))

    logger.info(f"Smoke email sent with {len(papers)} papers")
    for index, paper in enumerate(papers, start=1):
        print(f"[{index}] {paper.title}")
        print(f"    URL: {paper.url}")
        print(f"    Teaser: {paper.teaser}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=3)
    args = parser.parse_args()
    if args.max_papers <= 0:
        raise ValueError("--max-papers must be positive")
    run(args.max_papers)


if __name__ == "__main__":
    main()
