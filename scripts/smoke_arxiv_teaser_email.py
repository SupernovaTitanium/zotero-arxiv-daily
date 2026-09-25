"""Smoke-test arXiv retrieval, teaser generation, and email delivery.

Uses the production retrieval path (``arxiv.retrieve_papers``: search API with
backoff plus OAI-PMH fallback) over the configured lookback window, then
teasers and sends one email. Run:
    uv run python scripts/smoke_arxiv_teaser_email.py --max-papers 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from zotero_arxiv_daily import arxiv  # noqa: E402
from zotero_arxiv_daily.config import load_config  # noqa: E402
from zotero_arxiv_daily.email import render_email  # noqa: E402
from zotero_arxiv_daily.mailer import send_email  # noqa: E402
from zotero_arxiv_daily.teaser import generate_teaser, make_llm_client  # noqa: E402


def run(max_papers: int) -> None:
    config = load_config(REPO_ROOT / "config")
    logger.info(
        f"Retrieving arXiv papers for {config.executor.categories} "
        f"(lookback {config.executor.lookback_days}d, production path)..."
    )
    papers = arxiv.retrieve_papers(config, seen_keys=set())
    if not papers:
        raise RuntimeError("Smoke test found no arXiv papers")
    for index, paper in enumerate(papers, start=1):
        paper.score = float(len(papers) - index + 1)  # deterministic display order
    papers = papers[:max_papers]

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
