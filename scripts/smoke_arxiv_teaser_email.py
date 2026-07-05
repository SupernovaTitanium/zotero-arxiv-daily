"""Smoke-test arXiv retrieval, teaser generation, and email delivery.

This intentionally does not use the daily RSS feed. The daily production
workflow should stay quiet when there are no new papers, while this workflow
needs deterministic signal when run manually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arxiv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import rate_limit_chat_client
from zotero_arxiv_daily.personal_summary import generate_teaser
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.utils import send_email


def _load_config() -> DictConfig:
    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "config"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="default")


def _category_query(categories: list[str]) -> str:
    if not categories:
        raise ValueError("source.arxiv.category must contain at least one category")
    return " OR ".join(f"cat:{category}" for category in categories)


def _fetch_recent_arxiv_papers(config: DictConfig, max_papers: int) -> list[Paper]:
    query = _category_query(list(config.source.arxiv.category))
    logger.info(f"Fetching {max_papers} recent arXiv papers with query: {query}")
    client = arxiv.Client(num_retries=3, delay_seconds=5)
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for index, result in enumerate(client.results(search), start=1):
        papers.append(
            Paper(
                source="arxiv",
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                full_text=None,
                score=float(max_papers - index + 1),
            )
        )
    return papers


def run(max_papers: int) -> None:
    config = _load_config()
    OmegaConf.resolve(config)

    papers = _fetch_recent_arxiv_papers(config, max_papers)
    if not papers:
        raise RuntimeError("Smoke test found no arXiv papers")

    openai_client = rate_limit_chat_client(
        OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url),
        config.llm.get("requests_per_minute", 10),
    )
    for paper in papers:
        logger.info(f"Generating teaser: {paper.title}")
        teaser = generate_teaser(openai_client, config.llm, paper.title, paper.abstract, paper.full_text)
        if not teaser:
            raise RuntimeError(f"Failed to generate teaser for {paper.url}")
        paper.teaser = teaser
        paper.tldr = teaser

    logger.info("Rendering and sending smoke-test email")
    email_content = render_email(papers, config.llm.summary)
    send_email(config, email_content)

    logger.info(f"Smoke email sent with {len(papers)} papers")
    for index, paper in enumerate(papers, start=1):
        print(f"[{index}] {paper.title}")
        print(f"    URL: {paper.url}")
        print(f"    Teaser: {paper.tldr}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=3)
    args = parser.parse_args()
    if args.max_papers <= 0:
        raise ValueError("--max-papers must be positive")
    run(args.max_papers)


if __name__ == "__main__":
    main()
