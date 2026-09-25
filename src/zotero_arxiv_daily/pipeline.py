"""The nightly pipeline: Zotero corpus -> arXiv retrieval -> rank -> topics
-> full text -> teasers -> email -> dedup history. Each stage logs its timing;
the rendered email and a run summary are written before sending, and history
is only persisted after the email is delivered (a failed send must not swallow
that day's papers)."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from . import arxiv
from .config import Config
from .email import render_email
from .embed import LocalEmbedder, Ranker
from .history import RecommendedHistory
from .mailer import send_email
from .paper import Paper
from .preferences import load_preferences
from .teaser import generate_teasers_batch, make_llm_client
from .zotero import fetch_corpus, filter_corpus


def _fetch_full_texts(config: Config, retriever_fn, papers: list[Paper]) -> None:
    """Two-stage pipeline: fetch full text only for the top ranked papers, in
    parallel. The rest keep metadata-only teasers from the abstract."""
    top_n = config.executor.fulltext_paper_num
    if top_n <= 0 or not papers:
        return
    targets = papers[:top_n]
    workers = max(1, config.executor.fulltext_workers)

    def _fetch(paper: Paper):
        return paper, retriever_fn(paper)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, p) for p in targets]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching full texts"):
            paper, full_text = future.result()
            if full_text:
                paper.full_text = full_text
    got = sum(1 for p in targets if p.full_text)
    logger.info(f"Full text fetched for {got}/{len(targets)} top papers")


def _write_outputs(output_dir: str | None, email_html: str, run_summary: dict) -> None:
    if not output_dir:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"email_{run_summary['date']}.html").write_text(email_html, encoding="utf-8")
    (out / f"run_summary_{run_summary['date']}.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Wrote run outputs to {out}")


def run(config: Config) -> None:
    timings: dict[str, float] = {}
    t_start = time.monotonic()

    corpus = filter_corpus(fetch_corpus(config.zotero), config.zotero)
    if len(corpus) == 0:
        logger.error(f"No zotero papers found. Please check your zotero settings: {config.zotero}")
        return

    # Keys of papers that must not be recommended: already in the Zotero
    # corpus, or already processed in a previous run.
    seen_keys: set[str] = set()
    for c in corpus:
        seen_keys.update(c.dedup_keys())
    history = RecommendedHistory.load(config.executor.state_file) if config.executor.state_file else None
    if history is not None:
        seen_keys.update(history.seen_keys())
    logger.info(f"Deduplicating against {len(seen_keys)} known keys")

    t_retrieve = time.monotonic()
    all_papers = arxiv.retrieve_papers(config, seen_keys)
    timings["retrieve"] = round(time.monotonic() - t_retrieve, 1)

    reranked: list[Paper] = []
    presented: list[Paper] = []
    presented_ids: set[int] = set()
    llm_requests = None
    if all_papers:
        t_rerank = time.monotonic()
        logger.info("Reranking papers...")
        ranker = Ranker(config, LocalEmbedder(config.embedding.model, config.embedding.encode_kwargs))
        reranked = ranker.rank(all_papers, corpus)
        prefs = load_preferences(config.executor.preferences_file)
        if not prefs.is_empty():
            reranked = ranker.apply_preferences(reranked, prefs)
        timings["rerank"] = round(time.monotonic() - t_rerank, 1)

        presented = reranked[: config.executor.max_paper_num]
        presented_ids = {id(p) for p in presented}
        ranker.assign_topics(presented)

        t_fulltext = time.monotonic()
        _fetch_full_texts(config, arxiv.fetch_full_text, presented)
        timings["fulltext"] = round(time.monotonic() - t_fulltext, 1)

        t_tldr = time.monotonic()
        logger.info("Generating teasers (batched)...")
        llm_requests = generate_teasers_batch(make_llm_client(config.llm), config.llm, presented)
        timings["tldr"] = round(time.monotonic() - t_tldr, 1)
    elif not config.executor.send_empty:
        logger.info("No new papers found. No email will be sent.")
        return

    timings["total"] = round(time.monotonic() - t_start, 1)
    run_summary = {
        "date": date.today().isoformat(),
        "timings_seconds": timings,
        "counts": {
            "corpus": len(corpus),
            "retrieved": len(all_papers),
            "ranked": len(reranked),
            "presented": len(presented),
        },
        "llm_requests": llm_requests,
        "papers": [
            {
                "rank": i + 1,
                "title": p.title,
                "source": p.source,
                "score": round(p.score, 4) if p.score is not None else None,
                "url": p.url,
                "presented": i < config.executor.max_paper_num,
                "has_full_text": bool(p.full_text),
                "topic": p.topic,
                "summary": (p.teaser or "")[:300],
            }
            for i, p in enumerate(reranked)
        ],
    }

    logger.info("Sending email...")
    email_html = render_email(presented, config.llm.teaser_char_limit)
    # Outputs are written before sending, so a failed send still leaves the
    # rendered email and full ranking behind for inspection.
    _write_outputs(config.executor.output_dir, email_html, run_summary)
    send_email(config.email, email_html)
    logger.info(f"Email sent successfully ({run_summary['counts']}, {timings})")

    # Only persist history after the email is delivered, so a failed run
    # does not silently swallow that day's papers.
    if history is not None:
        for p in all_papers:
            history.record_paper(
                p.dedup_keys(),
                title=p.title or "",
                abstract=p.abstract or "",
                presented=id(p) in presented_ids,
            )
        history.prune(config.executor.history_days)
        history.save()
        logger.info(f"Recorded {len(presented)}/{len(all_papers)} presented papers into {config.executor.state_file}")
