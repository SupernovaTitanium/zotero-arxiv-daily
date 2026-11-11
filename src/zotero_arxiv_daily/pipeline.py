"""The nightly pipeline: Zotero corpus -> arXiv retrieval -> rank -> topics
-> teasers -> email -> dedup history. Each stage logs its timing;
the rendered email and a run summary are written before sending, and history
is only persisted after the email is delivered (a failed send must not swallow
that day's papers). A failing LLM endpoint does not stop the run: the papers
go out with their abstracts and a notice."""

from __future__ import annotations

import json
import platform
import time
from datetime import date
from pathlib import Path

from loguru import logger
from openai import OpenAIError

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


# Waits before asking the LLM again after its endpoint failed. The one such
# failure in 39 days of runs, a 401 from the proxy's account pool, was gone
# when the run was repeated 19 minutes later.
TEASER_RETRY_WAITS_SECONDS = (60, 300, 900)


def _write_teasers(config: Config, papers: list[Paper]) -> tuple[int, dict | None, str | None]:
    """Teasers for ``papers``. Returns the LLM requests made, the tokens the
    endpoint reported, and the error if the endpoint was still failing after
    the last wait. The error is returned, not raised: the papers are worth
    sending with their abstracts, and the email says what happened.

    An endpoint failure is an ``OpenAIError`` (connection, timeout, HTTP
    status) or a ``ValueError`` (a 200 response whose body is not JSON).
    Other exceptions are bugs in this code and end the run."""
    client = make_llm_client(config.llm)
    requests = 0
    error = None
    for wait in (*TEASER_RETRY_WAITS_SECONDS, None):
        try:
            requests += generate_teasers_batch(client, config.llm, [p for p in papers if not p.teaser])
            error = None
            break
        except (OpenAIError, ValueError) as e:
            error = config.redact(f"{type(e).__name__}: {e}")
            missing = sum(1 for p in papers if not p.teaser)
            if wait is None:
                logger.error(f"LLM endpoint still failing ({error}); {missing} papers go out without a teaser")
            else:
                logger.warning(f"LLM endpoint failed ({error}); {missing} papers lack a teaser, next attempt in {wait} s")
                time.sleep(wait)
    usage = getattr(client, "token_usage", None)
    if usage is None:
        return requests, None, error
    # The client also counts the requests of attempts that failed.
    return usage.requests, {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens}, error


def run(config: Config) -> None:
    timings: dict[str, float] = {}
    t_start = time.monotonic()

    # Loading the embedding model takes 14-28 s of imports and disk reads;
    # fetching the library and harvesting arXiv take 25-40 s of waiting on the
    # network. Neither needs the other, so the load starts first and runs in
    # the background.
    embedder = LocalEmbedder(config.embedding.model, config.embedding.encode_kwargs)
    embedder.preload()

    corpus = filter_corpus(fetch_corpus(config.zotero), config.zotero)
    timings["zotero"] = round(time.monotonic() - t_start, 1)
    if len(corpus) == 0:
        # A failure, not a quiet day: returning here made the run end like one
        # that found no new papers.
        raise RuntimeError(
            "No zotero papers found. Please check your Zotero credentials and collection filters locally."
        )

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
    llm_tokens = None
    teaser_error = None
    if all_papers:
        t_rerank = time.monotonic()
        logger.info("Reranking papers...")
        ranker = Ranker(config, embedder)
        reranked = ranker.rank(all_papers, corpus)
        prefs = load_preferences(config.executor.preferences_file)
        if not prefs.is_empty():
            reranked = ranker.apply_preferences(reranked, prefs)
        timings["rerank"] = round(time.monotonic() - t_rerank, 1)

        presented = reranked[: config.executor.max_paper_num]
        presented_ids = {id(p) for p in presented}
        t_topics = time.monotonic()
        ranker.assign_topics(presented)
        timings["topics"] = round(time.monotonic() - t_topics, 1)

        t_tldr = time.monotonic()
        logger.info("Generating teasers (batched)...")
        llm_requests, llm_tokens, teaser_error = _write_teasers(config, presented)
        timings["tldr"] = round(time.monotonic() - t_tldr, 1)
        logger.info(
            f"Teasers for {sum(1 for p in presented if p.teaser)}/{len(presented)} papers: "
            f"{llm_requests} LLM requests, tokens {llm_tokens}"
        )
    elif not config.executor.send_empty:
        logger.info(f"No new papers found. No email will be sent. ({timings})")
        return

    timings["total"] = round(time.monotonic() - t_start, 1)
    run_summary = {
        "date": date.today().isoformat(),
        "host": platform.node(),
        "timings_seconds": timings,
        "counts": {
            "corpus": len(corpus),
            "retrieved": len(all_papers),
            "ranked": len(reranked),
            "presented": len(presented),
        },
        "llm_requests": llm_requests,
        "llm_tokens": llm_tokens,
        "teaser_error": teaser_error,
        "papers": [
            {
                "rank": i + 1,
                "title": p.title,
                "source": p.source,
                "score": round(p.score, 4) if p.score is not None else None,
                "url": p.url,
                "presented": i < config.executor.max_paper_num,
                "topic": p.topic,
                "summary": (p.teaser or "")[:300],
            }
            for i, p in enumerate(reranked)
        ],
    }

    notice = None
    if teaser_error:
        missing = sum(1 for p in presented if not p.teaser)
        notice = f"今天有 {missing} 篇論文的速覽沒有產生，改列摘要開頭。LLM 服務的錯誤：{teaser_error[:300]}"

    logger.info("Sending email...")
    email_html = render_email(presented, config.llm.teaser_char_limit, notice, run_summary["host"])
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
        try:
            history.save()
        except OSError as e:
            raise RuntimeError(
                f"The email was sent, but the dedup history could not be saved ({e}); "
                "the next email may repeat today's papers"
            ) from e
        logger.info(f"Recorded {len(presented)}/{len(all_papers)} presented papers into {config.executor.state_file}")
