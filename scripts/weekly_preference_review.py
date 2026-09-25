"""Weekly preference review: infer topic-level boost/mute keywords from
implicit feedback and write preferences.yaml.

Positive signal = presented recommendations the user later saved to Zotero.
Negative signal = presented recommendations older than preference_grace_days
that are still absent from the Zotero corpus.

The script only writes the file when there is enough evidence; the workflow
commits it. Run: uv run python scripts/weekly_preference_review.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from loguru import logger  # noqa: E402

from zotero_arxiv_daily.config import load_config  # noqa: E402
from zotero_arxiv_daily.history import RecommendedHistory  # noqa: E402
from zotero_arxiv_daily.preferences import (  # noqa: E402
    build_review_messages,
    enough_evidence,
    filter_review_papers,
    parse_review_response,
    save_preferences,
)
from zotero_arxiv_daily.rate_limit import rate_limit_openai_client  # noqa: E402
from zotero_arxiv_daily.teaser import make_llm_client  # noqa: E402
from zotero_arxiv_daily.zotero import fetch_corpus  # noqa: E402

PREFERENCES_PATH = REPO_ROOT / "preferences.yaml"


def main() -> None:
    os.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "config")

    state_file = config.executor.state_file
    if not state_file or not Path(state_file).exists():
        logger.warning(f"No recommendation history at {state_file}; nothing to review")
        return

    history = RecommendedHistory.load(state_file)
    corpus = fetch_corpus(config.zotero)
    corpus_keys: set[str] = set()
    for c in corpus:
        corpus_keys.update(c.dedup_keys())

    cutoff = date.today() - timedelta(days=config.executor.preference_grace_days)
    saved, ignored = history.presented_not_saved(corpus_keys, cutoff)
    saved, ignored = filter_review_papers(saved, ignored)
    logger.info(f"Review window: {len(saved)} saved / {len(ignored)} ignored presented papers")

    if not enough_evidence(saved, ignored):
        logger.info("Not enough evidence (<5 papers) to update preferences; keeping the existing file")
        return

    client = rate_limit_openai_client(
        make_llm_client(config.llm),
        config.llm.requests_per_minute,
    )

    def _chat(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            messages=messages, model=config.llm.model, max_tokens=config.llm.max_tokens
        )
        return response.choices[0].message.content or ""

    prefs = parse_review_response(_chat(build_review_messages(saved, ignored, config.llm.language)))
    if prefs.is_empty():
        logger.warning("Review produced empty preferences; keeping the existing file")
        return

    save_preferences(
        PREFERENCES_PATH,
        prefs,
        generated_on=datetime.now().strftime("%Y-%m-%d"),
        stats={"saved": len(saved), "ignored": len(ignored)},
    )
    logger.info(f"Wrote {PREFERENCES_PATH}: boost={prefs.boost} mute={prefs.mute}")


if __name__ == "__main__":
    main()
