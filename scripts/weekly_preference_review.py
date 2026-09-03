"""Weekly preference review: infer topic-level boost/mute keywords from
implicit feedback and write preferences.yaml.

Positive signal = presented recommendations the user later saved to Zotero.
Negative signal = presented recommendations older than preference_grace_days
that are still absent from the Zotero corpus.

The script only writes the file when there is enough evidence; the workflow
commits it. Run: uv run python scripts/weekly_preference_review.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from zotero_arxiv_daily.executor import Executor  # noqa: E402
from zotero_arxiv_daily.llm import request_llm  # noqa: E402
from zotero_arxiv_daily.preferences import (  # noqa: E402
    build_review_messages,
    enough_evidence,
    filter_review_papers,
    parse_review_response,
    save_preferences,
)
from zotero_arxiv_daily.rate_limit import rate_limit_openai_client  # noqa: E402
from zotero_arxiv_daily.state import RecommendedHistory  # noqa: E402

PREFERENCES_PATH = REPO_ROOT / "preferences.yaml"


def _load_config():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        config = compose(config_name="default")
    OmegaConf.resolve(config)
    return config


def main() -> None:
    config = _load_config()

    state_file = config.executor.get("state_file", None)
    if not state_file or not Path(state_file).exists():
        logger.warning(f"No recommendation history at {state_file}; nothing to review")
        return

    history = RecommendedHistory.load(state_file)
    executor = Executor(config)
    corpus = executor.fetch_zotero_corpus()
    corpus_keys = set()
    for c in corpus:
        corpus_keys.update(c.dedup_keys())

    grace_days = int(config.executor.get("preference_grace_days", 5) or 5)
    cutoff = date.today() - timedelta(days=grace_days)
    saved, ignored = history.presented_not_saved(corpus_keys, cutoff)
    saved, ignored = filter_review_papers(saved, ignored)
    logger.info(f"Review window: {len(saved)} saved / {len(ignored)} ignored presented papers")

    if not enough_evidence(saved, ignored):
        logger.info(
            f"Not enough evidence (<{5} papers) to update preferences; keeping the existing file"
        )
        return

    language = config.llm.get("language", "English")
    messages = build_review_messages(saved, ignored, language)
    client = rate_limit_openai_client(
        executor.openai_client,
        config.llm.get("requests_per_minute", 10),
    )
    raw = request_llm(client, config.llm, messages)
    prefs = parse_review_response(raw)
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
