"""Render a preview email without Zotero, network full-text fetches, or SMTP.

- Zotero corpus is faked in-process.
- arXiv retrieval is real (metadata only).
- Embeddings are stubbed with deterministic hash vectors; the LLM is stubbed
  with canned Traditional-Chinese teasers, so no API key is needed.

Usage: uv run python scripts/preview_email.py [--max-papers 10]
Writes output/email_YYYY-MM-DD.html + run_summary JSON and prints the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import zotero_arxiv_daily.teaser as teaser_module  # noqa: E402
from loguru import logger  # noqa: E402

# Dummy credentials so load_config passes; nothing external is contacted with
# them (the LLM client is stubbed below and no email is sent).
os.environ.setdefault("ZOTERO_ID", "0")
os.environ.setdefault("ZOTERO_KEY", "preview")
os.environ.setdefault("SENDER", "preview@example.com")
os.environ.setdefault("RECEIVER", "preview@example.com")
os.environ.setdefault("SENDER_PASSWORD", "preview")
os.environ.setdefault("OPENAI_API_KEY", "sk-preview")
os.environ.setdefault("OPENAI_API_BASE", "https://localhost/v1")
os.environ.setdefault("EMAIL_SMTP_SERVER", "localhost")
os.environ.setdefault("EMAIL_SMTP_PORT", "465")

from zotero_arxiv_daily import arxiv  # noqa: E402
from zotero_arxiv_daily.config import load_config  # noqa: E402
from zotero_arxiv_daily.email import render_email  # noqa: E402
from zotero_arxiv_daily.embed import Ranker  # noqa: E402
from zotero_arxiv_daily.paper import CorpusPaper  # noqa: E402
from zotero_arxiv_daily.teaser import generate_teasers_batch  # noqa: E402


class FakeEmbedder:
    """Deterministic bag-of-token-hash embeddings; no model download."""

    def model_cache_key(self) -> str:
        return "preview|fake"

    def embed(self, texts: list[str]):
        vectors = []
        for text in texts:
            vec = [0.0] * 64
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                vec[digest[0] % 64] += 1.0
                vec[digest[1] % 64] += 0.5
            vectors.append(vec)
        return vectors


def _fake_corpus() -> list[CorpusPaper]:
    base = date.today() - timedelta(days=30)
    papers = [
        (
            "Vision-Language Models for Robotic Manipulation: A Survey",
            "We survey vision-language-action models that map visual observations and "
            "natural-language instructions to robot actions, covering architectures, "
            "training data, and evaluation benchmarks.",
        ),
        (
            "Diffusion Models for Text-to-Image Generation: A Survey",
            "This survey reviews denoising diffusion models for text-to-image synthesis, "
            "including classifier-free guidance, latent diffusion, and alignment techniques.",
        ),
        (
            "Scaling Laws for Sparse Mixture-of-Experts Language Models",
            "We study scaling laws for sparse mixture-of-experts transformers and show "
            "routing regularization improves token balance and downstream accuracy.",
        ),
    ]
    return [
        CorpusPaper(
            title=title,
            abstract=abstract,
            added_date=datetime(*(base - timedelta(days=7 * i)).timetuple()[:3]),
            paths=["2026/survey"] if i < 2 else ["2026/reading-group"],
        )
        for i, (title, abstract) in enumerate(papers)
    ]


def _stub_chat(client, llm, system: str, prompt: str) -> str:
    """Return a canned teaser: a JSON array for batch prompts, plain text otherwise."""
    if "輸出 JSON 陣列" in prompt:
        count = prompt.count("\n[")
        items = [
            {"index": i, "teaser": f"預覽速覽 {i + 1}：這是測試用的固定摘要文字。"}
            for i in range(count)
        ]
        return json.dumps(items, ensure_ascii=False)
    return "預覽速覽：這是測試用的固定摘要文字。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=10)
    args = parser.parse_args()

    config = load_config(REPO_ROOT / "config")
    config.executor.max_paper_num = args.max_papers
    config.executor.fulltext_paper_num = 0
    config.executor.output_dir = str(REPO_ROOT / "output")
    config.executor.preferences_file = None

    corpus = _fake_corpus()
    seen_keys: set[str] = set()
    for c in corpus:
        seen_keys.update(c.dedup_keys())

    logger.info("Retrieving arXiv papers (real, metadata only)...")
    papers = arxiv.retrieve_papers(config, seen_keys)
    if not papers:
        raise SystemExit("No arXiv papers retrieved; check the category list / network")

    ranker = Ranker(config, FakeEmbedder())  # type: ignore[arg-type]
    papers = ranker.rank(papers, corpus)[: args.max_papers]
    ranker.assign_topics(papers)

    original_chat = teaser_module._chat
    teaser_module._chat = _stub_chat
    try:
        llm_requests = generate_teasers_batch(object(), config.llm, papers)
    finally:
        teaser_module._chat = original_chat

    email_html = render_email(papers, config.llm.teaser_char_limit)
    out_dir = Path(config.executor.output_dir)
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / f"email_{date.today().isoformat()}.html"
    html_path.write_text(email_html, encoding="utf-8")
    (out_dir / f"run_summary_{date.today().isoformat()}.json").write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "mode": "preview",
                "llm_requests": llm_requests,
                "papers": [
                    {"rank": i + 1, "title": p.title, "score": round(p.score or 0, 3), "topic": p.topic}
                    for i, p in enumerate(papers)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Preview email written to {html_path}")


if __name__ == "__main__":
    main()
