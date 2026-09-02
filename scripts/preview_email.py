"""Render a preview email without Zotero, network full-text fetches, or SMTP.

- Zotero corpus is faked in-process.
- arXiv retrieval is real (metadata only; fulltext_paper_num=0).
- The reranker and LLM are stubbed (deterministic embeddings; canned
  Traditional-Chinese teasers), so no API key is needed.

Usage: uv run python scripts/preview_email.py [--max-papers 10]
Writes output/email_YYYY-MM-DD.html + run_summary JSON and prints the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf

import zotero_arxiv_daily.executor as executor_module
from zotero_arxiv_daily.personal_summary import generate_teaser, generate_teasers_batch

# ---------------------------------------------------------------------------
# Fake Zotero corpus
# ---------------------------------------------------------------------------

_FAKE_COLLECTIONS = [
    {"key": "C1", "data": {"name": "2026", "parentCollection": False}},
    {"key": "C2", "data": {"name": "survey", "parentCollection": "C1"}},
]

_FAKE_ITEMS = [
    {
        "data": {
            "title": t,
            "abstractNote": a,
            "dateAdded": f"2026-0{m}-{d:02d}T10:00:00Z",
            "collections": ["C2"],
        }
    }
    for t, a, m, d in [
        (
            "Vision-Language Models for Robotic Manipulation: A Survey",
            "We survey vision-language-action models that map visual observations and natural-language instructions to robot actions, covering architectures, training data, and evaluation benchmarks.",
            6, 2,
        ),
        (
            "Diffusion Models for Text-to-Image Generation: A Survey",
            "This survey reviews denoising diffusion models for text-to-image synthesis, including classifier-free guidance, latent diffusion, and alignment techniques.",
            7, 5,
        ),
        (
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP",
            "We study retrieval-augmented generation, which combines a dense retriever with a seq2seq generator to ground LLM outputs in external knowledge and reduce hallucination.",
            8, 1,
        ),
        (
            "Parameter-Efficient Fine-Tuning of Large Language Models",
            "We review parameter-efficient fine-tuning methods such as LoRA, adapters, and prompt tuning that update a tiny fraction of parameters while matching full fine-tuning quality.",
            8, 8,
        ),
        (
            "Self-Supervised Learning on Graphs: A Survey",
            "This survey covers contrastive, generative, and masked self-supervised objectives for graph representation learning across molecules, social networks, and knowledge graphs.",
            8, 15,
        ),
        (
            "Multimodal Chain-of-Thought Reasoning in LLMs",
            "We propose multimodal chain-of-thought prompting that interleaves images and text rationales, improving visual question answering and embodied planning.",
            8, 22,
        ),
        (
            "Efficient Inference for Large Language Models: A Survey",
            "We survey inference efficiency techniques for LLMs: KV-cache compression, speculative decoding, quantization, and batching strategies.",
            8, 28,
        ),
        (
            "Reinforcement Learning from Human Feedback: Foundations",
            "We formalize RLHF as a two-stage pipeline of reward modeling and policy optimization, analyzing reward hacking and KL regularization.",
            9, 1,
        ),
    ]
]


def _install_fake_zotero() -> None:
    stub = SimpleNamespace(
        everything=lambda gen: gen,
        collections=lambda: _FAKE_COLLECTIONS,
        items=lambda **kw: _FAKE_ITEMS,
    )
    executor_module.zotero = SimpleNamespace(Zotero=lambda *a, **kw: stub)
    executor_module.send_email = lambda config, html: logger.info("SMTP skipped (preview mode)")


# ---------------------------------------------------------------------------
# Stub LLM client: JSON-batch teasers + deterministic hash embeddings
# ---------------------------------------------------------------------------

def _stub_create(**kwargs):
    messages = kwargs.get("messages", [])
    text = str(messages)
    if "輸出 JSON 陣列" in text or "输出 JSON" in text:
        import re
        indexes = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
        payload = [
            {"index": i, "teaser": f"這是第 {i} 篇論文的測試速覽：提出一個新方法解決既有基準上的痛點，並展示顯著提升。"}
            for i in indexes
        ]
        content = json.dumps(payload, ensure_ascii=False)
    else:
        content = "單篇測試速覽：提出新方法解決既有基準的痛點，成效顯著且值得關注。"
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _hash_vector(text: str, dim: int = 16):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [b / 255.0 - 0.5 for b in digest[:dim]]
    return vals


def _install_stub_openai() -> None:
    stub = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_stub_create)),
        embeddings=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                data=[
                    SimpleNamespace(embedding=_hash_vector(str(t)), index=i, object="embedding")
                    for i, t in enumerate(kw.get("input", []))
                ],
                model=kw.get("model", "stub"),
                object="list",
            )
        ),
    )
    executor_module.OpenAI = lambda **kw: stub
    import zotero_arxiv_daily.reranker.api as api_module
    api_module.OpenAI = lambda **kw: stub


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(max_papers: int):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "config"), version_base=None):
        return compose(
            config_name="default",
            overrides=[
                "zotero.user_id=0",
                "zotero.api_key=fake",
                "zotero.include_path=null",
                "zotero.ignore_path=null",
                "email.sender=preview@example.com",
                "email.receiver=preview@example.com",
                "email.smtp_server=localhost",
                "email.smtp_port=1025",
                "email.sender_password=fake",
                "llm.api.key=sk-fake",
                "llm.api.base_url=http://localhost:9/v1",
                "llm.generation_kwargs.model=preview-model",
                "llm.requests_per_minute=0",
                "llm.language=Traditional Chinese",
                "llm.summary.mode=teaser",
                "llm.summary.teaser_char_limit=150",
                "llm.summary.batch_size=5",
                "source.arxiv.category=[cs.AI,cs.CV,cs.LG,cs.CL]",
                "executor.source=[arxiv]",
                "executor.reranker=api",
                "executor.debug=false",
                "executor.send_empty=true",
                "executor.lookback_days=1",
                "executor.state_file=null",
                "executor.output_dir=output",
                f"executor.max_paper_num={max_papers}",
                "executor.fulltext_paper_num=0",
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-papers", type=int, default=10)
    args = parser.parse_args()

    _install_fake_zotero()
    _install_stub_openai()

    config = _load_config(args.max_papers)
    OmegaConf.resolve(config)
    # null fields in base.yaml cannot be overridden from the CLI
    config.reranker.api.key = "sk-fake"
    config.reranker.api.base_url = "http://localhost:9/v1"
    config.reranker.api.model = "preview-embedding"

    executor = executor_module.Executor(config)
    executor.run()

    email_path = Path("output") / f"email_{__import__('datetime').date.today().isoformat()}.html"
    summary_path = Path("output") / f"run_summary_{__import__('datetime').date.today().isoformat()}.json"
    print("\n" + "=" * 60)
    print(f"Email HTML : {email_path.resolve()}")
    print(f"Run summary: {summary_path.resolve()}")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"Counts     : {summary['counts']}")
        print(f"LLM reqs   : {summary['llm_requests']}")
        print(f"Timings    : {summary['timings_seconds']}")
        for p in summary["papers"][:5]:
            print(f"  #{p['rank']} [{p['score']}] {p['title'][:60]}")


if __name__ == "__main__":
    main()
