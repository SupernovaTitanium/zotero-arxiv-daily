"""Batched teaser generation: one LLM request per batch of papers, falling
back to per-paper requests for papers missing from a malformed response."""

from __future__ import annotations

import json

from loguru import logger
from openai import OpenAI

from .config import LlmConfig
from .rate_limit import rate_limit_openai_client


def make_llm_client(llm: LlmConfig) -> OpenAI:
    """One rate-limited client per run: the limiter's adaptive minimum-interval
    state must be shared across all calls, so never build a client per request."""
    return rate_limit_openai_client(
        OpenAI(api_key=llm.api_key, base_url=llm.base_url),
        llm.requests_per_minute,
        max_retries=llm.rate_limit_max_retries,
        backoff_seconds=llm.rate_limit_backoff_seconds,
        max_interval_seconds=llm.rate_limit_max_interval_seconds,
    )


def _chat(client: OpenAI, llm: LlmConfig, system: str, prompt: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model=llm.model,
        max_tokens=llm.max_tokens,
    )
    return response.choices[0].message.content or ""


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _paper_context(title: str, abstract: str, full_text: str | None) -> str:
    parts = [f"題目：{title or '[來源缺失]'}", f"摘要：{abstract or '[來源缺失]'}"]
    if full_text:
        parts.append(f"正文預覽：{full_text}")
    return "\n\n".join(parts)


def generate_teaser(client: OpenAI, llm: LlmConfig, title: str, abstract: str, full_text: str | None) -> str:
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請寫一段極短速覽。\n\n"
        "要求：\n"
        f"- 使用 {llm.language}。\n"
        f"- 最多 {llm.teaser_char_limit} 個字。\n"
        "- 只說：研究問題、核心方法或新意、為什麼值得看。\n"
        "- 不要誇大，不要加入輸入中沒有的結論。\n"
        "- 不要使用 Markdown。\n"
        "- 只輸出摘要文字。\n\n"
        f"可用資料：\n"
        f"{_paper_context(_clip(title, 500), _clip(abstract, 3000), _clip(full_text or '', 3000) or None)}"
    )
    teaser = _chat(client, llm, "你是一個精簡的學術摘要專家。", prompt).strip()
    return teaser[: llm.teaser_char_limit].rstrip()


def _teasers_for_batch(client: OpenAI, llm: LlmConfig, batch: list) -> dict[int, str]:
    lines = [
        f"[{i}] " + _paper_context(_clip(p.title, 500), _clip(p.abstract or "", 2000), None)
        for i, p in enumerate(batch)
    ]
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請為下列每一篇論文各寫一段極短速覽。\n\n"
        "要求：\n"
        f"- 使用 {llm.language}。\n"
        f"- 每段最多 {llm.teaser_char_limit} 個字。\n"
        "- 每段只說：研究問題、核心方法或新意、為什麼值得看。\n"
        "- 不要誇大，不要加入輸入中沒有的結論。\n"
        "- 輸出 JSON 陣列，依論文編號排序，格式："
        '[{"index": 0, "teaser": "..."}, {"index": 1, "teaser": "..."}]\n'
        "- 只輸出 JSON，不要其他文字。\n\n"
        "論文清單：\n" + "\n\n".join(lines)
    )
    raw = _chat(client, llm, "你是一個精簡的學術摘要專家。", prompt)
    start, end = raw.find("["), raw.rfind("]")
    parsed = json.loads(raw[start : end + 1])
    if isinstance(parsed, dict):
        parsed = parsed.get("results", parsed.get("teasers", []))
    teasers = {}
    for item in parsed:
        text = str(item.get("teaser", "")).strip()
        if text:
            teasers[int(item["index"])] = text[: llm.teaser_char_limit].rstrip()
    return teasers


def generate_teasers_batch(client: OpenAI, llm: LlmConfig, papers: list) -> int:
    """Generate teasers for many papers with one LLM request per batch.

    Sets ``paper.teaser`` for every paper; papers missing from a malformed
    batch response fall back to one-by-one generation. Returns the number of
    LLM requests made."""
    batch_size = max(1, llm.teaser_batch_size)
    requests = 0
    for start in range(0, len(papers), batch_size):
        batch = papers[start : start + batch_size]
        teasers = {}
        if batch_size > 1:
            requests += 1  # the API is called even if the response turns out unusable
            try:
                teasers = _teasers_for_batch(client, llm, batch)
            except Exception as e:
                logger.warning(f"Batch teaser generation failed ({e}); falling back to per-paper requests")
        for i, paper in enumerate(batch):
            text = teasers.get(i)
            if not text:
                text = generate_teaser(client, llm, paper.title, paper.abstract, paper.full_text)
                requests += 1
            paper.teaser = text
    return requests
