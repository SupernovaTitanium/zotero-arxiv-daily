import json
from typing import Any

from openai import OpenAI

from .llm import request_llm


DEEP_DIGEST_TITLE = "深度速覽"
SUMMARY_ANCHOR_ID = "super-summary"

DEFAULT_SYSTEM_PROMPT = (
    "你是嚴謹但好懂的數學與科研導讀助手。\n"
    "讀者是聰明、好奇、但離開學界一段時間的繁體中文讀者。\n\n"
    "規則：\n"
    "- 只根據提供的題目、摘要、正文預覽回答。\n"
    "- 不要編造頁碼、定理號、實驗結果或作者沒有說的主張。\n"
    "- 每個重要判斷標註來源類型： [題目]、[摘要]、[正文預覽]、[推論]。\n"
    "- 若資料不足，直接寫 [來源不足]。\n"
    "- 短句、清楚、可掃讀。\n"
)


def get_summary_config(llm_params: Any) -> Any:
    summary = llm_params.get("summary")
    return summary if hasattr(summary, "get") else {}


def get_summary_mode(llm_params: Any, default: str = "tldr") -> str:
    mode = str(get_summary_config(llm_params).get("mode", default)).lower()
    return mode if mode in {"teaser", "full", "legacy"} else "legacy"


def get_teaser_char_limit(llm_params: Any) -> int:
    try:
        return max(1, int(get_summary_config(llm_params).get("teaser_char_limit", 150)))
    except (TypeError, ValueError):
        return 150


def get_teaser_batch_size(llm_params: Any) -> int:
    try:
        return max(1, int(get_summary_config(llm_params).get("batch_size", 10)))
    except (TypeError, ValueError):
        return 1


def get_system_prompt(llm_params: Any) -> str:
    return get_summary_config(llm_params).get("system_prompt") or DEFAULT_SYSTEM_PROMPT


def get_language(llm_params: Any) -> str:
    return str(llm_params.get("language", "English"))


def paper_context(title: str, abstract: str, full_text: str | None) -> str:
    parts = [f"題目：{title or '[來源缺失]'}", f"摘要：{abstract or '[來源缺失]'}"]
    if full_text:
        parts.append(f"正文預覽：{full_text}")
    return "\n\n".join(parts)


def chat(openai_client: OpenAI, llm_params: Any, system: str, prompt: str) -> str:
    return request_llm(
        openai_client,
        llm_params,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def generate_teaser(
    openai_client: OpenAI,
    llm_params: Any,
    title: str,
    abstract: str,
    full_text: str | None,
) -> str:
    limit = get_teaser_char_limit(llm_params)
    language = get_language(llm_params)
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請寫一段極短速覽。\n\n"
        "要求：\n"
        f"- 使用 {language}。\n"
        f"- 最多 {limit} 個字。\n"
        "- 只說：研究問題、核心方法或新意、為什麼值得看。\n"
        "- 不要誇大，不要加入輸入中沒有的結論。\n"
        "- 不要使用 Markdown。\n"
        "- 只輸出摘要文字。\n\n"
        f"可用資料：\n{paper_context(_clip(title, 500), _clip(abstract, 3000), _clip(full_text or '', 3000) or None)}"
    )
    teaser = (chat(openai_client, llm_params, "你是一個精簡的學術摘要專家。", prompt) or "").strip()
    return teaser[:limit].rstrip() if len(teaser) > limit else teaser


def generate_teasers_batch(openai_client: OpenAI, llm_params: Any, papers: list) -> int:
    """Generate teasers for many papers with one LLM request per batch.

    Sets ``paper.teaser`` and ``paper.tldr`` for every paper. Papers missing
    from a malformed batch response fall back to one-by-one generation.
    Returns the number of LLM requests made.
    """
    batch_size = get_teaser_batch_size(llm_params)
    requests = 0
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        teasers = {}
        if batch_size > 1:
            requests += 1  # the API is called even if the response turns out unusable
            try:
                teasers = _teasers_for_batch(openai_client, llm_params, batch)
            except Exception as e:
                from loguru import logger
                logger.warning(f"Batch teaser generation failed ({e}); falling back to per-paper requests")
        for i, paper in enumerate(batch):
            text = teasers.get(i)
            if not text:
                text = generate_teaser(
                    openai_client, llm_params, paper.title, paper.abstract, paper.full_text
                )
                requests += 1
            paper.teaser = text
            paper.tldr = text
    return requests


def _teasers_for_batch(openai_client: OpenAI, llm_params: Any, batch: list) -> dict[int, str]:
    limit = get_teaser_char_limit(llm_params)
    language = get_language(llm_params)
    lines = []
    for i, paper in enumerate(batch):
        lines.append(f"[{i}] " + paper_context(
            _clip(paper.title, 500), _clip(paper.abstract or "", 2000), None
        ))
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請為下列每一篇論文各寫一段極短速覽。\n\n"
        "要求：\n"
        f"- 使用 {language}。\n"
        f"- 每段最多 {limit} 個字。\n"
        "- 每段只說：研究問題、核心方法或新意、為什麼值得看。\n"
        "- 不要誇大，不要加入輸入中沒有的結論。\n"
        "- 輸出 JSON 陣列，依論文編號排序，格式："
        '[{"index": 0, "teaser": "..."}, {"index": 1, "teaser": "..."}]\n'
        "- 只輸出 JSON，不要其他文字。\n\n"
        "論文清單：\n" + "\n\n".join(lines)
    )
    batch_params = {k: v for k, v in llm_params.items() if k != "summary"}
    generation_kwargs = dict(batch_params.get("generation_kwargs", {}))
    max_tokens = generation_kwargs.get("max_tokens")
    needed = len(batch) * (limit * 4) + 1000
    if max_tokens is None or max_tokens > needed:
        generation_kwargs["max_tokens"] = needed
    batch_params["generation_kwargs"] = generation_kwargs

    raw = chat(openai_client, batch_params, "你是一個精簡的學術摘要專家。", prompt)
    start, end = raw.find("["), raw.rfind("]")
    parsed = json.loads(raw[start:end + 1])
    if isinstance(parsed, dict):
        parsed = parsed.get("results", parsed.get("teasers", []))
    teasers = {}
    for item in parsed:
        idx = int(item["index"])
        text = str(item.get("teaser", "")).strip()
        if text:
            teasers[idx] = text[:limit].rstrip()
    return teasers


def generate_deep_digest(
    openai_client: OpenAI,
    llm_params: Any,
    title: str,
    abstract: str,
    full_text: str | None,
) -> str:
    language = get_language(llm_params)
    prompt = (
        f"請為《{title or '[來源不足]'}》產生「深度速覽」，輸出語言為 {language}。\n\n"
        f"可用資料：\n{paper_context(_clip(title, 500), _clip(abstract, 3000), _clip(full_text or '', 6000) or None)}\n\n"
        "輸出格式固定如下：\n\n"
        "## 一句話版\n"
        "用一句話說明本文解決什麼問題與核心新意。\n\n"
        "## 前置導覽\n"
        "列出 3 到 5 個理解本文最需要的術語。每項包含白話解釋與它在本文中的角色。\n\n"
        "## 核心故事\n"
        "用「問題 → 障礙 → 方法 → 結果 → 意義」五行說明。\n\n"
        "## 主要貢獻\n"
        "列出 2 到 4 點。每點標註 [摘要]、[正文預覽] 或 [推論]。\n\n"
        "## 可遷移技巧\n"
        "列出可借到其他研究的技巧，以及使用限制。\n\n"
        "## 弱點與風險\n"
        "列出資料中可看出的假設、缺口或不確定性。資料不足時寫 [來源不足]。\n\n"
        "## 新 idea\n"
        "提出一個受本文啟發的新研究 idea，包含：可行性、需要條件、最低成本驗證方式。"
    )
    return (chat(openai_client, llm_params, get_system_prompt(llm_params), prompt) or "[來源不足]").strip()
