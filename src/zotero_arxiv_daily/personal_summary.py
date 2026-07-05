from typing import Any

from openai import OpenAI


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


def get_system_prompt(llm_params: Any) -> str:
    return get_summary_config(llm_params).get("system_prompt") or DEFAULT_SYSTEM_PROMPT


def paper_context(title: str, abstract: str, full_text: str | None) -> str:
    parts = [f"題目：{title or '[來源缺失]'}", f"摘要：{abstract or '[來源缺失]'}"]
    if full_text:
        parts.append(f"正文預覽：{full_text}")
    return "\n\n".join(parts)


def chat(openai_client: OpenAI, llm_params: Any, system: str, prompt: str) -> str:
    response = openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **llm_params.get("generation_kwargs", {}),
    )
    if llm_params.get("generation_kwargs", {}).get("stream"):
        chunks = []
        for chunk in response:
            if not getattr(chunk, "choices", None):
                continue
            if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content is not None:
                chunks.append(content)
        return "".join(chunks)
    return response.choices[0].message.content


def generate_teaser(
    openai_client: OpenAI,
    llm_params: Any,
    title: str,
    abstract: str,
    full_text: str | None,
) -> str:
    limit = get_teaser_char_limit(llm_params)
    prompt = (
        "你是嚴謹的學術每日摘要編輯。請用繁體中文寫一段極短速覽。\n\n"
        "要求：\n"
        f"- 最多 {limit} 個中文字。\n"
        "- 只說：研究問題、核心方法或新意、為什麼值得看。\n"
        "- 不要誇大，不要加入輸入中沒有的結論。\n"
        "- 不要使用 Markdown。\n"
        "- 只輸出摘要文字。\n\n"
        f"可用資料：\n{paper_context(title, abstract, full_text)}"
    )
    teaser = (chat(openai_client, llm_params, "你是一個精簡的學術摘要專家。", prompt) or "").strip()
    return teaser[:limit].rstrip() if len(teaser) > limit else teaser


def generate_deep_digest(
    openai_client: OpenAI,
    llm_params: Any,
    title: str,
    abstract: str,
    full_text: str | None,
) -> str:
    prompt = (
        f"請為《{title or '[來源不足]'}》產生「深度速覽」。\n\n"
        f"可用資料：\n{paper_context(title, abstract, full_text)}\n\n"
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
