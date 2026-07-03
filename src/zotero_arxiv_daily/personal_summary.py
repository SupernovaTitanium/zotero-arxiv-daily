from typing import Any

from openai import OpenAI


DEEP_DIGEST_TITLE = "深度速覽"
SUMMARY_ANCHOR_ID = "super-summary"

DEFAULT_SYSTEM_PROMPT = (
    "角色：你是壯年、嚴謹又帶童心的 ENTP 大數學家（希爾伯特 × 歐拉混合體）。\n"
    "受眾：全程對聰明好奇但離開學界多年的長輩讀者講解，語氣溫柔自信，"
    "開場先說出目前解說的論文名稱。\n"
    "語言：句子要短、概念要清楚，抽象術語立刻用生活化例子輔助；數學部分保持精煉嚴格。\n"
    "風格：ENTP 式機智、密集而結構化，勇於提出反例與反詰；遇到資訊不足必說「[來源缺失]」。\n"
    "證據：所有可驗證主張必附頁碼/圖表/命題等局部引用，格式如 [Lin 2025, Thm 2, p.14]，"
    "缺資料則標註 [來源缺失]，絕不臆測。\n"
    "創新：每篇都要提出一個受論文啟發的新 idea，說明可行性、需要的條件，以及如何收斂成有價值的學術工作。\n"
)

COMMON_OUTPUT_RULES = (
    "通用規則：\n"
    "- **極致精簡**：嚴格限制字數，能用短句就別用長句，能用詞語就別用句子。\n"
    "- 全程使用繁體中文。\n"
    "- 每個可驗證主張後附引用；若缺資料或無可查證來源，直接寫 [來源缺失]。\n"
    "- 使用 Markdown 結構保持可讀性。\n"
)

SECTION_SPECS = [
    ("prelude", "**前置導覽**", "挑選 3~6 個關鍵術語，格式：`- 術語：白話解釋；數學刻畫（引用）`。"),
    ("A", "**A. 總結敘事**", "依「問題→障礙→方法→結果→意義」順序，每點不超過 20 字。結尾一行 `**一句話版**：...`。"),
    ("B", "**B. 術語與縮寫表**", "Markdown 表格：名詞/縮寫 | 白話定義 | 角色 | 依賴。至少 3 筆。"),
    ("C", "**C. 嚴謹定義**", "對核心術語給出最小數學描述，附引用。最後 `**一句話版**：...`。"),
    ("D", "**D. 主要貢獻**", "列出貢獻，格式 `- 貢獻：... → 證據（引用）`。每項限 30 字。"),
    ("E", "**E. 文本分析**", "Markdown 表格：前人缺陷 | 本文改進 | 有效機制 | 證據頁碼。"),
    ("F", "**F. 可遷移技巧**", "條列可複用技巧與限制；提出一個新 idea，說明可行性與條件。"),
    ("G", "**G. 弱點與邏輯 Gap**", "列出脆弱假設、盲點、風險等，每項說明並附引用。"),
    ("QA", "### 連續發問檢核", "簡答：類似解法與新意、拒稿/接收理由、可學技巧、衍生方向。"),
]


def get_summary_config(llm_params: Any) -> Any:
    summary = llm_params.get("summary")
    return summary if hasattr(summary, "get") else {}


def get_summary_mode(llm_params: Any, default: str = "tldr") -> str:
    return str(get_summary_config(llm_params).get("mode", default)).lower()


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
    return response.choices[0].message.content


def generate_teaser(
    openai_client: OpenAI,
    llm_params: Any,
    title: str,
    abstract: str,
    full_text: str | None,
    abstract_only: bool = False,
) -> str:
    limit = get_teaser_char_limit(llm_params)
    source = f"摘要：\n{abstract}" if abstract_only else paper_context(title, abstract, full_text)
    prompt = (
        "請用繁體中文寫一段極短學術速覽。\n"
        f"限制：嚴格限制在 {limit} 個中文字以內；只講問題與核心新意；只輸出總結文字。\n\n"
        f"{source}"
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
    context = paper_context(title, abstract, full_text)
    system = get_system_prompt(llm_params)
    blocks = []
    for _, heading, instruction in SECTION_SPECS:
        prompt = (
            f"目前你正在講解《{title or '[來源缺失]'}》，請產出「{heading}」。\n\n"
            f"【可用資訊】\n{context}\n\n"
            f"【任務】\n{instruction}\n\n"
            f"{COMMON_OUTPUT_RULES}"
        )
        body = (chat(openai_client, llm_params, system, prompt) or "[來源缺失]").strip()
        blocks.append(f"{heading}\n{body}")
    return "\n\n".join(blocks)
