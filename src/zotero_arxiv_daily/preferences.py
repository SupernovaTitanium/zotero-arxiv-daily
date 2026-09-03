"""Topic-level preference keywords applied to the ranking.

``preferences.yaml`` holds two keyword lists::

    boost:            # see MORE of this
      - verifiable rewards
    mute:             # see LESS of this
      - text-to-image safety

The file is written weekly by the preference-review workflow, which infers the
keywords from the implicit feedback: recommendations the user saved to Zotero
(positive) versus recommendations the user ignored (negative). No manual
paper-by-paper clicking anywhere.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

REVIEW_MIN_PAPERS = 5
MAX_KEYWORDS = 8


@dataclass
class Preferences:
    boost: list[str] = field(default_factory=list)
    mute: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.boost and not self.mute


def load_preferences(path: str | Path | None) -> Preferences:
    if not path:
        return Preferences()
    path = Path(path)
    if not path.exists():
        return Preferences()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        boost = [str(k).strip() for k in (data.get("boost") or []) if str(k).strip()]
        mute = [str(k).strip() for k in (data.get("mute") or []) if str(k).strip()]
        return Preferences(boost=boost[:MAX_KEYWORDS], mute=mute[:MAX_KEYWORDS])
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"Ignoring unreadable preferences file {path}: {e}")
        return Preferences()


def save_preferences(path: str | Path, prefs: Preferences, generated_on: str, stats: dict) -> None:
    path = Path(path)
    body = yaml.safe_dump(
        {"boost": prefs.boost, "mute": prefs.mute},
        allow_unicode=True,
        sort_keys=False,
    )
    header = (
        f"# Topic-level preference keywords, generated {generated_on} by the weekly review.\n"
        f"# Evidence: {stats.get('saved', 0)} saved / {stats.get('ignored', 0)} ignored recommendations.\n"
        "# Edit freely (one keyword per line); the next weekly run overwrites this file.\n"
    )
    path.write_text(header + body, encoding="utf-8")


def apply_preferences(papers: list, prefs: Preferences, boost_weight: float, mute_weight: float) -> list:
    """Adjust paper scores by keyword hits and re-sort (descending). Papers
    without hits keep their embedding-based score unchanged."""
    if prefs.is_empty() or not papers:
        return papers
    for p in papers:
        text = f"{p.title or ''} {p.abstract or ''}".lower()
        boost_hits = sum(1 for k in prefs.boost if k.lower() in text)
        mute_hits = sum(1 for k in prefs.mute if k.lower() in text)
        if boost_hits or mute_hits:
            base = p.score if p.score is not None else 0.0
            p.score = base + boost_hits * boost_weight - mute_hits * mute_weight
    papers.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
    return papers


def filter_review_papers(
    presented_saved: list[dict], presented_ignored: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Drop degenerate entries (no title) and enforce the minimum evidence
    threshold: below it, no preference update is written."""
    saved = [e for e in presented_saved if e.get("title")]
    ignored = [e for e in presented_ignored if e.get("title")]
    return saved, ignored


def enough_evidence(saved: list[dict], ignored: list[dict]) -> bool:
    return len(saved) + len(ignored) >= REVIEW_MIN_PAPERS


def build_review_messages(saved: list[dict], ignored: list[dict], language: str) -> list[dict]:
    def _entry_block(entries: list[dict]) -> str:
        return "\n\n".join(
            f"- 題目:{e['title']}\n  摘要:{(e.get('abstract') or '')[:300]}"
            for e in entries
        ) or "(無)"

    prompt = (
        "以下是一個學術論文推薦系統過去幾週送達用戶信箱的論文,以及用戶的隱式回饋。\n"
        "【保存】= 用戶後來把論文存進 Zotero(喜歡的訊號)。\n"
        "【忽略】= 用戶始終沒有保存(不感興趣的訊號)。\n\n"
        f"== 保存({len(saved)} 篇)==\n{_entry_block(saved)}\n\n"
        f"== 忽略({len(ignored)} 篇)==\n{_entry_block(ignored)}\n\n"
        "請推斷用戶的主題級偏好,輸出 JSON 物件:\n"
        '{"boost": ["想看到更多的研究主題", ...], "mute": ["想看到更少的研究主題", ...]}\n\n'
        "規則:\n"
        f"- 使用 {language} 或領域通用英文術語,每項 2 到 6 個詞。\n"
        f"- boost 最多 {MAX_KEYWORDS} 項,來自保存論文的共同主題。\n"
        f"- mute 最多 {MAX_KEYWORDS} 項,來自忽略論文中、與保存論文無關的主題。\n"
        "- 主題要是可以被關鍵字比對的具體研究方向的短語,不要是「機器學習」這種過寬的詞。\n"
        "- 只輸出 JSON,不要其他文字。"
    )
    return [
        {"role": "system", "content": "你是嚴謹的學術閱讀偏好分析助手。"},
        {"role": "user", "content": prompt},
    ]


def parse_review_response(raw: str) -> Preferences:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in review response")
    data = json.loads(raw[start:end + 1])

    def _clean(values) -> list[str]:
        out = []
        for v in values or []:
            text = str(v).strip()
            if text:
                out.append(text)
        return out[:MAX_KEYWORDS]

    return Preferences(boost=_clean(data.get("boost")), mute=_clean(data.get("mute")))
