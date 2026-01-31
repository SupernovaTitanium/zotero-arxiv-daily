# -*- coding: utf-8 -*-
"""
paper.py — 適配新版 llm.py（支援 temperature），並將 TLDR 升級為
「姥姥速覽」格式：前置導覽＋A~G 七個章節，以及連續發問檢核，
涵蓋摘要敘事、術語表、嚴謹定義、貢獻、對照分析、可遷移技巧、
弱點/推理缺口與 QA checklist。輸出預設為 Markdown，render_email
仍可直接轉成 HTML。

保留：
- affiliations 從 LaTeX 抽取
- code_url 從 Papers with Code
- 下載/拼接 .tex
- 與 render_email 相容（p.tldr 為純文字；p.tldr_markdown 供 Markdown→HTML）
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
import os
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken  # 僅用於 affiliations 提示截斷（保留原行為）
from contextlib import ExitStack
from urllib.error import HTTPError


# ------------------------------
# LaTeX / 文字處理輔助
# ------------------------------

_SEC_PAT = re.compile(r'\\section\*?\{([^}]*)\}', re.IGNORECASE)


def _latex_strip(s: str) -> str:
    """
    清掉 cite/ref/數學環境/圖表等，保留自然語言。
    """
    if not s:
        return ""
    s = re.sub(r'~?\\cite[t]?\{.*?\}', '', s)
    s = re.sub(r'\\ref\{.*?\}|\\eqref\{.*?\}', '', s)
    kill_envs = ["figure", "table", "algorithm", "lstlisting", "equation", "align", "gather", "multline"]
    for env in kill_envs:
        s = re.sub(rf'\\begin\{{{env}\}}.*?\\end\{{{env}\}}', '', s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r'\$[^$]*\$', '', s)  # 行內數學
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _pick_sections_from_tex(tex_all: str) -> Dict[str, str]:
    """
    從 LaTeX 主檔分割章節；支援 \section 與 \section*。
    回傳：{section_title_clean: section_body_clean}
    """
    sections: Dict[str, str] = {}
    if not tex_all:
        return sections
    positions = [(m.start(), m.group(1)) for m in _SEC_PAT.finditer(tex_all)]
    positions.append((len(tex_all), 'EOF'))
    for i in range(len(positions) - 1):
        start, name = positions[i]
        end, _ = positions[i + 1]
        sec_raw = tex_all[start:end]
        try:
            title = _latex_strip(name)
            body = _latex_strip(sec_raw)
            sections[title] = body
        except Exception:
            continue
    return sections


def _harvest_contrib_like(txt: str, limit: int = 10) -> str:
    """
    抽取貢獻語句（We propose / Our contributions / In this paper we ...）
    回傳串接字串，最多 limit 句。
    """
    spans = re.findall(
        r'([^.]*?(?:we\s+(?:propose|present|introduce)|our\s+contributions?|in\s+this\s+paper\s+we)[^.]*\.)',
        txt, flags=re.IGNORECASE)
    return ' '.join(spans[:limit])


def _normalize_text(value: Any) -> str:
    """
    將巢狀容器轉成可拼接的字串，避免 list/tuple 導致 join 失敗。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_normalize_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(_normalize_text(v) for v in value.values())
    return str(value)


def _value_or_missing(*values: Any) -> str:
    """
    依序回傳第一個有內容的字串，否則標註來源缺失。
    """
    for raw in values:
        text = _normalize_text(raw).strip()
        if text:
            return text
    return "[來源缺失]"


_DEFAULT_BASE_SYSTEM_PROMPT = (
    "角色：你是壯年、嚴謹又帶童心的 ENTP 大數學家（希爾伯特 × 歐拉混合體）。\n"
    "受眾：全程對 90 歲、聰明好奇但離開學界多年的姥姥講解，語氣溫柔自信，"
    "開場先說出目前解說的論文名稱。\n"
    "語言：句子要短、概念要清楚，抽象術語立刻用生活化例子輔助；數學部分保持精煉嚴格。\n"
    "風格：ENTP 式機智、密集而結構化，勇於提出反例與反詰；遇到資訊不足必說「[來源缺失]」。\n"
    "證據：所有可驗證主張必附頁碼/圖表/命題等局部引用，格式如 [Lin 2025, Thm 2, p.14]，"
    "缺資料則標註 [來源缺失]，絕不臆測。\n"
    "創新：每篇都要提出一個受論文啟發的新 idea，說明可行性、需要的條件，以及如何收斂成有價值的學術工作。\n"
)


def _base_system_prompt() -> str:
    """
    Returns the base system prompt; override via env `LLM_SYSTEM_PROMPT`.
    """
    return os.environ.get("LLM_SYSTEM_PROMPT", _DEFAULT_BASE_SYSTEM_PROMPT)


def _teaser_char_limit() -> int:
    try:
        return max(1, int(os.environ.get("TEASER_CHAR_LIMIT", "150")))
    except Exception:
        return 150


def _summary_mode() -> str:
    """
    FULL_SUMMARY:
      - "1"/true/on -> full sections
      - "-1" -> abstract-only teaser (no other LLM calls)
      - others -> teaser only
    """
    raw = os.environ.get("FULL_SUMMARY", "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return "full"
    if raw in {"-1"}:
        return "abstract"
    return "teaser"


def _full_summary_enabled() -> bool:
    return _summary_mode() == "full"

COMMON_OUTPUT_RULES = (
    "通用規則：\n"
    "- **極致精簡**：嚴格限制字數，能用短句就別用長句，能用詞語就別用句子。\n"
    "- 全程使用繁體中文。\n"
    "- 每個可驗證主張後附引用（例如 [Lin 2025, Thm 2, p.14] 或 [Fig.3, p.7]）。\n"
    "- 若缺資料或無可查證來源，直接寫 [來源缺失]。\n"
    "- 盡量使用 Markdown 結構（項目符號、表格、粗體標題）保持可讀性。\n"
)

SECTION_SPECS = [
    {
        "key": "prelude",
        "label": "前置導覽",
        "title": "**前置導覽**",
        "instruction": (
            "挑選 3~6 個關鍵術語，格式：`- 術語：白話解釋；數學刻畫（引用）`。"
            "解釋需極短。"
        ),
    },
    {
        "key": "A",
        "label": "A. 總結敘事",
        "title": "**A. 總結敘事**",
        "instruction": (
            "依「問題→障礙→方法→結果→意義」順序，每點不超過 20 字，逐句附引用。"
            "結尾一行 `**姥姥版一句話**：...` (限 30 字)。"
        ),
    },
    {
        "key": "B",
        "label": "B. 術語與縮寫表",
        "title": "**B. 術語與縮寫表**",
        "instruction": (
            "Markdown 表格：名詞/縮寫 | 白話定義 | 角色 | 依賴。"
            "至少 3 筆。表格後一行 `**姥姥版一句話**：...` (限 30 字)。"
        ),
    },
    {
        "key": "C",
        "label": "C. 嚴謹定義",
        "title": "**C. 嚴謹定義**",
        "instruction": (
            "對 B 節核心術語，給出最小數學描述，附引用。"
            "最後 `**姥姥版一句話**：...` (限 30 字)。"
        ),
    },
    {
        "key": "D",
        "label": "D. 主要貢獻",
        "title": "**D. 主要貢獻**",
        "instruction": (
            "列出貢獻，格式 `- 貢獻：... → 證據（引用）`。"
            "每項限 30 字。"
        ),
    },
    {
        "key": "E",
        "label": "E. 文本分析",
        "title": "**E. 文本分析**",
        "instruction": (
            "Markdown 表格：前人缺陷 | 本文改進 | 有效機制 | 證據頁碼。"
            "表格後補 1 句說明差異 (限 50 字)。"
        ),
    },
    {
        "key": "F",
        "label": "F. 可遷移技巧",
        "title": "**F. 可遷移技巧**",
        "instruction": (
            "條列可複用技巧與限制；"
            "提出一個新 idea，說明可行性與條件。"
            "最後 `**奶奶版一句話**：...` 與 `**生活化例子**：...` (各限 30 字)。"
        ),
    },
    {
        "key": "G",
        "label": "G. 弱點與邏輯 Gap",
        "title": "**G. 弱點與邏輯 Gap**",
        "instruction": (
            "列出脆弱假設、盲點、風險等，每項說明並附引用。"
            "每項限 30 字。"
        ),
    },
    {
        "key": "QA",
        "label": "連續發問檢核",
        "title": "### 連續發問檢核",
        "instruction": (
            "簡答四組問題：\n"
            "1) 類似問題解法與新意？\n"
            "2) 拒稿/接收理由 (各 2~3 點)。\n"
            "3) 可學到的 3 個技巧。\n"
            "4) 可衍生的 3 個方向。\n"
            "每點限 20 字，需附引用。"
        ),
    },
]

# ------------------------------
# ArxivPaper 類別
# ------------------------------

class ArxivPaper:
    """
    與原始專案相容：
    - render_email 會用到：title, authors, affiliations, arxiv_id, tldr, pdf_url, code_url, score
    - tldr：回傳純文字 TLDR（Markdown 也可讀）
    - tldr_markdown：更適合郵件/前端的 Markdown 版本
    - tldr_json：提供 structured_digest_markdown 供除錯或外部流程使用
    """

    def __init__(self, paper: arxiv.Result):
        self._paper = paper
        self.score = None  # 供外部評分用（render_email -> get_stars）

    # ---------------- 基本屬性 ----------------

    @property
    def title(self) -> str:
        return self._paper.title

    @property
    def summary(self) -> str:
        return self._paper.summary

    @property
    def authors(self) -> list[str]:
        # 原專案在 render_email 用 [a.name for a in p.authors]
        # 這裡維持回傳 arxiv 的作者物件列表（型別提示為 list[str] 僅為相容）
        return self._paper.authors

    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())

    @property
    def pdf_url(self) -> str:
        pdf_url = getattr(self._paper, "pdf_url", None)
        if pdf_url:
            return pdf_url

        pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        links = getattr(self._paper, "links", None)
        if links:
            first_link = links[0]
            href = getattr(first_link, "href", None)
            if href:
                pdf_url = href.replace('abs', 'pdf')

        # Cache the derived URL so downstream download calls succeed (Issue #119).
        self._paper.pdf_url = pdf_url
        return pdf_url
    
    @cached_property
    def code_url(self) -> Optional[str]:
        """
        從 Papers with Code API 取第一個 repository URL。
        """
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count', 0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count', 0) == 0:
            return None
        return repo_list['results'][0]['url']

    # ---------------- 下載與解析 LaTeX ----------------

    @cached_property
    def tex(self) -> Optional[dict[str, str]]:
        """
        下載 arXiv source（tar），擷取 .tex 內容；拼接 main tex 的 \input/\include。
        回傳 dict：{<filename>.tex: <content>, ..., "all": <main_merged_or_None>}
        若源檔不存在（404），回傳 None（後續流程會降級只用摘要）。
        """
        pdf_url = getattr(self._paper, "pdf_url", None)
        if not pdf_url:
            logger.warning(f"No pdf_url for {self.arxiv_id}. Skipping source download and falling back to abstract.")
            return None

        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            try:
                file = self._paper.download_source(dirpath=tmpdirname)
            except HTTPError as e:
                if e.code == 404:
                    logger.warning(f"Source for {self.arxiv_id} not found (404). Skipping source analysis.")
                    return None
                else:
                    logger.error(f"HTTP Error {e.code} when downloading source for {self.arxiv_id}: {e.reason}")
                    raise
            except AttributeError as e:
                logger.warning(
                    f"Download source failed for {self.arxiv_id} because pdf_url is missing: {e}. "
                    "Falling back to abstract-only processing."
                )
                return None
            except Exception as e:
                logger.error(f"Error when downloading source for {self.arxiv_id}: {e}")
                return None
            try:
                tar = stack.enter_context(tarfile.open(file))
            except tarfile.ReadError:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file.")
                return None

            tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None

            bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
            match len(bbl_file):
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: multiple tex files but no bbl.")
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl', '')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: bbl does not match any tex.")
                        main_tex = None
                case _:
                    logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: multiple bbl files.")
                    main_tex = None

            if main_tex is None:
                logger.debug(f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")

            file_contents: Dict[str, str] = {}
            for t in tex_files:
                fobj = tar.extractfile(t)
                if fobj is None:
                    continue
                content = fobj.read().decode('utf-8', errors='ignore')
                # 去註解與壓平
                content = re.sub(r'%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                content = re.sub(r'\n+', '\n', content)
                content = re.sub(r'\\\\', '', content)
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content

            if main_tex is not None:
                main_source: str = file_contents[main_tex]
                include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                for f in include_files:
                    if not f.endswith('.tex'):
                        file_name = f + '.tex'
                    else:
                        file_name = f
                    main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                    main_source = main_source.replace(f'\\include{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
            else:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                file_contents["all"] = None
        return file_contents

    # ---------------- 內部：TLDR 上下文彙整 ----------------

    @cached_property
    def _ctx_for_tldr(self) -> Dict[str, str]:
        """
        彙整上下文：標題、摘要、章節（若有）與貢獻句群。
        """
        intro = method = expts = limits = concl = contrib_spans = ""
        if self.tex is not None:
            content = self.tex.get("all") if isinstance(self.tex, dict) else None
            if content is None and isinstance(self.tex, dict):
                try:
                    content = "\n".join([v for v in self.tex.values() if isinstance(v, str)])
                except Exception:
                    content = None
            if isinstance(self.tex, str) and not content:
                content = self.tex

            if content:
                content_clean = _latex_strip(content)
                sections = _pick_sections_from_tex(content_clean)
                joined = ' '.join([k + ': ' + v for k, v in sections.items()]) if sections else content_clean

                def pick(keys: List[str]) -> str:
                    buf = []
                    if sections:
                        for name, body in sections.items():
                            for k in keys:
                                if re.search(k, name, re.IGNORECASE):
                                    buf.append(f"{name}\n{body}")
                                    break
                    return '\n'.join(buf[:3])

                intro = pick([r'Intro|Background|Overview|Motivation'])
                method = pick([r'Method|Approach|Algorithm|Model'])
                expts = pick([r'Experiments|Results|Evaluation'])
                limits = pick([r'Limitations|Discussion|Threats'])
                concl = pick([r'Conclusion|Summary'])
                contrib_spans = _harvest_contrib_like(joined, limit=10)

        return {
            "title": self.title or "",
            "abstract": self.summary or "",
            "intro": intro,
            "method": method,
            "expts": expts,
            "limits": limits,
            "concl": concl,
            "contrib_spans": contrib_spans,
        }

    # ---------------- 結構化 digest 生成 ----------------

    def _compose_digest_input_block(self) -> str:
        ctx = self._ctx_for_tldr
        authors = []
        for author in self.authors:
            name = getattr(author, "name", None) or str(author)
            name = (name or "").strip()
            if name:
                authors.append(name)
        authors_text = ", ".join(authors)
        categories = getattr(self._paper, "categories", None)
        cat_text = ", ".join(categories) if categories else ""
        venue = _value_or_missing(
            getattr(self._paper, "journal_ref", ""),
            getattr(self._paper, "comment", ""),
            getattr(self._paper, "primary_category", ""),
            cat_text,
        )
        published = getattr(self._paper, "published", None)
        year_value = getattr(published, "year", None) if published is not None else None
        year = str(year_value) if year_value else "[來源缺失]"
        meta_line = f"{_value_or_missing(self.title)}，{_value_or_missing(authors_text)}，{venue}，{year}"
        entries = [
            ("題目／作者／年份／會議期刊", meta_line),
            ("核心問題與動機", _value_or_missing(ctx.get("intro"), ctx.get("abstract"))),
            ("假設與設定", _value_or_missing(ctx.get("limits"), ctx.get("intro"))),
            ("方法重點", _value_or_missing(ctx.get("method"), ctx.get("abstract"))),
            ("數據與評測", _value_or_missing(ctx.get("expts"), ctx.get("method"))),
            ("定理/引理/命題", "[來源缺失]"),
            ("圖表與證據", "[來源缺失]"),
            ("原始碼/資源", _value_or_missing(self.code_url)),
        ]
        lines = []
        for label, raw in entries:
            text = _normalize_text(raw).strip()
            if not text:
                text = "[來源缺失]"
            text = text.replace("```", "'").replace("`", "'")
            if "\n" in text:
                lines.append(f"* {label}：\n```text\n{text}\n```")
            else:
                lines.append(f"* {label}：`{text}`")
        return "\n".join(lines)

    def _build_section_prompt(self, section_spec: Dict[str, str]) -> str:
        inputs_block = self._compose_digest_input_block()
        label = section_spec["label"]
        return (
            f"目前你正在向 90 歲聰明的姥姥講解《{_value_or_missing(self.title)}》，"
            f"請專注產出「{label}」部分。\n\n"
            f"【可用資訊】\n{inputs_block}\n\n"
            f"【該部分任務】\n{section_spec['instruction']}\n\n"
            f"{COMMON_OUTPUT_RULES}"
        )

    def _refine_section_content(self, llm, section_spec: Dict[str, str], draft: str) -> str:
        """
        以相同系統人格再次提示 LLM，要求在初稿基礎上進行格式與內容微調。
        """
        if not draft or draft.strip() == "[來源缺失]":
            return draft
        label = section_spec["label"]
        title = _value_or_missing(self.title)
        refine_prompt = (
            f"你剛依據規範寫完《{title}》的「{label}」初稿，內容如下：\n"
            "```markdown\n"
            f"{draft}\n"
            "```\n\n"
            "請保留所有正確資訊，重新潤飾並做以下檢查：\n"
            "1. Markdown 標題、粗體與表格語法需合法並與指示一致。\n"
            "2. 引用必須保留或視情況補上 [來源缺失]；不得杜撰頁碼。\n"
            "3. 條列應使用 Markdown 無序列表；多段敘事需保持短句與姥姥語氣。\n"
            "4. 僅輸出最終 Markdown 版本，不要附加評論。\n\n"
            f"【原任務提醒】\n{section_spec['instruction']}\n\n"
            f"{COMMON_OUTPUT_RULES}"
        )
        try:
            refined = llm.generate(
                messages=[
                    {"role": "system", "content": _base_system_prompt()},
                    {"role": "user", "content": refine_prompt},
                ],
                temperature=0.15,
            )
            return refined.strip() or draft
        except Exception as exc:
            logger.error(f"Failed to refine section '{section_spec['key']}' for {self.arxiv_id}: {exc}")
            return draft

    def _generate_section_content(self, section_spec: Dict[str, str]) -> str:
        llm = get_llm()
        prompt = self._build_section_prompt(section_spec)
        system_prompt = _base_system_prompt()
        try:
            output = llm.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            if not output or not output.strip():
                raise ValueError("empty output")
            draft = output.strip()
            return self._refine_section_content(llm, section_spec, draft)
        except Exception as exc:
            logger.error(f"Failed to generate section '{section_spec['key']}' for {self.arxiv_id}: {exc}")
            return "[來源缺失]"

    @cached_property
    def digest_sections(self) -> Dict[str, str]:
        if not _full_summary_enabled():
            return {}
        sections: Dict[str, str] = {}
        for spec in SECTION_SPECS:
            sections[spec["key"]] = self._generate_section_content(spec)
        return sections

    def _generate_structured_digest_markdown(self) -> str:
        if not _full_summary_enabled():
            return "[Full summary disabled (FULL_SUMMARY != 1)]"
        blocks = []
        for spec in SECTION_SPECS:
            body = self.digest_sections.get(spec["key"], "[來源缺失]") or "[來源缺失]"
            blocks.append(f"{spec['title']}\n{body.strip()}")
        return "\n\n".join(blocks)

    @cached_property
    def digest_markdown(self) -> str:
        return self._generate_structured_digest_markdown()

    @cached_property
    def tldr(self) -> str:
        if not _full_summary_enabled():
            return "[Full summary disabled (FULL_SUMMARY != 1)]"
        return self.digest_markdown

    @cached_property
    def tldr_markdown(self) -> str:
        if not _full_summary_enabled():
            return "[Full summary disabled (FULL_SUMMARY != 1)]"
        return self.digest_markdown

    @cached_property
    def tldr_json(self) -> dict:
        if not _full_summary_enabled():
            return {
                "structured_digest_markdown": "[Full summary disabled (FULL_SUMMARY != 1)]",
                "sections": {},
                "teaser": self.teaser,
            }
        return {
            "structured_digest_markdown": self.digest_markdown,
            "sections": self.digest_sections,
            "teaser": self.teaser,
        }

    # ---------------- Teaser 生成 ----------------

    @cached_property
    def abstract_teaser(self) -> str:
        """
        只根據 abstract 生成極短中文摘要（FULL_SUMMARY=-1）。
        """
        abstract = (self.summary or "").strip()
        if not abstract:
            return "無摘要"
        llm = get_llm()
        limit = _teaser_char_limit()
        prompt = (
            "請只根據以下論文摘要，用繁體中文寫一段極短總結。\n"
            f"限制：**嚴格限制在 {limit} 個中文字以內**。\n"
            "不可加入摘要以外的資訊，不要提及作者、機構、標題或年份。\n"
            "只輸出總結文字。\n\n"
            f"摘要：\n{abstract}\n"
        )
        try:
            output = llm.generate(
                messages=[
                    {"role": "system", "content": "你是一個精簡的學術摘要專家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            teaser = (output or "").strip()
            if len(teaser) > limit:
                teaser = teaser[:limit].rstrip()
            return teaser
        except Exception as exc:
            logger.error(f"Failed to generate abstract teaser for {self.arxiv_id}: {exc}")
            return "無法生成摘要"

    @cached_property
    def teaser(self) -> str:
        """
        生成 100 字以內的中文 Teaser，包含問題與新意。
        """
        if _summary_mode() == "abstract":
            return self.abstract_teaser
        llm = get_llm()
        inputs_block = self._compose_digest_input_block()
        limit = _teaser_char_limit()
        prompt = (
            f"請為《{_value_or_missing(self.title)}》寫一段極短的中文介紹（Teaser）。\n"
            "目標：快速解釋「這篇論文解決什麼問題」以及「它的核心新意是什麼」，讓讀者決定是否繼續閱讀。\n"
            "限制：\n"
            f"1. **嚴格限制在 {limit} 個中文字以內**。\n"
            "2. 直接講重點，不要有「這篇論文...」之類的開頭廢話。\n"
            "3. 使用繁體中文。\n\n"
            f"【可用資訊】\n{inputs_block}\n"
        )
        try:
            output = llm.generate(
                messages=[
                    {"role": "system", "content": "你是一個精簡的學術摘要專家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            teaser = (output or "").strip()
            if len(teaser) > limit:
                teaser = teaser[:limit].rstrip()
            return teaser
        except Exception as exc:
            logger.error(f"Failed to generate teaser for {self.arxiv_id}: {exc}")
            return "無法生成摘要"

    # ---------------- affiliations：從 LaTeX 抽取作者單位 ----------------

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        """
        盡量從 LaTeX 作者區段抽取作者單位（回傳去重後的頂層單位）。
        與原專案相容：若無法抽取，回傳 None。
        """
        if _summary_mode() == "abstract":
            return None
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                try:
                    content = "\n".join(self.tex.values())
                except Exception:
                    content = None
            if not content:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: empty tex content.")
                return None

            # 兩個常見區域：\author... \maketitle 或 \begin{document} 到 \begin{abstract}
            possible_regions = [r'\\author.*?\\maketitle', r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
            match = next((m for m in matches if m), None)
            if match:
                information_region = match.group(0)
            else:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None

            # 組 LLM 提示詞（保留你原先的風格與行為）
            prompt = (
                "Given the author information of a paper in latex format, extract the affiliations of the authors "
                "in a python list format, which is sorted by the author order. If there is no affiliation found, "
                "return an empty list '[]'. Following is the author information:\n" + information_region
            )
            # 保留原行為：粗略截斷
            try:
                enc = tiktoken.encoding_for_model("gpt-4o")
                prompt_tokens = enc.encode(prompt)
                prompt = enc.decode(prompt_tokens[:4000])
            except Exception:
                pass

            llm = get_llm()
            affiliations_text = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. "
                            "You should return a python list of affiliations sorted by the author order, like "
                            "['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, "
                            "like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation "
                            "'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you "
                            "should return an empty list [ ]. You should only return the final list of affiliations, and do not return "
                            "any intermediate results."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            try:
                m = re.search(r'\[.*?\]', affiliations_text, flags=re.DOTALL)
                if not m:
                    raise ValueError("No list found in model output.")
                affils = eval(m.group(0))  # 延用原寫法；若要更穩健可換成 json.loads
                affils = list(set(affils))  # 去重
                affils = [str(a) for a in affils]
            except Exception as e:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affils

        # 無 tex 可用
        return None

    # ---------------- 舊版相容：保留最原始一行 TLDR（可選） ----------------
    # 若你仍想保留最初「一行 TLDR」函式，可在此新增 property，例如 tldr_one_line；
    # 但目前 render_email 會直接使用 p.tldr（新版多行）或 p.tldr_markdown（Markdown）。
