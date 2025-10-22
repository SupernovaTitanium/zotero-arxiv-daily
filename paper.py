# -*- coding: utf-8 -*-
"""
paper.py — 適配新版 llm.py（支援 temperature），分階段遞進式 TLDR 生成；
信件版面改為：
TLDR:
1. 摘要精煉
2. 主要貢獻
3. 可用創新技巧
4. 前人弱點和本文改進
5. 文章弱點和 Reasoning Gap
6. 補充解釋（只對 TLDR 文字中出現的術語做白話解釋）

保留：
- affiliations 從 LaTeX 抽取
- code_url 從 Papers with Code
- 下載/拼接 .tex
- 與 render_email 相容（p.tldr 為純文字；p.tldr_markdown 供 Markdown→HTML）
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
import time  # 供外部流程使用
from llm import get_llm
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken  # 僅用於 affiliations 提示截斷（保留原行為）
from contextlib import ExitStack
from urllib.error import HTTPError
import json
import os


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


def _json_from_text(s: str) -> dict:
    """
    盡力從模型輸出抽出 JSON。
    """
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r'\{.*\}', s, flags=re.DOTALL)
    if not m:
        return {}
    txt = m.group(0)
    txt = re.sub(r',\s*([\]}])', r'\1', txt)  # 移除尾逗號
    try:
        return json.loads(txt)
    except Exception:
        return {}


def _filter_glossary_by_usage(glossary: List[dict], used_text: str) -> List[dict]:
    """
    只保留在 used_text 內實際出現的術語（不區分大小寫），並去重。
    """
    used = []
    seen = set()
    text_l = used_text.lower()
    for item in glossary or []:
        term = (item.get("term") or "").strip()
        defi = (item.get("simple_def_zh") or "").strip()
        if not term or not defi:
            continue
        if term.lower() in seen:
            continue
        # 出現即保留（簡單包含判斷；若要更嚴謹可做 token 邊界）
        if term.lower() in text_l:
            used.append({"term": term, "simple_def_zh": defi})
            seen.add(term.lower())
    return used


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


BASE_SYSTEM_PROMPT = (
    "角色：支持型數學家（風格近陶哲軒）。\n\n"
    "核心準則\n\n"
    "先釐清：在下結論前，精確定義問題、目標、輸入／輸出、記號與假設。\n\n"
    "嚴謹為上：每一步推理皆可檢驗；不得以權威取代證明；錯誤與漏洞須明確定位並修正。\n\n"
    "誠實陳述完成度：若無法完全解決，只給出已證明的重要部分結果，並清晰說明未解的障礙、條件或反例。\n\n"
    "不逢迎：只評價論證與證據，不迎合立場或能力。\n\n"
    "反事實挑戰：主動尋找反例、必要條件、最壞情形與極端案例，以檢驗論斷的穩健性。\n\n"
    "第一性原理：從定義、公理與基本性質出發，少用類比，多用可驗證的邏輯與構造。\n\n"
    "費曼作風：以簡明語言重述觀念；反覆自檢；允許試錯；以可重現的實驗或計算支撐結論。\n\n"
    "磨礪求進：遇到阻礙時，分解為子目標、設定里程碑，持續演算與驗證。\n"
)

DEFAULT_COGNITIVE_LEVEL = "研究生"
_COGNITIVE_LEVEL_SPEC = {
    "本科": "面向高年級大學生，優先建立直觀，減少高等統計符號，必要時以例子輔助。",
    "研究生": "面向研究生，務必交代關鍵假設、漸近條件與必要資訊（例：可逆費雪資訊、局部錯配階數、QMD 連續性假設）。",
    "實務": "面向產業實務者，強調決策含義、風險與適用範圍，同時保留必要的統計條件。"
}


def _resolve_cognitive_level_name() -> str:
    raw = os.getenv("TLDR_COGNITIVE_LEVEL", "").strip()
    if not raw:
        raw = DEFAULT_COGNITIVE_LEVEL
    if raw not in _COGNITIVE_LEVEL_SPEC:
        logger.debug(f"Unknown TLDR_COGNITIVE_LEVEL '{raw}', fallback to '{DEFAULT_COGNITIVE_LEVEL}'.")
        return DEFAULT_COGNITIVE_LEVEL
    return raw


def _cognitive_level_clause() -> str:
    level = _resolve_cognitive_level_name()
    spec = _COGNITIVE_LEVEL_SPEC[level]
    return f"認知層級設定：{level}。{spec}\n"


def _cognitive_level_hint() -> str:
    level = _resolve_cognitive_level_name()
    spec = _COGNITIVE_LEVEL_SPEC[level]
    return f"{level}｜{spec}"


def _call_llm(messages: List[Dict[str, str]],
              temperature: Optional[float] = None,
              reasoning_hint: Optional[str] = None) -> dict:
    """
    呼叫全域 LLM；messages 第一個條目應為 system。
    - temperature：不同階段取樣溫度
    - reasoning_hint：插入 system 的內部指示（要求更深思考），不外露
    回傳：模型應輸出的 JSON 物件；若非 JSON，盡力擷取。
    """
    llm = get_llm()
    sys_boost = ""
    if reasoning_hint:
        sys_boost = f"（內部指示：{reasoning_hint}。請勿外露推理過程。）"
    first = messages[0].copy()
    first["content"] = first.get("content", "") + sys_boost
    msgs = [first] + messages[1:]

    try:
        raw = llm.generate(messages=msgs, temperature=temperature)
    except TypeError:
        raw = llm.generate(messages=msgs)
    return _json_from_text(raw)


# ------------------------------
# ArxivPaper 類別
# ------------------------------

class ArxivPaper:
    """
    與原始專案相容：
    - render_email 會用到：title, authors, affiliations, arxiv_id, tldr, pdf_url, code_url, score
    - tldr：回傳純文字 TLDR（Markdown 也可讀）
    - tldr_markdown：更適合郵件/前端的 Markdown 版本
    - tldr_json：完整 JSON（含 stages 與 used_glossary_zh）
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
        return self._paper.pdf_url

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

    # ---------------- 五個階段的 LLM 呼叫 ----------------

    def _stage1_summary_and_glossary(self) -> dict:
        """
        S1：產生摘要精煉與候選術語。
        """
        sys = (
            BASE_SYSTEM_PROMPT +
            _cognitive_level_clause() +
            "你要把提供的論文資訊轉成一段 90~130 字的繁體中文敘事，"
            "遵循「問題→障礙→方法→結果→意義」順序，語句需緊湊且可追溯來源。"
            "同時列出後續可能需解釋的術語，每個術語給一句白話定義。"
        )
        user = f"""
【任務】輸出 JSON 物件，鍵：
- summary_refined_zh：90~130 字，描述研究脈絡；若資訊不足要說明缺口。
- glossary_candidates：[{{"term": "<術語>", "simple_def_zh": "<一句白話解釋>"}} ...]。

【認知層級】{_cognitive_level_hint()}
【輸入】
Title: {self._ctx_for_tldr["title"]}
Abstract: {self._ctx_for_tldr["abstract"]}
Intro: {self._ctx_for_tldr["intro"]}
Conclusion: {self._ctx_for_tldr["concl"]}
""".strip()

        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
            reasoning_hint="先萃取核心資訊與前後因，再壓縮成一段敘事"
        ) or {"summary_refined_zh": "未知", "glossary_candidates": []}

    def _stage2_main_contributions(self, s1: dict) -> dict:
        """
        S2：說明主要貢獻。
        """
        sys = (
            BASE_SYSTEM_PROMPT +
            _cognitive_level_clause() +
            "你要以導師視角列出本文主要貢獻，必須指出驗證證據或必要條件。"
            "請以繁體中文撰寫 90~130 字，資訊不足時需註明。"
        )
        user = f"""
【任務】只輸出 JSON，鍵：
- main_contributions_zh：90~130 字，高密度描述本文的核心貢獻與支撐證據。

【認知層級】{_cognitive_level_hint()}
【可引用的名詞解釋（候選）】
{json.dumps(s1.get("glossary_candidates", []), ensure_ascii=False)}

【輸入】
Title: {self._ctx_for_tldr["title"]}
Abstract: {self._ctx_for_tldr["abstract"]}
Contrib-like sentences: {self._ctx_for_tldr["contrib_spans"]}
Method: {self._ctx_for_tldr["method"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.35,
            reasoning_hint="先列可能的貢獻與證據，再合併成一段細節充足的描述"
        ) or {"main_contributions_zh": "未知"}

    def _stage3_innovations(self, s1: dict, s2: dict) -> dict:
        """
        S3：萃取可用創新技巧。
        """
        sys = (
            BASE_SYSTEM_PROMPT +
            _cognitive_level_clause() +
            "你要提煉本文可複現的創新技巧，描述操作步驟、需要的條件與驗證訊號。"
            "若技巧合理性存疑，要明確標註風險或尚未驗證的假設。"
        )
        user = f"""
【任務】輸出 JSON，鍵：
- usable_innovations_zh：110~150 字，條理化說明 1~3 個創新技巧，需涵蓋操作步驟、必要條件與驗證訊號；若缺乏佐證請註明風險。

【認知層級】{_cognitive_level_hint()}
【前序資訊】
summary_refined_zh: {s1.get("summary_refined_zh","")}
main_contributions_zh: {s2.get("main_contributions_zh","")}

【輸入】
Method: {self._ctx_for_tldr["method"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.4,
            reasoning_hint="列舉候選技巧，評估條件與風險後組成高密度敘述"
        ) or {"usable_innovations_zh": "未知"}

    def _stage4_prior_and_improvement(self, s1: dict, s2: dict, s3: dict) -> dict:
        """
        S4：比較前人弱點與本文改進。
        """
        sys = (
            BASE_SYSTEM_PROMPT +
            _cognitive_level_clause() +
            "你要評估過往方法的關鍵缺陷，並逐點對應本文的改進及其有效機制。"
            "輸出需包含缺陷、對應改進與為何生效，語氣務實。"
        )
        user = f"""
【任務】輸出 JSON，鍵：
- prior_weakness_and_improvement_zh：110~150 字，按照「前人弱點 → 本文改進 → 為何有效」順序撰寫；若資料不足須註明。

【認知層級】{_cognitive_level_hint()}
【前序資訊】
summary_refined_zh: {s1.get("summary_refined_zh","")}
main_contributions_zh: {s2.get("main_contributions_zh","")}
usable_innovations_zh: {s3.get("usable_innovations_zh","")}

【輸入】
Intro: {self._ctx_for_tldr["intro"]}
Method: {self._ctx_for_tldr["method"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
Limitations/Discussion: {self._ctx_for_tldr["limits"]}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.45,
            reasoning_hint="建立缺陷清單，再與創新逐點對應，確認機制後輸出"
        ) or {"prior_weakness_and_improvement_zh": "未知"}

    def _stage5_weaknesses(self, s1: dict, s2: dict, s3: dict, s4: dict) -> dict:
        """
        S5：指出文章弱點與推理缺口。
        """
        sys = (
            BASE_SYSTEM_PROMPT +
            _cognitive_level_clause() +
            "你是誠實直接的仲裁者，要指出本文最脆弱的假設、資料/實驗盲點與推理缺口，"
            "並說明最容易失效的場景。"
        )
        user = f"""
【任務】輸出 JSON，鍵：
- paper_weakness_reasoning_gap_zh：110~150 字，描述最嚴重的弱點、缺乏驗證的假設與可能失效情境；若資訊不足須明示。

【認知層級】{_cognitive_level_hint()}
【前序資訊】
summary_refined_zh: {s1.get("summary_refined_zh","")}
main_contributions_zh: {s2.get("main_contributions_zh","")}
usable_innovations_zh: {s3.get("usable_innovations_zh","")}
prior_weakness_and_improvement_zh: {s4.get("prior_weakness_and_improvement_zh","")}

【輸入】
Experiments/Results: {self._ctx_for_tldr["expts"]}
Limitations/Discussion: {self._ctx_for_tldr["limits"]}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.45,
            reasoning_hint="列出風險與缺口，檢查是否有佐證，再挑出最嚴重者"
        ) or {"paper_weakness_reasoning_gap_zh": "未知"}

    def _build_glossary(self, merged: dict) -> List[dict]:
        """
        根據最終段落與候選術語，產生補充解釋清單。
        """
        candidates = merged.get("glossary_candidates", []) or []
        sections = {
            "摘要精煉": merged.get("summary_refined_zh", ""),
            "主要貢獻": merged.get("main_contributions_zh", ""),
            "可用創新技巧": merged.get("usable_innovations_zh", ""),
            "前人弱點和本文改進": merged.get("prior_weakness_and_improvement_zh", ""),
            "文章弱點和Reasoning Gap": merged.get("paper_weakness_reasoning_gap_zh", ""),
        }
        sections_text = "\n".join(f"{k}：{_normalize_text(v)}" for k, v in sections.items())
        sys = (
            BASE_SYSTEM_PROMPT +
            _cognitive_level_clause() +
            "你要整理上列段落中的專有名詞、統計條件與假設，"
            "確保「補充解釋」涵蓋每一個需要釐清的術語，尤其是弱點段所提到的條件。"
            "輸出需保持高密度且避免冗語。"
        )
        user = f"""
【任務】輸出 JSON 陣列，格式為 [{{"term": "...", "simple_def_zh": "..."}}]。
- 覆蓋所有段落中出現、讀者可能陌生的術語或假設（例：可逆費雪資訊、QMD、錯配階數）。
- 優先沿用候選術語的定義，必要時補充或調整。
- 每個解釋 1~2 句，符合認知層級要求；避免重複或無意義的詞條。
- 若完全不需要補充，請輸出 []。

【認知層級】{_cognitive_level_hint()}
【候選術語】{json.dumps(candidates, ensure_ascii=False)}
【段落】{sections_text}
""".strip()

        glossary_items: List[dict] = []
        try:
            result = _call_llm(
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.2,
                reasoning_hint="逐段檢視術語，對缺漏項補齊定義並去重"
            )
        except Exception as exc:
            logger.warning(f"Glossary generation failed for {self.arxiv_id}: {exc}")
            result = None

        items = []
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict):
            if isinstance(result.get("glossary"), list):
                items = result["glossary"]
            elif isinstance(result.get("items"), list):
                items = result["items"]

        seen = set()
        for entry in items or []:
            if not isinstance(entry, dict):
                continue
            term = _normalize_text(entry.get("term")).strip()
            defi = _normalize_text(entry.get("simple_def_zh")).strip()
            if not term or not defi:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            glossary_items.append({"term": term, "simple_def_zh": defi})

        if not glossary_items:
            return _filter_glossary_by_usage(candidates, " ".join(sections.values()))
        return glossary_items

    # ---------------- 最終 TLDR 組裝 ----------------

    @cached_property
    def tldr_json(self) -> dict:
        """
        分階段生成與合併，回傳 JSON：
          keys: summary_refined_zh, main_contributions_zh, usable_innovations_zh,
                prior_weakness_and_improvement_zh, paper_weakness_reasoning_gap_zh,
                glossary_candidates, used_glossary_zh, stages
        """
        s1 = self._stage1_summary_and_glossary()
        s2 = self._stage2_main_contributions(s1)
        s3 = self._stage3_innovations(s1, s2)
        s4 = self._stage4_prior_and_improvement(s1, s2, s3)
        s5 = self._stage5_weaknesses(s1, s2, s3, s4)

        merged = {
            "summary_refined_zh": s1.get("summary_refined_zh", "未知"),
            "main_contributions_zh": s2.get("main_contributions_zh", "未知"),
            "usable_innovations_zh": s3.get("usable_innovations_zh", "未知"),
            "prior_weakness_and_improvement_zh": s4.get("prior_weakness_and_improvement_zh", "未知"),
            "paper_weakness_reasoning_gap_zh": s5.get("paper_weakness_reasoning_gap_zh", "未知"),
            "glossary_candidates": s1.get("glossary_candidates", []),
            "stages": {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5},
        }

        merged["used_glossary_zh"] = self._build_glossary(merged)
        return merged

    @cached_property
    def tldr(self) -> str:
        """
        純文字 TLDR（Markdown 也可讀），嚴格依你指定的版面：
        1. 摘要精煉
        2. 主要貢獻
        3. 可用創新技巧
        4. 前人弱點和本文改進
        5. 文章弱點和 Reasoning Gap
        6. 補充解釋
        """
        d = self.tldr_json
        # 補充解釋只列 TLDR 內實際出現的術語
        if d.get("used_glossary_zh"):
            glossary_text = "；".join(f"{e['term']}：{e['simple_def_zh']}" for e in d["used_glossary_zh"])
        else:
            glossary_text = "無需特別解釋的術語。"

        lines = [
            f"1. 摘要精煉：{d.get('summary_refined_zh','未知')}",
            f"2. 主要貢獻：{d.get('main_contributions_zh','未知')}",
            f"3. 可用創新技巧：{d.get('usable_innovations_zh','未知')}",
            f"4. 前人弱點和本文改進：{d.get('prior_weakness_and_improvement_zh','未知')}",
            f"5. 文章弱點和Reasoning Gap：{d.get('paper_weakness_reasoning_gap_zh','未知')}",
            f"6. 補充解釋：{glossary_text}",
        ]
        return "\n".join(lines)

    @cached_property
    def tldr_markdown(self) -> str:
        """
        Markdown 版本（更易掃讀）。
        """
        d = self.tldr_json
        if d.get("used_glossary_zh"):
            exp_md = "；".join([f"**{e['term']}**：{e['simple_def_zh']}" for e in d.get("used_glossary_zh", [])])
        else:
            exp_md = "無需特別解釋的術語。"
        md = (
            f"1. **摘要精煉**：{d.get('summary_refined_zh','未知')}\n"
            f"2. **主要貢獻**：{d.get('main_contributions_zh','未知')}\n"
            f"3. **可用創新技巧**：{d.get('usable_innovations_zh','未知')}\n"
            f"4. **前人弱點和本文改進**：{d.get('prior_weakness_and_improvement_zh','未知')}\n"
            f"5. **文章弱點和Reasoning Gap**：{d.get('paper_weakness_reasoning_gap_zh','未知')}\n"
            f"6. **補充解釋**：{exp_md}\n"
        )
        return md

    # ---------------- affiliations：從 LaTeX 抽取作者單位 ----------------

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        """
        盡量從 LaTeX 作者區段抽取作者單位（回傳去重後的頂層單位）。
        與原專案相容：若無法抽取，回傳 None。
        """
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
