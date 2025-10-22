# -*- coding: utf-8 -*-
"""
paper.py — 適配新版 llm.py（支援 temperature），分階段遞進式 TLDR 生成；
信件版面改為：
TLDR:
1. 摘要精煉
2. 貢獻
3. 新技巧
4. 困難與超越
5. 弱點
腦洞
補充解釋（只對 TLDR 文字中出現的術語做白話解釋）

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

    def _stage1_story_and_glossary(self) -> dict:
        """
        S1：低溫（0.2）
        - abstract_story_zh：故事型精煉摘要（問題→障礙→方法→結果→意義）
        - glossary_candidates：候選術語（每詞一句白話定義）
        """
        sys = (
            "你是一位非常有聲望且經驗豐富的數學系教授（風格近陶哲軒）。"
            "請用繁體中文；用直覺與洞見讓 5 歲孩童也能理解。"
            "僅根據提供文本作答；不要外露推理過程。"
        )
        user = f"""
【任務】只輸出一個 JSON 物件，鍵：
- abstract_story_zh：用「問題→障礙→方法→結果→意義」講清楚本文在做什麼與得到什麼；語句精煉但資訊密。
- glossary_candidates：[{{
    "term": "<術語>", "simple_def_zh": "<一句白話定義>"
}} ...]（列出你在後續(1)~(6)會用到、且對 5 歲孩童需要解釋的術語）

【輸入】
Title: {self._ctx_for_tldr["title"]}
Abstract: {self._ctx_for_tldr["abstract"]}
Intro: {self._ctx_for_tldr["intro"]}
Conclusion: {self._ctx_for_tldr["concl"]}
""".strip()

        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
            reasoning_hint="先在內部萃取設定/目標/方法/結果/意義，再組成兩三句高密度敘事"
        ) or {"abstract_story_zh": "未知", "glossary_candidates": []}

    def _stage2_contributions(self, s1: dict) -> dict:
        """
        S2：0.35
        - contributions_zh：本文主要貢獻（條理清楚、避免空話）
        """
        sys = (
            "你是嚴謹、直率且鼓勵後進的教授。"
            "請用繁體中文，以可驗證的語句描述本文的主要貢獻；避免口號。"
            "可引用前一階段提供的名詞解釋，但不重複長篇定義。"
        )
        user = f"""
【任務】只輸出 JSON，鍵：
- contributions_zh：條列或短段落均可，但要密度高且具體。

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
            reasoning_hint="在內部列候選貢獻與證據，再做保守聚合"
        ) or {"contributions_zh": "未知"}

    def _stage3_prior(self, s1: dict, s2: dict) -> dict:
        """
        S3：0.45
        - prior_challenges_zh：以往文獻的核心困難與限制（精煉）
        """
        sys = (
            "你是嚴謹的領域評審，指出過往方法的關鍵困難與限制。"
            "請用繁體中文；僅依據提供文本；不外露推理過程。"
        )
        user = f"""
【任務】只輸出 JSON，鍵：
- prior_challenges_zh：濃縮描述以往文獻在本題上的核心困難與限制。

【前序資訊】
abstract_story_zh: {s1.get("abstract_story_zh","")}
contributions_zh: {s2.get("contributions_zh","")}

【輸入】
Intro: {self._ctx_for_tldr["intro"]}
Method: {self._ctx_for_tldr["method"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
Limitations/Discussion: {self._ctx_for_tldr["limits"]}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.45,
            reasoning_hint="先列三到五條困難，合併去重，輸出最關鍵者"
        ) or {"prior_challenges_zh": "未知"}

    def _stage4_techniques_and_beyond(self, s1: dict, s2: dict, s3: dict) -> dict:
        """
        S4：0.45
        - techniques_zh：達成貢獻的新技巧（點出機制/構造/關鍵步驟）
        - why_better_zh：清楚說明「新技巧為何能解以往做不到的事」（對應 prior_challenges）
        - difficulties_and_beyond_zh：把「困難→此文如何跨越→為何重要」敘事化成一個緊湊段落
        """
        sys = (
            "你是善於做機制拆解的導師。"
            "請用繁體中文，說清楚新技巧的關鍵構造/步驟/機制，以及它如何逐點擊破以往的困難。"
            "避免空話，強調對應關係。"
        )
        user = f"""
【任務】只輸出 JSON，鍵：
- techniques_zh：點出關鍵步驟/構造/機制與直覺。
- why_better_zh：以「舊困難 → 新技巧 → 為何有效（機制）」的對應關係表述。
- difficulties_and_beyond_zh：把「以往困難 + 本文如何跨越 + 為何重要」講成一段緊湊的故事。

【前序資訊】
abstract_story_zh: {s1.get("abstract_story_zh","")}
contributions_zh: {s2.get("contributions_zh","")}
prior_challenges_zh: {s3.get("prior_challenges_zh","")}

【輸入】
Method: {self._ctx_for_tldr["method"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.45,
            reasoning_hint="建立逐點映射：每個 prior challenge 連到一個技巧與其有效機制"
        ) or {
            "techniques_zh": "未知",
            "why_better_zh": "未知",
            "difficulties_and_beyond_zh": "未知",
        }

    def _stage5_brainstorm(self, s1: dict, s2: dict, s3: dict, s4: dict) -> dict:
        """
        S5：0.9
        - brainstorm_zh：以本文為起點的新研究問題（要有可驗證信號或最小實驗）
        """
        sys = (
            "你現在是有創造力但務實的導師。"
            "請用繁體中文，提出新的學術問題（工作性假說），必須包含可驗證信號或最小實驗。"
        )
        user = f"""
【任務】只輸出 JSON，鍵：
- brainstorm_zh：短而有操作性；包含可驗證信號或最小實驗。

【前序資訊】
abstract_story_zh: {s1.get("abstract_story_zh","")}
contributions_zh: {s2.get("contributions_zh","")}
prior_challenges_zh: {s3.get("prior_challenges_zh","")}
techniques_zh: {s4.get("techniques_zh","")}
why_better_zh: {s4.get("why_better_zh","")}
""".strip()
        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.9,
            reasoning_hint="展開數個方向再收斂為一條可落地的題目描述"
        ) or {"brainstorm_zh": "未知"}

    # ---------------- 最終 TLDR 組裝 ----------------

    @cached_property
    def tldr_json(self) -> dict:
        """
        分階段生成與合併，回傳 JSON：
          keys: abstract_story_zh, contributions_zh, prior_challenges_zh,
                techniques_zh, why_better_zh, difficulties_and_beyond_zh,
                weaknesses_zh, brainstorm_zh,
                glossary_candidates, used_glossary_zh, stages
        """
        # 依序跑 S1~S5；弱點沿用先前單獨抽取的 S3 型，但放在 S4/前後都可
        s1 = self._stage1_story_and_glossary()
        s2 = self._stage2_contributions(s1)
        s3 = self._stage3_prior(s1, s2)
        s4 = self._stage4_techniques_and_beyond(s1, s2, s3)

        # 弱點：沿用原來思路（中溫、批判）
        def _stage_weaknesses() -> dict:
            sys = (
                "你是誠實直接的仲裁者（INTJ 風格）。"
                "用繁體中文，指出最脆弱的假設、資料/實驗的盲點、可能失效的場景。"
                "僅依據提供文本與前序摘要；不外露推理過程。"
            )
            user = f"""
【任務】只輸出 JSON，鍵：
- weaknesses_zh：精煉指出本文最可能的弱點或威脅到效度之處。

【前序資訊】
abstract_story_zh: {s1.get("abstract_story_zh","")}
contributions_zh: {s2.get("contributions_zh","")}
prior_challenges_zh: {s3.get("prior_challenges_zh","")}
techniques_zh: {s4.get("techniques_zh","")}
why_better_zh: {s4.get("why_better_zh","")}

【輸入】
Limitations/Discussion: {self._ctx_for_tldr["limits"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
""".strip()
            return _call_llm(
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.45,
                reasoning_hint="先尋找反例與最壞情況，再輸出最具說服力的一點"
            ) or {"weaknesses_zh": "未知"}

        sW = _stage_weaknesses()
        s5 = self._stage5_brainstorm(s1, s2, s3, s4)

        merged = {
            "abstract_story_zh": s1.get("abstract_story_zh", "未知"),
            "contributions_zh": s2.get("contributions_zh", "未知"),
            "prior_challenges_zh": s3.get("prior_challenges_zh", "未知"),
            "techniques_zh": s4.get("techniques_zh", "未知"),
            "why_better_zh": s4.get("why_better_zh", "未知"),
            "difficulties_and_beyond_zh": s4.get("difficulties_and_beyond_zh", "未知"),
            "weaknesses_zh": sW.get("weaknesses_zh", "未知"),
            "brainstorm_zh": s5.get("brainstorm_zh", "未知"),
            "glossary_candidates": s1.get("glossary_candidates", []),
            "stages": {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "sW": sW, "s5": s5},
        }

        # 依最終 TLDR 文字過濾術語，生成 used_glossary_zh（補充解釋用）
        tl_text = " ".join(
            _normalize_text(part) for part in [
                merged.get("abstract_story_zh"),
                merged.get("contributions_zh"),
                merged.get("techniques_zh"),
                merged.get("difficulties_and_beyond_zh"),
                merged.get("weaknesses_zh"),
                merged.get("brainstorm_zh"),
            ]
        )
        merged["used_glossary_zh"] = _filter_glossary_by_usage(merged["glossary_candidates"], tl_text)
        return merged

    @cached_property
    def tldr(self) -> str:
        """
        純文字 TLDR（Markdown 也可讀），嚴格依你指定的版面：
        1. 摘要精煉
        2. 貢獻
        3. 新技巧
        4. 困難與超越
        5. 弱點
        腦洞
        補充解釋
        """
        d = self.tldr_json
        # 補充解釋只列 TLDR 內實際出現的術語
        if d.get("used_glossary_zh"):
            explain_lines = ["補充解釋："] + [f"- {e['term']}：{e['simple_def_zh']}" for e in d["used_glossary_zh"]]
        else:
            explain_lines = ["補充解釋：無需特別解釋的術語。"]

        lines = [
            "TLDR:",
            f"1. 摘要精煉: {d.get('abstract_story_zh','未知')}",
            f"2. 貢獻: {d.get('contributions_zh','未知')}",
            f"3. 新技巧: {d.get('techniques_zh','未知')}",
            f"4. 困難與超越: {d.get('difficulties_and_beyond_zh','未知')}",
            f"5. 弱點: {d.get('weaknesses_zh','未知')}",
            f"腦洞: {d.get('brainstorm_zh','未知')}",
            *explain_lines,
        ]
        return "\n".join(lines)

    @cached_property
    def tldr_markdown(self) -> str:
        """
        Markdown 版本（更易掃讀）。
        """
        d = self.tldr_json
        exp_md = "\n".join([f"- **{e['term']}**：{e['simple_def_zh']}" for e in d.get("used_glossary_zh", [])]) \
                 or "無需特別解釋的術語。"
        md = (
            f"**TLDR**\n\n"
            f"1. **摘要精煉**：{d.get('abstract_story_zh','未知')}\n"
            f"2. **貢獻**：{d.get('contributions_zh','未知')}\n"
            f"3. **新技巧**：{d.get('techniques_zh','未知')}\n"
            f"4. **困難與超越**：{d.get('difficulties_and_beyond_zh','未知')}\n"
            f"5. **弱點**：{d.get('weaknesses_zh','未知')}\n"
            f"**腦洞**：{d.get('brainstorm_zh','未知')}\n\n"
            f"**補充解釋**\n{exp_md}\n"
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
