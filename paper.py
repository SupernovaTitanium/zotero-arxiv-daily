# -*- coding: utf-8 -*-
"""
paper.py — 適配新版 llm.py（支援 temperature），分階段遞進式 TLDR 生成，
並保留 affiliations 解析（從 LaTeX 作者區段抽取）。

提供：
- ArxivPaper 類別（保持與既有介面相容）
  - 屬性：title, summary, authors, arxiv_id, pdf_url, code_url, affiliations, tex, score
  - 生成屬性：
      * tldr_json: dict（含 stages 與合併結果）
      * tldr: str（純文字 TLDR，包含名詞解釋 + (1)~(6)）
      * tldr_markdown: str（Markdown 版本，適合郵件/前端）
設計原則：
- 不做硬性字數裁切；字數提示僅作語氣指導。
- 四個 stage，使用不同溫度與系統提示，逐步收斂（S1 低溫→S4 高溫）。
- 只根據提供文本；不外露推理過程；未知則標「未知」。
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import re
import json
import time  # 供外部可能使用，不在本檔內直接使用
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


def _call_llm(messages: List[Dict[str, str]],
              temperature: Optional[float] = None,
              reasoning_hint: Optional[str] = None) -> dict:
    """
    呼叫全域 LLM；messages 需含第一個 system 與隨後的 user。
    - temperature：不同階段的取樣溫度
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
        # 若 llm.generate 不接受 temperature（理論上你已更新），則退化
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
    - tldr_json：完整 JSON（含 stages）
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

    # ---------------- 四個階段的 LLM 呼叫 ----------------

    def _stage1_abstract_and_glossary(self) -> dict:
        """
        S1：低溫（0.2）
        - explanations：名詞解釋（每詞一句白話，給 5 歲能懂）
        - abstract_2sentences_zh：精煉摘要（指導原則：兩句以內）
        """
        sys = (
            "你是一位非常有聲望且經驗豐富的數學系教授（風格近陶哲軒）。"
            "請用繁體中文；用直覺與洞見讓 5 歲孩童也能理解。"
            "先把接下來會用到的專有名詞做『名詞解釋』列點（每詞一句白話定義），再產出精煉摘要。"
            "只根據提供文本；不要外露推理過程。"
        )
        user = f"""
【任務】輸出 JSON，鍵為：
- explanations: [{{"term": "<術語>", "simple_def_zh": "<一句白話定義>"}} , ...]
- abstract_2sentences_zh: 以直覺與洞見壓縮的摘要（指導原則：兩句以內，非硬性）

【輸入】
Title: {self._ctx_for_tldr["title"]}
Abstract: {self._ctx_for_tldr["abstract"]}
Intro: {self._ctx_for_tldr["intro"]}
Conclusion: {self._ctx_for_tldr["concl"]}
""".strip()

        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
            reasoning_hint="先在內部列出關鍵概念與最小充分描述，再輸出 JSON"
        ) or {"explanations": [], "abstract_2sentences_zh": "未知"}

    def _stage2_contrib_tech_prior(self, s1: dict) -> dict:
        """
        S2：中低溫（0.35）
        - contributions_50zh：本文主要貢獻（不做硬性字數）
        - techniques_100zh：達成貢獻的新技巧與為何有效（不做硬性字數）
        - prior_challenges_50zh：以往困難與為何此作可解且重要（不做硬性字數）
        """
        sys = (
            "你是嚴謹、直率且鼓勵後進的教授。"
            "用繁體中文，以直覺語言說清楚：做了什麼、怎麼做到、為何重要與可行。"
            "先定義後使用：你可以引用下列名詞解釋，但不要外露推理過程。"
            "只根據提供文本作答。"
        )
        user = f"""
【任務】輸出 JSON，鍵為：
- contributions_50zh
- techniques_100zh
- prior_challenges_50zh
（以上皆為指導字數上限，非硬性限制）

【可引用的名詞解釋】
{json.dumps(s1.get("explanations", []), ensure_ascii=False)}

【輸入（僅依據）】
Title: {self._ctx_for_tldr["title"]}
Abstract: {self._ctx_for_tldr["abstract"]}
Intro: {self._ctx_for_tldr["intro"]}
Contrib-like sentences: {self._ctx_for_tldr["contrib_spans"]}
Method: {self._ctx_for_tldr["method"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
""".strip()

        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.35,
            reasoning_hint="在內部先列候選貢獻與證據，再做保守聚合"
        ) or {
            "contributions_50zh": "未知",
            "techniques_100zh": "未知",
            "prior_challenges_50zh": "未知",
        }

    def _stage3_weaknesses(self, s1: dict, s2: dict) -> dict:
        """
        S3：中溫（0.45）
        - weaknesses_50zh：本文最可能的弱點/效度威脅（不做硬性字數）
        """
        sys = (
            "你是誠實直接的仲裁者（INTJ 風格）。"
            "用繁體中文，指出最脆弱的假設、資料/實驗的盲點、可反例的場景。"
            "只根據提供的文本與前序摘要；不外露推理過程。"
        )
        user = f"""
【任務】輸出 JSON，鍵為：
- weaknesses_50zh

【前序資訊】
explanations: {json.dumps(s1.get("explanations", []), ensure_ascii=False)}
contributions: {s2.get("contributions_50zh", "")}
techniques: {s2.get("techniques_100zh", "")}
prior challenges: {s2.get("prior_challenges_50zh", "")}

【輸入（僅依據）】
Limitations/Discussion: {self._ctx_for_tldr["limits"]}
Experiments/Results: {self._ctx_for_tldr["expts"]}
""".strip()

        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.45,
            reasoning_hint="在內部先搜索反例與最壞情況，最後輸出最具說服力的一點"
        ) or {"weaknesses_50zh": "未知"}

    def _stage4_brainstorm(self, s1: dict, s2: dict, s3: dict) -> dict:
        """
        S4：高溫（0.9）
        - brainstorm_100zh：以本文為起點的新研究問題（要有可驗證信號/最小實驗）
        """
        sys = (
            "你現在是有創造力但務實的導師。"
            "用繁體中文，提出新的學術問題（工作性假說），必須指出可驗證信號或最小實驗。"
            "不外露推理過程。"
        )
        user = f"""
【任務】輸出 JSON，鍵為：
- brainstorm_100zh  （建議≤100字，非硬性，務必包含可驗證信號或最小實驗）

【前序資訊】
explanations: {json.dumps(s1.get("explanations", []), ensure_ascii=False)}
contributions: {s2.get("contributions_50zh", "")}
techniques: {s2.get("techniques_100zh", "")}
weaknesses: {s3.get("weaknesses_50zh", "")}
""".strip()

        return _call_llm(
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.9,
            reasoning_hint="展開多路構想，最後收斂為一條實做得起的題目描述"
        ) or {"brainstorm_100zh": "未知"}

    # ---------------- 最終 TLDR 組裝 ----------------

    @cached_property
    def tldr_json(self) -> dict:
        """
        分階段生成與合併，回傳 JSON：
          keys: explanations, abstract_2sentences_zh, contributions_50zh,
                techniques_100zh, prior_challenges_50zh, weaknesses_50zh,
                brainstorm_100zh, stages
        """
        s1 = self._stage1_abstract_and_glossary()
        s2 = self._stage2_contrib_tech_prior(s1)
        s3 = self._stage3_weaknesses(s1, s2)
        s4 = self._stage4_brainstorm(s1, s2, s3)

        merged = {
            "explanations": s1.get("explanations", []),
            "abstract_2sentences_zh": s1.get("abstract_2sentences_zh", "未知"),
            "contributions_50zh": s2.get("contributions_50zh", "未知"),
            "techniques_100zh": s2.get("techniques_100zh", "未知"),
            "prior_challenges_50zh": s2.get("prior_challenges_50zh", "未知"),
            "weaknesses_50zh": s3.get("weaknesses_50zh", "未知"),
            "brainstorm_100zh": s4.get("brainstorm_100zh", "未知"),
            "stages": {"s1": s1, "s2": s2, "s3": s3, "s4": s4},
        }
        return merged

    @cached_property
    def tldr(self) -> str:
        """
        純文字 TLDR（Markdown 也可讀），包含名詞解釋 + (1)~(6)。
        與現有 render_email 相容：可直接作為 get_block_html 的內容。
        """
        d = self.tldr_json
        expl = d.get("explanations", [])
        if expl:
            expl_lines = ["名詞解釋（不佔配額）："] + [f"- {e.get('term','')}：{e.get('simple_def_zh','')}" for e in expl]
        else:
            expl_lines = ["名詞解釋（不佔配額）：無"]

        lines = [
            "TLDR：這包含了以下資訊",
            *expl_lines,
            f"(1) 大概兩句精煉摘要：{d.get('abstract_2sentences_zh', '未知')}",
            f"(2) 貢獻：{d.get('contributions_50zh', '未知')}",
            f"(3) 值得學的新技巧：{d.get('techniques_100zh', '未知')}",
            f"(4) 以往困難與為何可解：{d.get('prior_challenges_50zh', '未知')}",
            f"(5) 弱點：{d.get('weaknesses_50zh', '未知')}",
            f"(6) 腦動新題：{d.get('brainstorm_100zh', '未知')}",
        ]
        return "\n".join(lines)

    @cached_property
    def tldr_markdown(self) -> str:
        """
        Markdown 版本（更易掃讀）。你可在 render_email 中轉為 HTML。
        """
        d = self.tldr_json
        expl = d.get("explanations", [])
        ex_md = "\n".join([f"- **{e.get('term','')}**：{e.get('simple_def_zh','')}" for e in expl]) or "無"
        md = (
            f"### TLDR\n\n"
            f"**名詞解釋（不佔配額）**\n{ex_md}\n\n"
            f"1. **大概兩句精鍊摘要**：{d.get('abstract_2sentences_zh','未知')}\n"
            f"2. **貢獻**：{d.get('contributions_50zh','未知')}\n"
            f"3. **值得學的新技巧**：{d.get('techniques_100zh','未知')}\n"
            f"4. **以往困難與為何可解**：{d.get('prior_challenges_50zh','未知')}\n"
            f"5. **弱點**：{d.get('weaknesses_50zh','未知')}\n"
            f"6. **腦動新題**：{d.get('brainstorm_100zh','未知')}\n"
        )
        return md

    # ---------------- affiliations：從 LaTeX 抽取作者單位 ----------------

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        """
        盡量從 LaTeX 作者區段抽取作者單位（回傳去重後的頂層單位列表）。
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

            # 嘗試兩個常見區域：\author... \maketitle 或 \begin{document} 到 \begin{abstract}
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
            # 仍沿用 gpt-4o 編碼做粗略截斷（保持與原行為一致）
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
                # 解析回傳：只取第一個 [ ... ] 片段
                m = re.search(r'\[.*?\]', affiliations_text, flags=re.DOTALL)
                if not m:
                    raise ValueError("No list found in model output.")
                affils = eval(m.group(0))  # 順著原始寫法；若要更穩健可改用 json.loads
                affils = list(set(affils))  # 去重
                affils = [str(a) for a in affils]
            except Exception as e:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affils

        # 無 tex 可用
        return None
