"""Text utilities: title/DOI normalization, glob matching, LaTeX/PDF text extraction."""

from __future__ import annotations

import glob
import math
import re
import tarfile
from collections import Counter

import pymupdf
import pymupdf.layout
from loguru import logger

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.layout.activate()

import pymupdf4llm  # noqa: E402

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str | None) -> str:
    """Canonical form of a paper title for cross-source duplicate matching."""
    return _NON_ALNUM_RE.sub("", (title or "").lower())


def normalize_doi(doi: str | None) -> str:
    """Canonical form of a DOI: lowercase, without any URL prefix."""
    doi = (doi or "").lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def glob_match(path: str, pattern: str) -> bool:
    return re.match(glob.translate(pattern, recursive=True), path) is not None


def extract_markdown_from_pdf(file_path: str) -> str:
    return pymupdf4llm.to_markdown(file_path, use_ocr=False, header=False, footer=False, ignore_code=True)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_pick(query: str, candidates: dict[str, str], k1: float = 1.5, b: float = 0.75) -> str:
    """Return the candidate key whose content best matches *query* by BM25."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return next(iter(candidates))

    doc_tokens = {name: _tokenize(content) for name, content in candidates.items()}
    n_docs = len(doc_tokens)
    avgdl = sum(len(t) for t in doc_tokens.values()) / max(n_docs, 1)

    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(set(tokens))

    best_name, best_score = None, -1.0
    for name, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for q in query_tokens:
            n_q = df.get(q, 0)
            idf = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1)
            f_q = tf.get(q, 0)
            score += idf * (f_q * (k1 + 1)) / (f_q + k1 * (1 - b + b * dl / max(avgdl, 1)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def extract_tex_code_from_tar(
    file_path: str, paper_id: str, paper_title: str | None = None
) -> dict[str, str] | None:
    """Extract (cleaned) LaTeX sources from an arXiv e-print tar, picking the
    main tex file (by .bbl match, \\begin{document} presence, or BM25 against
    the title). The merged main source is returned under the key ``"all"``."""
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None

    tex_files = [f for f in tar.getnames() if f.endswith(".tex")]
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None

    bbl_files = [f for f in tar.getnames() if f.endswith(".bbl")]
    match len(bbl_files):
        case 0:
            main_tex = tex_files[0] if len(tex_files) == 1 else None
            if main_tex is None:
                logger.debug(
                    f"Cannot find main tex file of {paper_id} from bbl: multiple tex files with no bbl file."
                )
        case 1:
            main_name = bbl_files[0].replace(".bbl", "")
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(
                    f"Cannot find main tex file of {paper_id} from bbl: the bbl file does not match any tex file."
                )
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: multiple bbl files.")
            main_tex = None

    file_contents: dict[str, str] = {}
    doc_block_candidates: list[str] = []
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode("utf-8", errors="ignore")
        content = re.sub(r"%.*\n", "\n", content)
        content = re.sub(r"\\begin{comment}.*?\\end{comment}", "", content, flags=re.DOTALL)
        content = re.sub(r"\\iffalse.*?\\fi", "", content, flags=re.DOTALL)
        content = re.sub(r"\n+", "\n", content)
        content = re.sub(r"\\\\", "", content)
        content = re.sub(r"[ \t\r\f]{3,}", " ", content)
        if main_tex is None and re.search(r"\\begin\{document\}", content) and not any(
            w in t for w in ["example", "sample", "template"]
        ):
            doc_block_candidates.append(t)
        file_contents[t] = content

    if main_tex is None and len(doc_block_candidates) > 1 and paper_title:
        main_tex = _bm25_pick(paper_title, {c: file_contents[c] for c in doc_block_candidates})
        logger.debug(f"Multiple document blocks found in {paper_id}; BM25 selected {main_tex}")
    elif main_tex is None and doc_block_candidates:
        main_tex = doc_block_candidates[0]
        logger.debug(f"Choose {main_tex} as main tex file of {paper_id}")

    if main_tex is not None:
        main_source: str = file_contents[main_tex]
        include_files = re.findall(r"\\input\{(.+?)\}", main_source) + re.findall(
            r"\\include\{(.+?)\}", main_source
        )
        for f in include_files:
            file_name = f if f.endswith(".tex") else f + ".tex"
            main_source = main_source.replace(f"\\input{{{f}}}", file_contents.get(file_name, ""))
        file_contents["all"] = main_source
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: no tex file containing the document block.")
        file_contents["all"] = None

    tar.close()
    return file_contents
