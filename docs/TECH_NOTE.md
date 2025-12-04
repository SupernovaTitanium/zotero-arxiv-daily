# Tech note: pipeline overview

This project recommends arXiv papers based on a Zotero library and emails a daily digest with jump links. Components and flow:

## 1) Execution and orchestration
- GitHub Actions (`.github/workflows/main.yml`) runs nightly (`0 22 * * *`) or manually. It checks out code, installs via `uv`, and executes `uv run main.py` with secrets/vars for Zotero, arXiv query, SMTP, LLM, and language.
- Local run: set env vars (or CLI flags) and `uv run main.py --debug --max_paper_num 3` for a small test.

## 2) Inputs
- Zotero: `get_zotero_corpus` (in `main.py`) pulls `conferencePaper/journalArticle/preprint` items with non-empty abstracts. Collection paths are attached for optional filtering.
- Filtering: `filter_corpus` applies a gitignore-style pattern (`ZOTERO_IGNORE`) to drop collections.
- arXiv: `get_arxiv_paper` reads the RSS feed for the query, collects IDs marked `new`, batches through `arxiv.Client`, and constructs `ArxivPaper` objects. `max_paper_num` limits count when skipping Zotero or in debug mode.

## 3) Scoring and reranking
- `recommender.rerank_paper`: encodes Zotero abstracts and candidate arXiv abstracts with `sentence_transformers` (`avsolatorio/GIST-small-Embedding-v0`). Applies time-decay weights favoring recent Zotero items. Scores = weighted similarity ×10, then candidates are sorted.
- If Zotero is skipped or empty, a default score is assigned.

## 4) LLM generation
- `llm.py`: `set_global_llm` chooses OpenAI-compatible API when `USE_LLM_API=1` (needs `OPENAI_API_KEY/BASE/MODEL_NAME`), or local `llama_cpp` with `Qwen2.5-3B-Instruct-GGUF` when `USE_LLM_API=0`.
- `paper.py`: builds structured prompts per section (`SECTION_SPECS` A–G + QA) with persona/system rules (`_base_system_prompt()`, `COMMON_OUTPUT_RULES`). Uses paper title/abstract/parsed LaTeX sections; Zotero corpus is not sent to the LLM. Each section is generated then refined. `tldr_markdown` is the combined digest; `teaser` is a Chinese intro capped by `TEASER_CHAR_LIMIT` (default 150). Affiliations are extracted via a separate prompt on the LaTeX author block. Override persona via env `LLM_SYSTEM_PROMPT` if you need a different tone/structure.
- FULL_SUMMARY=0: skips generating structured digest/tldr in `paper.py` to avoid long-form LLM calls; teasers still use the LLM.

## 5) Email rendering and jump links
- `construct_email.py`: renders summary (“今日超級速覽”) plus detailed blocks.
  - Anchors: summary anchor `super-summary`; per-paper anchor from `_anchor_from_arxiv_id` (e.g., `paper-<arxiv_id>`). Summary links are visibly styled (`🔗 title`) and point to `#anchor`.
  - Detail blocks add hidden anchors and a back-link “回到今日超級速覽 ↑” to `#super-summary`. “Detailed” anchor `detailed-section` separates summary from details.
  - Content: TL;DR Markdown → HTML, PDF/code buttons, authors/affiliations, arXiv ID, star rating from score.
- Summary length: teaser limit is configurable via env `TEASER_CHAR_LIMIT` (default 150); longer outputs are truncated before rendering.
- Detail rendering toggle: set env `FULL_SUMMARY=1` to include detailed sections; `0` (default) sends only the quick overview.
- `send_email`: builds HTML MIME and sends via SMTP (TLS then SSL fallback).

## 6) Outputs
- HTML email with stable in-mail navigation (summary ↔ per-paper). No Zotero content is emitted; only arXiv data and model-generated text.
- Logging: progress and counts; avoids secrets and payload dumps.

## 7) Extending safely
- Keep secrets in env/CI secrets. Avoid logging paper bodies or secrets.
- If changing models or providers, verify licenses and data-handling policies.
- For higher security/offline use, set `USE_LLM_API=0` and cache embedding + LLM models locally.
- Consider incremental Zotero fetches via `?since=<last_version>` if you need daily delta sync.
