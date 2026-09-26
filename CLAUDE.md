# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zotero-arXiv-Daily recommends new arXiv/bioRxiv/medRxiv/chemRxiv papers based on a user's Zotero library. It computes embedding similarity between new papers and the user's existing library, generates TLDRs via LLM, and delivers results by email. Designed to run as a GitHub Actions workflow at zero cost.

## Commands

```bash
# Run the application
uv run python -m zotero_arxiv_daily.main

# Run tests (excludes slow tests by default)
uv run pytest

# Run all tests including slow ones
uv run pytest -m ""

# Run a single test
uv run pytest tests/test_utils.py::TestGlobMatch -v

# Install/sync dependencies
uv sync
```

No linter or formatter is configured. Lint with `uv run ruff check src tests scripts`.

## Architecture

The app is a linear pipeline in `src/zotero_arxiv_daily/pipeline.py` (`run(config)`):

1. **Fetch Zotero corpus** — `zotero.py`, via pyzotero; `include_path`/`ignore_path` glob filtering
2. **Retrieve new arXiv papers** — `arxiv.py`: search API primary, OAI-PMH harvest fallback (arXiv hard-throttles runner IPs; do not add polling loops against arXiv)
3. **Rank** — `embed.py`: local sentence-transformers embeddings (disk-cached in `state/corpus_embeddings.npz`, namespaced by model key), time-decayed corpus similarity, weekly-review preference boost/mute
4. **Topic grouping + full text** — greedy clustering for the email; full text (LaTeX tar → HTML → PDF, subprocess hard timeout) only for the top papers
5. **Generate teasers** — `teaser.py`: one batched LLM request per N papers, per-paper fallback; teaser mode only (no TLDR/affiliations/deep-digest paths)
6. **Render + send email** — `email.py` + `mailer.py`; run outputs written before sending, dedup history (`history.py`) persisted only after the send succeeds

### Configuration

`config.py` loads `config/base.yaml` + optional `config/custom.yaml` (written from the CUSTOM_CONFIG variable in Actions) with `${VAR}` / `${VAR:default}` env interpolation into dataclasses. `EMAIL_SMTP_SERVER`/`EMAIL_SMTP_PORT` (preferred) or legacy `SMTP_SERVER`/`SMTP_PORT` supply SMTP settings; `LOOKBACK_DAYS`, `MAX_PAPER_NUM`, `DEBUG` override executor values. Entry point is `main.py` (no Hydra).

### Data Classes

`Paper` and `CorpusPaper` in `src/zotero_arxiv_daily/paper.py` (plain data; no LLM methods).

## Testing

Tests marked `@pytest.mark.slow` require heavy dependencies (e.g., sentence-transformers model download) and are skipped locally by default (`addopts = "-m 'not slow'"` in pyproject.toml). All other tests run with pure Python stubs (no Docker containers needed).

```bash
# Run tests (excludes slow tests)
uv run pytest

# Run all tests including slow ones
uv run pytest -m ""

# Run with coverage
uv run pytest --cov=src/zotero_arxiv_daily --cov-report=term-missing
```

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`.

If gstack skills aren't working, run `cd .claude/skills/gstack && ./setup` to build the binary and register skills.

## Git Workflow

- PRs should target the `dev` branch, not `main`
- Current development branch: `dev`
