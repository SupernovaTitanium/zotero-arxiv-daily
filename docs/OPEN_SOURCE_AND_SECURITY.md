# Open-source posture and safety notes

Project: Zotero-arXiv-Daily (AGPLv3). This doc summarizes security, privacy, and operational expectations for open-sourcing and running this repo.

## Secrets and configuration
- Required runtime secrets live in environment variables (or GitHub Actions secrets): `ZOTERO_KEY`, `ZOTERO_ID`, SMTP creds, OpenAI API key (when `USE_LLM_API=1`). No secrets are stored in the repo.
- GitHub Actions are configured to read secrets from `secrets.*` and non-sensitive defaults from `vars.*`. Do not hardcode keys; keep secrets in your CI/CD secret store.
- Set a separate Zotero API key with read-only scope for production, and rotate it periodically. Do not reuse personal SMTP or OpenAI keys outside this workflow.

## Data handling and privacy
- The workflow reads Zotero metadata (titles/abstracts/collections) to rerank papers. It does not persist Zotero data to disk beyond runtime.
- TL;DR prompts only include arXiv paper content (title/abstract/sections); Zotero corpus contents are not sent to the LLM.
- If `USE_LLM_API=1`, arXiv paper text is sent to the configured OpenAI-compatible endpoint. Choose a provider consistent with your data policy. Set `USE_LLM_API=0` to keep generation local (Qwen2.5-3B via llama.cpp).
- Email output contains only arXiv data plus generated summaries. No Zotero metadata is emitted.
- Logs are minimal (counts, progress bars). They do not print secrets or full paper bodies. Preserve that property in new code.

## Networked dependencies
- `sentence_transformers` pulls the embedding model (`avsolatorio/GIST-small-Embedding-v0`) from Hugging Face at first run. Cache or vendor if you need offline/locked-down deployments.
- Local LLM path downloads `Qwen/Qwen2.5-3B-Instruct-GGUF`. Ensure license compatibility with your distribution; mirror internally if needed.
- ArXiv access uses RSS + API; Zotero uses their public REST API with your key; SMTP hits your provider.

## Licensing and attributions
- Repository is AGPLv3. Keep notices intact. External assets and models carry their own licenses; check Hugging Face model cards and Qwen license before redistribution.
- Screenshots/logos under `assets/` should remain with attribution where applicable.

## Operational safeguards
- Validate `ARXIV_QUERY` to avoid accidental broad pulls. Current code raises on invalid feed titles.
- Keep timeouts/backoffs reasonable: arXiv client retries, SMTP TLS fallback are in place. Consider rate limits for Zotero; add backoff if you expect high volume.
- For new features, avoid logging payloads or secrets; treat environment variables as sensitive.

## Supply chain hygiene
- Pin dependencies via `pyproject.toml`/`uv.lock`. Verify hashes when vendoring models. Avoid running untrusted code from user-controlled inputs.
- CI workflows use `actions/checkout@v4` and `astral-sh/setup-uv@v3`; keep them updated for security patches.

## Contributing safely
- Use feature branches and PR reviews. Run `uv run main.py --debug --max_paper_num 3` for quick sanity checks.
- Do not commit test secrets. Prefer masked logs and scrub artifacts before sharing.

## Jump-link behavior (email UX)
- Summary anchor id: `super-summary`; per-paper anchors: `paper-<arxiv_id>` sanitized. Back-link in each paper points to `#super-summary`. Keep anchors stable for accessibility and email client compatibility.
