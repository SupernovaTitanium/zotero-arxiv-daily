# Codex notes

- Email jump links: summary anchor id is `super-summary`; each paper anchor comes from `_anchor_from_arxiv_id` (e.g., `paper-<arxiv-id>`). Summary links use `href="#<anchor>"`.
- Each detail block renders a hidden anchor plus a back link labeled “回到今日超級速覽 ↑” that points to `#super-summary`; the detailed section marker also has both `id` and `name` for compatibility.
- If you change layout/styling, keep these anchors intact so go-to/back navigation inside emails stays reliable across clients.
- Reference implementation lives in `construct_email.py` (see `_build_summary_section`, `get_block_html`, and `render_email`). The pattern mirrors `blog-stalking/construct_email.py`’s overview/post anchors.
