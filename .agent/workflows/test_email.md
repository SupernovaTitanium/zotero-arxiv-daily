---
description: Test email generation with random papers (skips Zotero)
---

This workflow runs the main script in debug mode, skipping the Zotero corpus retrieval. It fetches 5 random papers (or latest ones depending on debug logic), generates the email with the new teaser section, and sends it.

1. Run the main script with debug and skip_zotero flags.
   Replace `YOUR_EMAIL` and `YOUR_PASSWORD` with actual credentials if running manually, or ensure they are set in the environment.
   
   **Note**: This command assumes you have the necessary environment variables set for `SENDER`, `RECEIVER`, `SENDER_PASSWORD`, `SMTP_SERVER`, `SMTP_PORT`. If not, you need to provide them as arguments.

```bash
# // turbo
python3 main.py --debug --skip_zotero --arxiv_query "cat:cs.AI" --max_paper_num 5
```
