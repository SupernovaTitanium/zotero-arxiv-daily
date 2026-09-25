"""Re-send the email HTML produced by a previous run of this workflow.

Usage: resend_email.py --run-id <workflow run id>

Downloads the run-outputs artifact of that run and sends its email_*.html
through the configured mailer. No state is touched, so it is safe to re-run
and never disturbs the recommendation dedup history.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from zotero_arxiv_daily.config import load_config  # noqa: E402
from zotero_arxiv_daily.mailer import send_email  # noqa: E402

GITHUB_HOSTS = {"github.com", "api.github.com", "objects.githubusercontent.com"}


def _validated_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in GITHUB_HOSTS:
        raise ValueError(f"Refusing to fetch from unexpected URL: {url!r}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="workflow run whose email should be resent")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+", args.run_id):
        raise SystemExit("--run-id must be a numeric workflow run id")

    repo = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise SystemExit(f"GITHUB_REPOSITORY has an unexpected format: {repo!r}")
    headers = {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.get(
        _validated_url(f"https://api.github.com/repos/{repo}/actions/runs/{args.run_id}/artifacts"),
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    artifacts = [
        a for a in response.json().get("artifacts", []) if a["name"].startswith("run-outputs-")
    ]
    if not artifacts:
        raise SystemExit(f"No run-outputs artifact found on run {args.run_id}")

    # GitHub signs the download URL, and requests drops the auth header on the
    # cross-host redirect, which is what the signed URL expects.
    response = requests.get(_validated_url(artifacts[0]["archive_download_url"]), headers=headers, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = sorted(n for n in archive.namelist() if n.startswith("email_") and n.endswith(".html"))
    if not names:
        raise SystemExit(f"No email_*.html inside artifact {artifacts[0]['name']}")

    html = archive.read(names[0]).decode("utf-8")
    send_email(load_config(REPO_ROOT / "config").email, html)
    print(f"Resent {names[0]} ({len(html)} bytes) to the configured receiver")


if __name__ == "__main__":
    main()
