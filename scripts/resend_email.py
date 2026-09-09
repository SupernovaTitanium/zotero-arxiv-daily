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
import zipfile

import requests
from omegaconf import OmegaConf

from zotero_arxiv_daily.mailer import send_email


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="workflow run whose email should be resent")
    args = parser.parse_args()

    repo = os.environ["GITHUB_REPOSITORY"]
    headers = {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }

    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/runs/{args.run_id}/artifacts",
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
    response = requests.get(artifacts[0]["archive_download_url"], headers=headers, timeout=120)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = sorted(n for n in archive.namelist() if n.startswith("email_") and n.endswith(".html"))
    if not names:
        raise SystemExit(f"No email_*.html inside artifact {artifacts[0]['name']}")

    html = archive.read(names[0]).decode("utf-8")
    config = OmegaConf.load("config/custom.yaml")
    send_email(config, html)
    print(f"Resent {names[0]} ({len(html)} bytes) to the configured receiver")


if __name__ == "__main__":
    main()
