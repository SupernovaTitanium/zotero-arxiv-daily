"""Run the digest on a remote host with credentials read from stdin.

scripts/run_digest.ps1 copies the code, config and state to the remote work
directory and starts this file over SSH with config/local.env.json on stdin,
so the credentials live only in this process's environment and are never
written to the remote disk.
"""

import json
import os
import runpy
import sys
from pathlib import Path

ALLOWED = {
    "ZOTERO_ID", "ZOTERO_KEY", "SENDER", "RECEIVER",
    "SENDER_PASSWORD", "OPENAI_API_KEY", "EMAIL_SMTP_SERVER", "EMAIL_SMTP_PORT",
}

credentials = json.loads(sys.stdin.buffer.read().decode("utf-8"))
if set(credentials) != ALLOWED or not all(isinstance(v, str) and v for v in credentials.values()):
    sys.exit("remote_entry: stdin must hold exactly the eight credential entries")
os.environ.update(credentials)

repo = Path(__file__).resolve().parent.parent
os.chdir(repo)
sys.path.insert(0, str(repo / "src"))
# SSH carries stderr on a separate channel; merging here keeps tracebacks in
# order with the log lines when the dispatcher reads them.
os.dup2(sys.stdout.fileno(), sys.stderr.fileno())
runpy.run_module("zotero_arxiv_daily.main", run_name="__main__")
