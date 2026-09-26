"""The production entrypoint must be importable as a module.

CI invokes `uv run python -m zotero_arxiv_daily.main`; running the file
directly (`python src/.../main.py`) breaks its relative imports. This test
pins the module-mode invocation: it must reach config loading (ConfigError,
since the test env has no credentials), never ImportError.
"""

import subprocess
import sys

from zotero_arxiv_daily.config import ConfigError


def test_module_invocation_reaches_config_loading():
    proc = subprocess.run(
        [sys.executable, "-m", "zotero_arxiv_daily.main"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "ImportError" not in combined, combined
    assert "ConfigError" in combined, combined
