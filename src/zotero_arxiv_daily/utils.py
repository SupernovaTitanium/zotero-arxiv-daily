"""Text utilities: title/DOI normalization and glob matching."""

from __future__ import annotations

import glob
import re

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
