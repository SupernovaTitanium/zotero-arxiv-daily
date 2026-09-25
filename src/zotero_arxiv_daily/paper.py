"""Core data structures: a retrieved paper and a Zotero corpus paper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .utils import normalize_doi, normalize_title


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None = None
    full_text: str | None = None
    teaser: str | None = None
    score: float | None = None
    doi: str | None = None
    source_id: str | None = None
    topic: str | None = None

    def dedup_keys(self) -> list[str]:
        keys = []
        if self.doi:
            keys.append("doi:" + normalize_doi(self.doi))
        if self.title:
            keys.append("title:" + normalize_title(self.title))
        if self.source_id:
            keys.append(f"sid:{self.source}:{self.source_id}")
        return keys


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
    doi: str | None = None

    def dedup_keys(self) -> list[str]:
        keys = []
        if self.doi:
            keys.append("doi:" + normalize_doi(self.doi))
        if self.title:
            keys.append("title:" + normalize_title(self.title))
        return keys
