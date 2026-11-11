"""Persistent record of papers already processed, so they are never recommended twice.

Format (``{"papers": {primary_key: entry}}``) stores, per paper: the day it
was processed, its title and an abstract snippet, whether it was presented in
the email, and all of its dedup keys.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from time import sleep

from loguru import logger

REPLACE_ATTEMPTS = 10


class RecommendedHistory:
    def __init__(self, path: str | Path, papers: dict[str, dict] | None = None):
        self.path = Path(path)
        self.papers: dict[str, dict] = dict(papers or {})

    @classmethod
    def load(cls, path: str | Path) -> "RecommendedHistory":
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("papers"), dict):
                raise ValueError("history file is not a v2 JSON object")
            return cls(path, data["papers"])
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable recommendation history {path}: {e}")
            return cls(path)

    def seen_keys(self) -> set[str]:
        keys: set[str] = set()
        for entry in self.papers.values():
            keys.update(entry.get("keys", []))
        return keys

    def record_paper(
        self,
        keys: list[str],
        day: date | None = None,
        title: str = "",
        abstract: str = "",
        presented: bool | None = None,
    ) -> None:
        keys = [k for k in keys if k]
        if not keys:
            return
        day_iso = (day or date.today()).isoformat()
        # Merge into an existing entry when any key is already known (e.g. a
        # paper reprocessed with new identifiers).
        primary = next((k for k in keys if k in self.papers), None)
        if primary is None:
            primary = keys[0]
            self.papers[primary] = {
                "date": day_iso,
                "title": "",
                "abstract": "",
                "presented": None,
                "keys": [],
            }
        entry = self.papers[primary]
        all_keys = set(entry.get("keys", [])) | set(keys)
        # Fold duplicate entries that share any of these keys into this one.
        for other_primary in [k for k in all_keys if k in self.papers and k != primary]:
            other = self.papers.pop(other_primary)
            all_keys |= set(other.get("keys", []))
            if not title:
                title = other.get("title", "")
            if not abstract:
                abstract = other.get("abstract", "")
            if other.get("presented"):
                presented = True
            day_iso = max(day_iso, other.get("date", day_iso))
        entry["keys"] = sorted(all_keys)
        entry["date"] = max(entry.get("date", day_iso), day_iso)
        if title:
            entry["title"] = title
        if abstract:
            entry["abstract"] = abstract[:500]
        if presented:
            entry["presented"] = True
        elif entry.get("presented") is None and presented is not None:
            entry["presented"] = False

    def presented_not_saved(self, corpus_keys: set[str], cutoff: date) -> tuple[list[dict], list[dict]]:
        """Split presented papers into saved (keys intersect the current Zotero
        corpus) and ignored (no intersection, and older than the grace cutoff)."""
        saved: list[dict] = []
        ignored: list[dict] = []
        for entry in self.papers.values():
            if not entry.get("presented"):
                continue
            if set(entry.get("keys", [])) & corpus_keys:
                saved.append(entry)
            elif self._entry_date(entry) <= cutoff:
                ignored.append(entry)
        return saved, ignored

    @staticmethod
    def _entry_date(entry: dict) -> date:
        try:
            return datetime.strptime(entry.get("date", ""), "%Y-%m-%d").date()
        except ValueError:
            return date.min

    def prune(self, history_days: int, today: date | None = None) -> None:
        if history_days <= 0:
            return
        today = today or date.today()
        cutoff = today - timedelta(days=history_days)
        self.papers = {k: v for k, v in self.papers.items() if self._entry_date(v) >= cutoff}

    def save(self) -> None:
        """Write a temporary file and swap it in. Written in place, a crash in
        mid-write leaves a truncated file, which ``load`` reads as an empty
        history: the next run would recommend papers a second time.

        On Windows the swap fails while another program has the file open
        (a reader, a virus scanner), where writing in place would not, so it
        is tried again for a few seconds."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps({"papers": self.papers}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            for attempt in range(REPLACE_ATTEMPTS):
                try:
                    os.replace(temporary, self.path)
                    return
                except PermissionError:
                    if attempt == REPLACE_ATTEMPTS - 1:
                        raise
                    sleep(1)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
