"""Persistent record of papers already processed, so they are never recommended twice.

Format v2 (``{"papers": {primary_key: entry}}``) stores, per paper: the day it
was processed, its title and an abstract snippet, whether it was presented in
the email, and all of its dedup keys. This lets the weekly preference review
tell "recommended but never saved" (negative signal) apart from "recommended
and later added to Zotero" (positive signal) without any extra clicks.

Format v1 (flat ``key -> date``) is loaded transparently; v1 entries carry no
metadata, so they never count as presented.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger


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
            if isinstance(data, dict) and isinstance(data.get("papers"), dict):
                return cls(path, data["papers"])
            # v1 flat format: key -> iso date string
            if isinstance(data, dict):
                papers = {
                    str(k): {
                        "date": str(v)[:10],
                        "title": "",
                        "abstract": "",
                        "presented": None,
                        "keys": [str(k)],
                    }
                    for k, v in data.items()
                }
                return cls(path, papers)
            raise ValueError("history file is not a JSON object")
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
        # v1-migrated entry or a paper reprocessed with new identifiers).
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

    def presented_not_saved(
        self, corpus_keys: set[str], cutoff: date
    ) -> tuple[list[dict], list[dict]]:
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
        self.papers = {
            k: v for k, v in self.papers.items() if self._entry_date(v) >= cutoff
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"papers": self.papers}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
