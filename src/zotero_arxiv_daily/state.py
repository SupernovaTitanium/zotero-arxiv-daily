"""Persistent record of papers already processed, so they are never recommended twice.

Keys are stable identifiers shared with the dedup logic (see ``protocol.Paper.dedup_keys``):
``doi:<normalized doi>``, ``title:<normalized title>`` or ``sid:<source>:<source id>``.
The store is a plain JSON mapping ``key -> iso date``; entries older than the
retention window are pruned on save.
"""

import json
from datetime import date, timedelta
from pathlib import Path

from loguru import logger


class RecommendedHistory:
    def __init__(self, path: str | Path, entries: dict[str, str] | None = None):
        self.path = Path(path)
        self.entries: dict[str, str] = dict(entries or {})

    @classmethod
    def load(cls, path: str | Path) -> "RecommendedHistory":
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entries, dict):
                raise ValueError("history file is not a JSON object")
        except (OSError, ValueError) as e:
            logger.warning(f"Ignoring unreadable recommendation history {path}: {e}")
            return cls(path)
        return cls(path, {str(k): str(v) for k, v in entries.items()})

    def seen_keys(self) -> set[str]:
        return set(self.entries)

    def record(self, keys: list[str], day: date | None = None) -> None:
        day_iso = (day or date.today()).isoformat()
        for key in keys:
            if key:
                self.entries[key] = day_iso

    def prune(self, history_days: int, today: date | None = None) -> None:
        if history_days <= 0:
            return
        today = today or date.today()
        cutoff = (today - timedelta(days=history_days)).isoformat()
        self.entries = {k: v for k, v in self.entries.items() if v >= cutoff}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.entries, indent=0, sort_keys=True), encoding="utf-8"
        )
