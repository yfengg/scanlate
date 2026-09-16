"""Per-language knowledge resources.

Resources are JSON files in an adapter's ``data`` directory. The core knows
only the *generic* schemas (cultural terms, slang, SFX, relationship terms,
social markers, register rules, ambiguous terms); adapters may load any
additional private resources they need.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class KnowledgeBase:
    def __init__(self, language: str, root: Path | None = None):
        self.language = language
        self.root = root
        self._cache: dict[str, Any] = {}
        self._extra: dict[str, list[dict]] = {}

    def resource(self, name: str) -> Any:
        if name not in self._cache:
            data = None
            if self.root is not None:
                path = self.root / f"{name}.json"
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
            self._cache[name] = data
        return self._cache[name]

    def entries(self, name: str) -> list[dict]:
        data = self.resource(name)
        base = data if isinstance(data, list) else []
        return base + self._extra.get(name, [])

    def extend(self, name: str, entries: list[dict]) -> None:
        """Add entries at runtime (e.g. project- or user-supplied knowledge)."""
        self._extra.setdefault(name, []).extend(entries)
        self._cache.pop(f"__index__{name}", None)

    def cached(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)
