"""Deterministic phrase-table backend.

Exists so the surrounding logic — glossary enforcement, memory, validation,
API behaviour — can be tested without model randomness. Same input, same
output, always. It is honest about failure: an unknown sentence returns no
primary candidate rather than an invented one.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from ...core.types import LanguagePair
from ..backend import (BackendResult, TranslationBackend, TranslationCandidate,
                       TranslationRequest)

DATA = Path(__file__).parent / "data"


class LexiconBackend(TranslationBackend):
    name = "lexicon"
    development = True

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or DATA
        self._tables: dict[str, dict] = {}

    def _table(self, pair: LanguagePair) -> dict | None:
        key = f"{pair.source}-{pair.target}"
        if key not in self._tables:
            path = self.data_dir / f"{key}.json"
            self._tables[key] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return self._tables[key]

    def supports(self, pair: LanguagePair) -> bool:
        return self._table(pair) is not None

    def add_phrase(self, pair: LanguagePair, source: str, target: str, **kw) -> None:
        table = self._table(pair)
        if table is not None:
            table["phrases"].insert(0, {"source": source, "target": target, **kw})

    def translate(self, request: TranslationRequest) -> BackendResult:
        table = self._table(request.pair)
        if table is None:
            return BackendResult(None, metadata={"backend": self.name, "reason": "unsupported pair"})

        source = _key(request.source_text)
        for phrase in table["phrases"]:
            if _key(phrase["source"]) == source:
                primary = TranslationCandidate(phrase["target"], confidence=0.9,
                                               reason="phrase table")
                alternatives = [TranslationCandidate(a["text"], 0.7, a.get("note"), a.get("reason"))
                                for a in phrase.get("alternatives", [])]
                return BackendResult(primary, alternatives, {
                    "backend": self.name, "match": "exact phrase",
                    "source_register": phrase.get("register")})

        # Word-level fallback: enough to be useful, never dressed up as fluent.
        lexicon = table.get("lexicon", {})
        glossed = [lexicon[term] for term in sorted(lexicon, key=len, reverse=True)
                   if _key(term) in source]
        for constraint in request.constraints:
            if _key(constraint.source_term) in source and constraint.target_term not in glossed:
                glossed.insert(0, constraint.target_term)
        if glossed:
            return BackendResult(
                TranslationCandidate(" ".join(dict.fromkeys(glossed)), confidence=0.2,
                                     note="word-level gloss only", reason="gloss"),
                metadata={"backend": self.name, "match": "lexicon gloss"})
        return BackendResult(None, metadata={"backend": self.name, "match": "none"})


def _key(text: str) -> str:
    """Phrase-table lookup key.

    Adapters normalize before the backend is reached (NFKC folds ？ to ?), so
    the table is matched on the same footing rather than on raw authored text.
    """
    return unicodedata.normalize("NFKC", text).strip()
