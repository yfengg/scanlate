"""The LanguageAdapter interface.

Language-neutral by construction: this module knows *what* an adapter may
provide, never *how* any particular language works. Adapters implement only
what their language needs; everything else returns empty/default data rather
than a fabricated result. Use ``capabilities()`` to ask what is real.
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from .analysis import (AnalysisContext, LinguisticAnalysis, OcrCleanup,
                       OcrProfile, RegisterAnalysis, Token)
from .types import Span, terminal_punctuation

CAPABILITIES = frozenset({
    "normalize", "tokenize", "morphology", "sentences", "register",
    "social_markers", "relationships", "ambiguity", "entities", "slang",
    "cultural", "sfx", "ocr_cleanup",
})


@dataclass
class NormalizedText:
    text: str
    original: str
    edits: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.text != self.original


class LanguageAdapter(ABC):
    language: str = "und"
    display_name: str = "Undetermined"

    def capabilities(self) -> frozenset[str]:
        return frozenset({"normalize", "sentences"})

    # --- text handling -------------------------------------------------
    def normalize(self, text: str) -> NormalizedText:
        collapsed = " ".join(text.split())
        edits = ["collapsed whitespace"] if collapsed != text.strip() else []
        return NormalizedText(collapsed, text, edits)

    def segment_sentences(self, text: str) -> list[Span]:
        return [Span(0, len(text))] if text else []

    def tokenize(self, text: str) -> list[Token]:
        return []

    # --- analysis ------------------------------------------------------
    def analyze(self, text: str, context: AnalysisContext | None = None) -> LinguisticAnalysis:
        """Adapters override. The default produces a truthful empty analysis."""
        return LinguisticAnalysis(
            language=self.language,
            text=text,
            sentences=self.segment_sentences(text),
            tokens=self.tokenize(text),
            register=RegisterAnalysis(),
            terminal_punctuation=set(terminal_punctuation(text)),
        )

    # --- comic-specific hooks -----------------------------------------
    def lookup_sfx(self, text: str):
        """Return a core.sfx SfxMatch, or None when the form is unknown."""
        return None

    def looks_like_sfx(self, text: str) -> bool:
        return False

    # --- future image pipeline ----------------------------------------
    def ocr_cleanup(self, text: str) -> OcrCleanup:
        return OcrCleanup(text=text)

    def ocr_profile(self, text: str) -> OcrProfile:
        return OcrProfile()
