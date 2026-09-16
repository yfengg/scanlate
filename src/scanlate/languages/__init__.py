"""Adapter registry and script-based language detection.

Importing this package pulls in the shipped adapters. The core never imports
it; the pipeline receives a registry instance.
"""
from __future__ import annotations

from ..core.adapter import LanguageAdapter
from .detect import detect_language


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, LanguageAdapter] = {}

    def register(self, adapter: LanguageAdapter) -> None:
        self._adapters[adapter.language] = adapter

    def get(self, language: str) -> LanguageAdapter | None:
        return self._adapters.get(language) or self._adapters.get(language.split("-")[0])

    def require(self, language: str) -> LanguageAdapter:
        adapter = self.get(language)
        if adapter is None:
            raise KeyError(f"No adapter registered for language {language!r}")
        return adapter

    def languages(self) -> list[str]:
        return sorted(self._adapters)

    def resolve(self, text: str, default: str | None = None) -> str:
        """Detected language if confident, else the project default."""
        detected = detect_language(text)
        if detected and self.get(detected):
            return detected
        return default or detected or "und"


def default_registry() -> AdapterRegistry:
    from .en.adapter import EnglishAdapter
    from .ja.adapter import JapaneseAdapter
    from .ko.adapter import KoreanAdapter

    registry = AdapterRegistry()
    for adapter in (JapaneseAdapter(), KoreanAdapter(), EnglishAdapter()):
        registry.register(adapter)
    return registry


__all__ = ["AdapterRegistry", "default_registry", "detect_language"]
