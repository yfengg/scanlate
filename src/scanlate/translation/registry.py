"""Backend selection, per language pair.

Different pairs can use different models; the rest of the program depends only
on the interface.
"""
from __future__ import annotations

from ..core.types import LanguagePair
from .backend import TranslationBackend


class BackendRegistry:
    def __init__(self, default: TranslationBackend | None = None):
        self._by_pair: dict[str, TranslationBackend] = {}
        self._fallbacks: list[TranslationBackend] = []
        self._default = default

    def register(self, backend: TranslationBackend, pair: LanguagePair | None = None) -> None:
        if pair is not None:
            self._by_pair[str(pair)] = backend
        else:
            self._fallbacks.append(backend)

    def set_default(self, backend: TranslationBackend) -> None:
        self._default = backend

    def for_pair(self, pair: LanguagePair) -> TranslationBackend | None:
        if str(pair) in self._by_pair:
            return self._by_pair[str(pair)]
        for backend in self._fallbacks:
            if backend.supports(pair):
                return backend
        return self._default

    def describe(self) -> dict:
        backends = {str(p): b for p, b in self._by_pair.items()}
        for b in self._fallbacks:
            backends.setdefault("*", b)
        if self._default is not None:
            backends.setdefault("default", self._default)
        return {
            "by_pair": {p: b.name for p, b in backends.items()},
            "development": all(b.development for b in backends.values()) if backends else True,
            "names": sorted({b.name for b in backends.values()}),
        }
