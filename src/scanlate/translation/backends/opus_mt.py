"""Dedicated neural MT backend (Helsinki-NLP OPUS-MT via transformers).

A real translation model, not a chat LLM: it takes source text and returns
target text, nothing else. Optional — if torch/transformers or the weights are
absent, ``available()`` is False and the pipeline falls back to another backend.
Terminology constraints are passed through to the pipeline's enforcement step
rather than prompted for, because this model has no instruction interface.
"""
from __future__ import annotations

from ...core.types import LanguagePair
from ..backend import (BackendResult, TranslationBackend, TranslationCandidate,
                       TranslationRequest)

#: transformers 5 dropped the generic ``pipeline("translation")`` task this
#: backend is built on. Rather than let a clean install resolve to 5 and fail
#: at the first translation, the backend refuses to report itself available and
#: says why.
SUPPORTED_TRANSFORMERS = (4,)

MODELS = {
    "ja>en": "Helsinki-NLP/opus-mt-ja-en",
    "ko>en": "Helsinki-NLP/opus-mt-ko-en",
    "zh>en": "Helsinki-NLP/opus-mt-zh-en",
}


class OpusMtBackend(TranslationBackend):
    name = "opus-mt"

    def __init__(self, models: dict[str, str] | None = None, beams: int = 4):
        self.models = models or MODELS
        self.beams = beams
        self._pipelines: dict[str, object] = {}
        self._unavailable: set[str] = set()

    def __init_subclass__(cls, **kwargs):     # pragma: no cover - inheritance hook
        super().__init_subclass__(**kwargs)

    def unavailable_reason(self) -> str | None:
        """Why this backend can't run, or None when it can."""
        try:
            import transformers
        except Exception as error:
            return f"transformers isn't installed ({error})"
        version = getattr(transformers, "__version__", "0")
        try:
            major = int(str(version).split(".")[0])
        except ValueError:
            major = 0
        if major not in SUPPORTED_TRANSFORMERS:
            return (f"transformers {version} is installed, but this backend needs the generic "
                    f"translation pipeline that version 5 removed. "
                    f"Pin it: pip install 'transformers>=4.40,<5'")
        if not hasattr(transformers, "pipeline"):
            return f"transformers {version} exposes no pipeline() factory"
        return None

    def available(self) -> bool:
        return self.unavailable_reason() is None

    def supports(self, pair: LanguagePair) -> bool:
        return str(pair) in self.models and str(pair) not in self._unavailable and self.available()

    def _pipeline(self, pair: LanguagePair):
        key = str(pair)
        if key not in self._pipelines:
            try:
                from transformers import pipeline  # type: ignore
                self._pipelines[key] = pipeline("translation", model=self.models[key])
            except Exception:
                self._unavailable.add(key)
                self._pipelines[key] = None
        return self._pipelines[key]

    def translate(self, request: TranslationRequest) -> BackendResult:
        reason = self.unavailable_reason()
        if reason is not None:
            return BackendResult(None, metadata={"backend": self.name, "reason": reason})
        translator = self._pipeline(request.pair)
        if translator is None:
            return BackendResult(None, metadata={"backend": self.name, "reason": "model unavailable"})
        outputs = translator(request.source_text, num_beams=self.beams,
                             num_return_sequences=min(2, self.beams))
        texts = [o["translation_text"].strip() for o in outputs]
        if not texts:
            return BackendResult(None, metadata={"backend": self.name, "reason": "empty output"})
        alternatives = [TranslationCandidate(t, 0.5, note="second beam", reason="literal")
                        for t in texts[1:] if t != texts[0]]
        return BackendResult(TranslationCandidate(texts[0], 0.7, reason="neural MT"),
                             alternatives, {"backend": self.name, "model": self.models[str(request.pair)]})
