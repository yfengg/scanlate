"""OCR.

``OcrBackend`` is the seam. The shipped backend is Tesseract, chosen because it
is open source, runs locally with no network or GPU, ships trained data for
Japanese (including vertical `jpn_vert`), Korean and Chinese, and installs from
system packages — so adding a language later is a package install and a row in
``LANGUAGE_MODELS``, not a redesign.

Two rules the rest of the system relies on:

* Low confidence is never a failure. It is reported and flagged for review.
* Total failure never blocks the workflow; the user types the source text and
  continues. Every failure path here raises ``OcrError``, which callers turn
  into an empty, editable segment.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from .layout import BoundingBox, TextOrientation

# Shared language codes -> tesseract traineddata names.
LANGUAGE_MODELS: dict[str, str] = {
    "ja": "jpn",
    "ko": "kor",
    "zh": "chi_sim",
    "zh-Hant": "chi_tra",
    "en": "eng",
}
VERTICAL_MODELS: dict[str, str] = {"ja": "jpn_vert", "zh": "chi_sim_vert"}
LOW_CONFIDENCE = 0.55


class OcrError(Exception):
    """OCR could not run or produced nothing. Manual entry stays available."""


@dataclass
class OcrResult:
    text: str
    language: str | None = None
    #: None means the engine reports no score — not that the score was low.
    confidence: float | None = None
    #: None means the engine didn't determine which way the text ran.
    orientation: TextOrientation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def low_confidence(self) -> bool:
        return self.confidence is not None and self.confidence < LOW_CONFIDENCE

    @property
    def confidence_available(self) -> bool:
        return self.confidence is not None


class OcrBackend(ABC):
    name: str = "ocr"
    #: False for engines that return no per-result score, so the UI can say
    #: "no confidence score" rather than treating absence as certainty.
    reports_confidence: bool = True

    @abstractmethod
    def recognize(self, image_bytes: bytes, box: BoundingBox, language: str | None = None,
                  orientation: TextOrientation | None = None) -> OcrResult:
        """``orientation`` overrides the shape-based guess when the caller knows
        better — a saved region or a user correction."""

    def available(self) -> bool:
        return True

    def languages(self) -> list[str]:
        return []


class NullOcr(OcrBackend):
    """No engine configured. Fails cleanly so manual entry takes over."""
    name = "none"

    def available(self) -> bool:
        return False

    def recognize(self, image_bytes, box, language=None, orientation=None) -> OcrResult:
        raise OcrError("No OCR engine is configured. Type the source text instead.")


class TesseractOcr(OcrBackend):
    name = "tesseract"

    def __init__(self, upscale: int = 2):
        self.upscale = upscale

    def available(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def languages(self) -> list[str]:
        try:
            import pytesseract
            installed = set(pytesseract.get_languages())
        except Exception:
            return []
        return sorted({code for code, model in LANGUAGE_MODELS.items() if model in installed})

    def _model(self, language: str | None, vertical: bool) -> str:
        if language is None:
            # Let tesseract decide between the scripts we support rather than
            # imposing the project default on a foreign-language sign.
            available = self.languages() or ["en"]
            return "+".join(LANGUAGE_MODELS[code] for code in available)
        if vertical and language in VERTICAL_MODELS:
            return VERTICAL_MODELS[language]
        return LANGUAGE_MODELS.get(language, language)

    def recognize(self, image_bytes: bytes, box: BoundingBox, language: str | None = None,
                  orientation: TextOrientation | None = None) -> OcrResult:
        try:
            import pytesseract
            from PIL import Image
        except Exception as error:
            raise OcrError(f"OCR isn't available: {error}")

        try:
            with Image.open(BytesIO(image_bytes)) as page:
                crop = page.convert("L").crop(
                    (box.x, box.y, box.x + box.width, box.y + box.height))
                if self.upscale > 1:
                    crop = crop.resize((crop.width * self.upscale, crop.height * self.upscale))
                crop.load()
        except Exception as error:
            raise OcrError(f"The region couldn't be read from the page: {error}")

        # An explicit orientation — saved on the region, or corrected by the
        # user — decides. The aspect-ratio rule is only the initial guess for a
        # region that has never been told which way its text runs.
        vertical = (orientation is TextOrientation.VERTICAL_RL if orientation is not None
                    else box.height > box.width * 1.6)
        model = self._model(language, vertical)
        config = "--psm 5" if vertical else "--psm 6"

        try:
            data = pytesseract.image_to_data(crop, lang=model, config=config,
                                             output_type=pytesseract.Output.DICT)
        except Exception as error:
            raise OcrError(f"OCR failed on this region: {error}")

        words, scores = [], []
        for text, score in zip(data.get("text", []), data.get("conf", [])):
            if not text or not text.strip():
                continue
            words.append(text.strip())
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                scores.append(value / 100.0)

        text = _join(words)
        if not text:
            raise OcrError("No text was recognized in this region.")
        return OcrResult(
            text=text,
            language=language,
            confidence=round(sum(scores) / len(scores), 3) if scores else None,
            orientation=TextOrientation.VERTICAL_RL if vertical else TextOrientation.HORIZONTAL,
            metadata={"model": model, "psm": config, "words": len(words),
                      "orientation_source": "explicit" if orientation is not None else "geometry"})


def _join(words: list[str]) -> str:
    """CJK words are joined without spaces; Latin keeps them."""
    out = ""
    for word in words:
        if out and (_is_latin(out[-1]) or _is_latin(word[0])):
            out += " "
        out += word
    return re.sub(r"\s+", " ", out).strip()


def _is_latin(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def build_ocr(name: str | None = None) -> OcrBackend:
    """Assemble the OCR stack.

    The default routes by language: Japanese prefers manga-ocr when it is
    installed, everything else uses Tesseract. ``tesseract`` or ``manga-ocr``
    pin one engine; ``none`` disables OCR and leaves manual entry.
    """
    from .manga_ocr_backend import MangaOcr, OcrRouter

    name = (name or "auto").lower()
    if name in {"none", "null"}:
        return NullOcr()
    if name == "tesseract":
        backend = TesseractOcr()
        return backend if backend.available() else NullOcr()
    if name in {"manga-ocr", "manga_ocr", "manga"}:
        backend = MangaOcr()
        return backend if backend.available() else NullOcr()
    if name == "auto":
        general = TesseractOcr()
        return OcrRouter(general if general.available() else NullOcr(), {"ja": MangaOcr()})
    raise ValueError(f"Unknown OCR backend {name!r}")
