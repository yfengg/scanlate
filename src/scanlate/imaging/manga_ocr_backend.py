"""Japanese manga OCR, and language-aware routing between OCR backends.

``MangaOcr`` wraps kha-white/manga-ocr, a ViT encoder-decoder trained on manga
text. It takes a whole cropped bubble and reads it directly — no line
segmentation, and vertical text needs no special handling, which is the main
reason it beats a general engine on this material. Tesseract needs the right
`jpn` vs `jpn_vert` model chosen up front and degrades on stylised lettering.

``OcrRouter`` keeps the choice in one place: Japanese prefers manga-ocr when it
is genuinely usable, everything else falls through to the general backend.
Nothing Japanese-specific leaks into ``OcrBackend`` or the page service.

Two things this model does not give you, carried honestly rather than papered
over:

* **No confidence score.** ``confidence`` is None and ``reports_confidence`` is
  False, so the UI can say "no confidence score" instead of letting absence
  read as certainty.
* **No orientation report.** It handles vertical and horizontal text
  internally and tells you nothing about which it saw, so the result carries
  back whatever orientation the caller supplied and invents nothing.

Cost, stated plainly because it is not small:

* dependencies — torch plus transformers, roughly 2–3 GB installed
* model weights — about 450 MB, downloaded from Hugging Face on first use
* first run — that download, then a few seconds to load; afterwards the model
  stays in memory and a bubble takes well under a second on CPU
* offline — with no cached weights and no network, initialization fails and the
  backend reports itself unavailable with the reason, rather than hanging
"""
from __future__ import annotations

from io import BytesIO

from .layout import BoundingBox, TextOrientation
from .ocr import OcrBackend, OcrError, OcrResult

MODEL_ID = "kha-white/manga-ocr-base"
APPROXIMATE_WEIGHTS_MB = 450


class MangaOcr(OcrBackend):
    name = "manga-ocr"
    reports_confidence = False

    def __init__(self, model_id: str = MODEL_ID, upscale: int = 1):
        self.model_id = model_id
        self.upscale = upscale
        self._reader = None
        #: Why initialization failed, kept so the router and the UI can say
        #: what went wrong instead of only that something did.
        self.failure: str | None = None

    def languages(self) -> list[str]:
        return ["ja"]

    def available(self) -> bool:
        """Importable and not already known to be broken.

        Importability alone is not enough — the weights may be missing and the
        network unreachable — so a failed load is remembered here and the
        router stops choosing this backend.
        """
        if self.failure is not None:
            return False
        try:
            import manga_ocr  # noqa: F401
            return True
        except Exception as error:
            self.failure = f"manga-ocr is not installed ({error})"
            return False

    def load(self):
        """Load the model. Lazy: importing pulls in torch, and the first call
        may download ~450 MB of weights."""
        if self._reader is not None:
            return self._reader
        if self.failure is not None:
            raise OcrError(f"manga-ocr is unavailable: {self.failure}")
        try:
            from manga_ocr import MangaOcr as _MangaOcr
        except Exception as error:
            self.failure = f"manga-ocr is not installed ({error})"
            raise OcrError(f"manga-ocr is unavailable: {self.failure}")

        try:
            # A custom model id is honoured for real. If this build of
            # manga-ocr can't take one, that is an error rather than a silent
            # fall back to the default weights — otherwise the metadata would
            # claim a model that never ran.
            if self.model_id == MODEL_ID:
                self._reader = _MangaOcr()
            else:
                try:
                    self._reader = _MangaOcr(pretrained_model_name_or_path=self.model_id)
                except TypeError as error:
                    self.failure = (f"this build of manga-ocr doesn't accept a custom model id "
                                    f"({error})")
                    raise OcrError(f"Can't load {self.model_id!r}: {self.failure}")
        except OcrError:
            raise
        except Exception as error:
            self.failure = str(error)
            raise OcrError(
                f"manga-ocr couldn't start ({error}). The weights download on first use; "
                f"check the network, or type the source text instead.")
        return self._reader

    def recognize(self, image_bytes: bytes, box: BoundingBox, language: str | None = None,
                  orientation: TextOrientation | None = None) -> OcrResult:
        try:
            from PIL import Image
        except Exception as error:
            raise OcrError(f"Pillow isn't available: {error}")

        reader = self.load()
        try:
            with Image.open(BytesIO(image_bytes)) as page:
                crop = page.convert("RGB").crop(
                    (box.x, box.y, box.x + box.width, box.y + box.height))
                if self.upscale > 1:
                    crop = crop.resize((crop.width * self.upscale, crop.height * self.upscale))
                crop.load()
        except Exception as error:
            raise OcrError(f"The region couldn't be read from the page: {error}")

        try:
            # The whole bubble goes in as one image; the model handles vertical
            # and horizontal lettering itself.
            text = (reader(crop) or "").strip()
        except Exception as error:
            raise OcrError(f"manga-ocr failed on this region: {error}")

        if not text:
            raise OcrError("No text was recognized in this region.")
        return OcrResult(
            text=text, language="ja",
            confidence=None,               # the model exposes none; see reports_confidence
            orientation=orientation,       # whatever the caller knew; nothing invented
            metadata={"model": self.model_id, "whole_region": True,
                      "confidence_available": False,
                      "orientation_source": "caller" if orientation else "unknown"})


class OcrRouter(OcrBackend):
    """Picks the backend best suited to the region's language.

    When a preferred backend turns out to be unusable, the router either falls
    back to the general backend and says so in the result metadata, or reports
    the preferred backend's own error — ``on_error`` decides which. It never
    selects a backend that has already failed.
    """
    name = "router"

    def __init__(self, default: OcrBackend, by_language: dict[str, OcrBackend] | None = None,
                 on_error: str = "fallback"):
        if on_error not in {"fallback", "error"}:
            raise ValueError("on_error must be 'fallback' or 'error'")
        self.default = default
        self.by_language = by_language or {}
        self.on_error = on_error

    @property
    def reports_confidence(self) -> bool:
        return self.default.reports_confidence

    def available(self) -> bool:
        return self.default.available() or any(b.available() for b in self.by_language.values())

    def languages(self) -> list[str]:
        found = set(self.default.languages())
        for code, backend in self.by_language.items():
            if backend.available():
                found.add(code)
        return sorted(found)

    def backend_for(self, language: str | None) -> OcrBackend:
        preferred = self.by_language.get(language or "")
        if preferred is not None and preferred.available():
            return preferred
        return self.default

    def describe(self) -> dict:
        return {
            "default": {"name": self.default.name, "available": self.default.available()},
            "preferred": {code: {"name": b.name, "available": b.available(),
                                 "failure": getattr(b, "failure", None)}
                          for code, b in self.by_language.items()},
            "on_error": self.on_error,
        }

    def recognize(self, image_bytes: bytes, box: BoundingBox, language: str | None = None,
                  orientation: TextOrientation | None = None) -> OcrResult:
        backend = self.backend_for(language)
        try:
            result = backend.recognize(image_bytes, box, language, orientation)
        except OcrError as error:
            fallback_possible = (backend is not self.default and self.on_error == "fallback"
                                 and self.default.available())
            if not fallback_possible:
                raise
            result = self.default.recognize(image_bytes, box, language, orientation)
            result.metadata["fell_back_from"] = backend.name
            result.metadata["fallback_reason"] = str(error)
            backend = self.default
        result.metadata.setdefault("backend", backend.name)
        result.metadata.setdefault("confidence_available", backend.reports_confidence)
        return result
