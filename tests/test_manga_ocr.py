"""Regressions for the manga-OCR backend and OCR routing.

manga-ocr itself is not installed in CI, so a fake module stands in for it.
That is the point: these pin the behaviour around the model — selection,
failure, fallback, and what gets reported — not the model's accuracy.
"""
from __future__ import annotations

import sys
import types

import pytest

from scanlate.imaging.layout import BoundingBox, TextOrientation
from scanlate.imaging.manga_ocr_backend import MODEL_ID, MangaOcr, OcrRouter
from scanlate.imaging.ocr import NullOcr, OcrBackend, OcrError, OcrResult
from tests.test_imaging_corrections import png

BOX = BoundingBox(0, 0, 40, 60)


def install_fake_manga_ocr(monkeypatch, *, reads="こんにちは", raises=None,
                           accepts_model_id=True, record=None):
    """Put a stand-in `manga_ocr` module on the import path."""
    module = types.ModuleType("manga_ocr")

    class FakeMangaOcr:
        def __init__(self, *args, **kwargs):
            if not accepts_model_id and ("pretrained_model_name_or_path" in kwargs or args):
                raise TypeError("__init__() got an unexpected keyword argument")
            if raises is not None:
                raise raises
            if record is not None:
                record.append(kwargs.get("pretrained_model_name_or_path", MODEL_ID))

        def __call__(self, image):
            return reads

    module.MangaOcr = FakeMangaOcr
    monkeypatch.setitem(sys.modules, "manga_ocr", module)
    return module


class RecordingBackend(OcrBackend):
    name = "recording"

    def __init__(self, result=None, error=None):
        self.result = result or OcrResult("fallback text", language="ja", confidence=0.8)
        self.error = error
        self.calls = []

    def recognize(self, image_bytes, box, language=None, orientation=None):
        self.calls.append({"language": language, "orientation": orientation})
        if self.error:
            raise OcrError(self.error)
        return self.result


# --- 1. model_id is honoured, or refused ---------------------------------
def test_default_model_id_is_used(monkeypatch):
    seen: list[str] = []
    install_fake_manga_ocr(monkeypatch, record=seen)
    backend = MangaOcr()
    result = backend.recognize(png(), BOX)
    assert result.metadata["model"] == MODEL_ID
    assert seen == [MODEL_ID]          # constructed with no override


def test_custom_model_id_is_actually_passed_to_the_model(monkeypatch):
    seen: list[str] = []
    install_fake_manga_ocr(monkeypatch, record=seen)
    backend = MangaOcr(model_id="someone/manga-ocr-finetuned")
    result = backend.recognize(png(), BOX)
    assert seen == ["someone/manga-ocr-finetuned"]
    assert result.metadata["model"] == "someone/manga-ocr-finetuned"


def test_custom_model_id_fails_loudly_when_unsupported(monkeypatch):
    """Metadata must never claim a model that never ran."""
    install_fake_manga_ocr(monkeypatch, accepts_model_id=False)
    backend = MangaOcr(model_id="someone/other-model")
    with pytest.raises(OcrError, match="custom model id"):
        backend.recognize(png(), BOX)
    assert backend.available() is False
    assert "custom model id" in backend.failure


# --- 5. the failure reason survives --------------------------------------
def test_initialization_failure_keeps_its_reason(monkeypatch):
    install_fake_manga_ocr(monkeypatch, raises=RuntimeError("connection refused to huggingface.co"))
    backend = MangaOcr()

    with pytest.raises(OcrError, match="weights download on first use"):
        backend.recognize(png(), BOX)
    assert backend.failure == "connection refused to huggingface.co"
    assert backend.available() is False

    # A second attempt reports the original reason rather than retrying blindly.
    with pytest.raises(OcrError, match="connection refused"):
        backend.load()


def test_missing_package_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setitem(sys.modules, "manga_ocr", None)   # import raises
    backend = MangaOcr()
    assert backend.available() is False
    assert "not installed" in backend.failure


# --- 2. routing and fallback ---------------------------------------------
def test_japanese_prefers_manga_ocr_when_usable(monkeypatch):
    install_fake_manga_ocr(monkeypatch, reads="ありがとう")
    general = RecordingBackend()
    router = OcrRouter(general, {"ja": MangaOcr()})

    assert router.backend_for("ja").name == "manga-ocr"
    assert router.backend_for("ko").name == "recording"
    result = router.recognize(png(), BOX, "ja")
    assert result.text == "ありがとう"
    assert result.metadata["backend"] == "manga-ocr"
    assert general.calls == []


def test_router_falls_back_and_says_so(monkeypatch):
    install_fake_manga_ocr(monkeypatch, raises=RuntimeError("no weights cached"))
    general = RecordingBackend()
    router = OcrRouter(general, {"ja": MangaOcr()}, on_error="fallback")

    result = router.recognize(png(), BOX, "ja")
    assert result.text == "fallback text"
    assert result.metadata["backend"] == "recording"
    assert result.metadata["fell_back_from"] == "manga-ocr"
    assert "no weights cached" in result.metadata["fallback_reason"]
    assert len(general.calls) == 1


def test_router_can_be_told_to_report_instead_of_falling_back(monkeypatch):
    install_fake_manga_ocr(monkeypatch, raises=RuntimeError("no weights cached"))
    router = OcrRouter(RecordingBackend(), {"ja": MangaOcr()}, on_error="error")
    with pytest.raises(OcrError, match="manga-ocr couldn't start"):
        router.recognize(png(), BOX, "ja")


def test_a_failed_backend_is_not_selected_again(monkeypatch):
    install_fake_manga_ocr(monkeypatch, raises=RuntimeError("boom"))
    manga = MangaOcr()
    router = OcrRouter(RecordingBackend(), {"ja": manga})

    router.recognize(png(), BOX, "ja")                 # fails once, falls back
    assert manga.available() is False
    assert router.backend_for("ja").name == "recording"   # no dead-end second time
    assert router.describe()["preferred"]["ja"]["failure"] == "boom"


def test_no_fallback_target_means_the_original_error_surfaces(monkeypatch):
    install_fake_manga_ocr(monkeypatch, raises=RuntimeError("boom"))
    router = OcrRouter(NullOcr(), {"ja": MangaOcr()})
    with pytest.raises(OcrError):
        router.recognize(png(), BOX, "ja")


def test_router_rejects_an_unknown_error_policy():
    with pytest.raises(ValueError):
        OcrRouter(NullOcr(), {}, on_error="explode")


# --- 3. orientation is never invented ------------------------------------
def test_horizontal_japanese_without_an_explicit_orientation_stays_unknown(monkeypatch):
    install_fake_manga_ocr(monkeypatch)
    result = MangaOcr().recognize(png(), BoundingBox(0, 0, 200, 40), "ja")
    assert result.orientation is None                  # not VERTICAL_RL
    assert result.metadata["orientation_source"] == "unknown"


def test_supplied_orientation_is_preserved(monkeypatch):
    install_fake_manga_ocr(monkeypatch)
    backend = MangaOcr()
    for orientation in (TextOrientation.HORIZONTAL, TextOrientation.VERTICAL_RL):
        result = backend.recognize(png(), BOX, "ja", orientation)
        assert result.orientation is orientation
        assert result.metadata["orientation_source"] == "caller"


def test_region_orientation_is_reported_when_the_engine_cannot_say(monkeypatch, tmp_path):
    """The page service falls back to the region's own setting, not a guess."""
    from scanlate.core.types import SegmentKind
    from scanlate.fixtures import build_services
    from scanlate.storage.models import Chapter, Project

    install_fake_manga_ocr(monkeypatch)
    services = build_services(tmp_path / "db", media_root=tmp_path / "media",
                              ocr=OcrRouter(NullOcr(), {"ja": MangaOcr()}))
    with services.uow.transaction():
        services.repository.create_project(Project("p", "P", "ja", "en"))
        services.repository.create_chapter(Chapter("c", "p", 1))
    page = services.pages.import_page("p", "c", 1, png(400, 400))
    segment = services.page_service.create_region(page.id, BoundingBox(0, 0, 180, 220),
                                                  SegmentKind.DIALOGUE, "ja")

    result = services.page_service.ocr_segment(segment.id)
    assert result.ok
    assert result.orientation == "vertical_rl"        # the region's own setting
    assert result.backend == "manga-ocr"


# --- 4. absent confidence is not high confidence -------------------------
def test_manga_ocr_reports_that_it_has_no_confidence_score(monkeypatch):
    install_fake_manga_ocr(monkeypatch)
    backend = MangaOcr()
    assert backend.reports_confidence is False
    result = backend.recognize(png(), BOX, "ja")
    assert result.confidence is None
    assert result.metadata["confidence_available"] is False
    assert result.low_confidence is False             # absent is not low


def test_absent_confidence_reaches_the_api_as_an_explicit_fact(monkeypatch, tmp_path):
    from scanlate.api import schemas
    from scanlate.core.types import SegmentKind
    from scanlate.fixtures import build_services
    from scanlate.storage.models import Chapter, Project

    install_fake_manga_ocr(monkeypatch, reads="うるせぇな")
    services = build_services(tmp_path / "db", media_root=tmp_path / "media",
                              ocr=OcrRouter(NullOcr(), {"ja": MangaOcr()}))
    with services.uow.transaction():
        services.repository.create_project(Project("p", "P", "ja", "en"))
        services.repository.create_chapter(Chapter("c", "p", 1))
    page = services.pages.import_page("p", "c", 1, png(400, 400))
    segment = services.page_service.create_region(page.id, BoundingBox(0, 0, 180, 220),
                                                  SegmentKind.DIALOGUE, "ja")

    payload = schemas.ocr_result(services.page_service.ocr_segment(segment.id))
    assert payload["confidence"] is None
    assert payload["confidence_available"] is False
    assert payload["low_confidence"] is False
    assert payload["backend"] == "manga-ocr"


def test_tesseract_still_reports_confidence():
    from scanlate.imaging.ocr import TesseractOcr
    assert TesseractOcr().reports_confidence is True
