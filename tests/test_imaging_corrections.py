"""Regressions for the seven imaging corrections."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from scanlate.core.types import SegmentKind
from scanlate.fixtures import build_services, seed_demo
from scanlate.imaging.detection import (DetectedRegion, DetectorUnavailable, NullDetector,
                                        TextRegionDetector)
from scanlate.imaging.importer import ImageError, MediaStore, inspect
from scanlate.imaging.layout import BoundingBox, TextOrientation
from scanlate.imaging.ocr import OcrBackend, OcrError, OcrResult
from scanlate.storage.models import Chapter, Status


def png(width=60, height=40, colour="white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeDetector(TextRegionDetector):
    name = "fake"

    def __init__(self, regions):
        self.regions = regions

    def detect(self, image_bytes):
        return list(self.regions)


class FakeOcr(OcrBackend):
    """Records what it was asked for, returns whatever it was configured with."""
    name = "fake"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def recognize(self, image_bytes, box, language=None, orientation=None):
        self.calls.append({"box": box, "language": language, "orientation": orientation})
        if self.error:
            raise OcrError(self.error)
        return self.result


@pytest.fixture()
def app(tmp_path):
    services = build_services(tmp_path / "db.sqlite", media_root=tmp_path / "media",
                              detector=NullDetector(), ocr=FakeOcr(OcrResult("テスト")))
    seed_demo(services)
    with services.uow.transaction():
        services.repository.create_chapter(Chapter("ch-img", "demo", 9, "Imaging"))
    return services


def import_page(app, number=1, data=None):
    return app.pages.import_page("demo", "ch-img", number, data or png(400, 600))


# --- 1. media path containment -----------------------------------------
@pytest.mark.parametrize("reference", [
    "../../etc/passwd", "demo/../../etc/passwd", "/etc/passwd", "demo/../x.png",
    "a/b/c", "demo", "", "demo/..", "./demo/x.png",
])
def test_media_store_refuses_traversal(tmp_path, reference):
    store = MediaStore(tmp_path / "media")
    with pytest.raises(ImageError):
        store.path(reference)


def test_media_store_refuses_a_prefix_sibling_directory(tmp_path):
    """`/media-evil` starts with `/media` but is a different directory."""
    root = tmp_path / "media"
    root.mkdir()
    (tmp_path / "media-evil").mkdir()
    store = MediaStore(root)
    with pytest.raises(ImageError):
        store.path("../media-evil/x.png")


def test_media_store_refuses_unsafe_project_ids(tmp_path):
    store = MediaStore(tmp_path / "media")
    for project_id in ("../escape", "a/b", "..", "", "a" * 200):
        with pytest.raises(ImageError):
            store.project_dir(project_id)


def test_media_store_round_trips_a_legitimate_reference(tmp_path):
    store = MediaStore(tmp_path / "media")
    data = png()
    reference = store.store("demo", data, inspect(data))
    assert store.read(reference) == data
    assert store.path(reference).is_relative_to((tmp_path / "media").resolve())


# --- 2. EXIF orientation ------------------------------------------------
def rotated_jpeg() -> bytes:
    """A landscape image tagged as needing a 90° rotation (EXIF orientation 6)."""
    image = Image.new("RGB", (80, 40), "white")
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_exif_rotation_is_baked_into_the_stored_pixels():
    info = inspect(rotated_jpeg())
    # The browser would have displayed it rotated; stored dimensions must match.
    assert (info.width, info.height) == (40, 80)
    assert info.reoriented is True

    with Image.open(io.BytesIO(info.data)) as stored:
        assert stored.size == (40, 80)
        assert stored.getexif().get(274, 1) in (1, None)   # flag consumed, not left behind


def test_imported_page_dimensions_match_the_stored_file(app):
    page = app.pages.import_page("demo", "ch-img", 3, rotated_jpeg())
    assert (page.width, page.height) == (40, 80)
    with Image.open(io.BytesIO(app.media.read(page.image_path))) as stored:
        assert stored.size == (page.width, page.height)


def test_unrotated_images_are_stored_untouched():
    data = png(60, 40)
    info = inspect(data)
    assert info.reoriented is False
    assert info.data == data


# --- 3. overlap within a single detection run ---------------------------
def test_duplicate_detections_in_one_run_are_not_both_kept(app):
    page = import_page(app)
    app.page_service.detector = FakeDetector([
        DetectedRegion(BoundingBox(10, 10, 100, 50)),
        DetectedRegion(BoundingBox(14, 12, 100, 50)),    # same bubble, found twice
        DetectedRegion(BoundingBox(200, 300, 80, 40)),   # a different bubble
    ])
    result = app.page_service.detect_regions(page.id)
    assert len(result.regions) == 2
    boxes = {(s.region.box.x, s.region.box.y) for s in result.regions}
    assert boxes == {(10, 10), (200, 300)}


def test_a_second_run_does_not_duplicate_existing_regions(app):
    page = import_page(app)
    app.page_service.detector = FakeDetector([DetectedRegion(BoundingBox(10, 10, 100, 50))])
    assert len(app.page_service.detect_regions(page.id).regions) == 1
    assert app.page_service.detect_regions(page.id).regions == []
    assert len(app.repository.list_page_segments(page.id)) == 1


# --- 4. language change invalidates like a text change ------------------
def test_changing_source_language_reopens_an_approved_segment(app):
    page = import_page(app)
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 100, 50),
                                             SegmentKind.DIALOGUE, "ja", "行くぞ！")
    app.pipeline.process(app.repository.get_segment(segment.id))
    app.approvals.approve(segment.id, "Let's go!")
    entry = app.memory.current_variant_for(segment.id)
    assert entry is not None

    change = app.page_service.set_source_text(segment.id, "行くぞ！", language="ko")

    assert change.reopened is True
    assert change.changed == "source language"
    assert "no longer applies" in change.message
    reloaded = app.repository.get_segment(segment.id)
    assert reloaded.state.status is Status.MACHINE
    assert reloaded.language == "ko"
    assert app.memory.current_variant_for(segment.id) is None     # occurrence released
    assert app.memory.occurrence_count(entry.id) == 0             # variant still on record


def test_changing_language_through_update_region_also_invalidates(app):
    page = import_page(app)
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 100, 50),
                                             SegmentKind.DIALOGUE, "ja", "행복")
    app.approvals.approve(segment.id, "Happiness")
    app.page_service.update_region(segment.id, language="ko")

    reloaded = app.repository.get_segment(segment.id)
    assert reloaded.language == "ko"
    assert reloaded.state.status is Status.MACHINE
    assert app.memory.current_variant_for(segment.id) is None


def test_setting_the_same_language_changes_nothing(app):
    page = import_page(app)
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 100, 50),
                                             SegmentKind.DIALOGUE, "ja", "行くぞ！")
    app.approvals.approve(segment.id, "Let's go!")
    change = app.page_service.set_source_text(segment.id, "行くぞ！", language="ja")
    assert change.reopened is False
    assert app.repository.get_segment(segment.id).state.status is Status.APPROVED


# --- 5. low confidence and reopening are independent --------------------
def test_low_confidence_on_a_fresh_region_reopens_nothing(app):
    page = import_page(app)
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 100, 50))
    app.page_service.ocr = FakeOcr(OcrResult("うるせぇ", language="ja", confidence=0.21))

    result = app.page_service.ocr_segment(segment.id)
    assert result.low_confidence is True
    assert result.reopened is False
    assert result.needs_review is True


def test_confident_ocr_can_still_reopen_an_approved_segment(app):
    page = import_page(app)
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 100, 50),
                                             SegmentKind.DIALOGUE, "ja", "行くぞ！")
    app.approvals.approve(segment.id, "Let's go!")
    app.page_service.ocr = FakeOcr(OcrResult("何してるの？", language="ja", confidence=0.97))

    result = app.page_service.ocr_segment(segment.id)
    assert result.low_confidence is False
    assert result.reopened is True
    assert result.confidence == 0.97


# --- 6. unavailable detector vs. zero regions ---------------------------
def test_detector_unavailable_is_distinct_from_finding_nothing(app):
    page = import_page(app)

    app.page_service.detector = NullDetector()
    unavailable = app.page_service.detect_regions(page.id)
    assert unavailable.available is False
    assert unavailable.regions == []
    assert "by hand" in unavailable.message

    app.page_service.detector = FakeDetector([])
    empty = app.page_service.detect_regions(page.id)
    assert empty.available is True
    assert empty.regions == []
    assert "No text regions were found" in empty.message


def test_manual_regions_work_in_both_cases(app):
    page = import_page(app)
    for detector in (NullDetector(), FakeDetector([])):
        app.page_service.detector = detector
        app.page_service.detect_regions(page.id)
        segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 50, 20),
                                                 SegmentKind.DIALOGUE, "ja", "手動")
        assert app.repository.get_segment(segment.id).source_text == "手動"


def test_null_detector_reports_itself_unavailable():
    assert NullDetector().available() is False
    with pytest.raises(DetectorUnavailable):
        NullDetector().detect(png())


# --- 7. explicit orientation controls OCR -------------------------------
def test_saved_orientation_overrides_the_shape_guess(app):
    page = import_page(app)
    # A wide box would be guessed horizontal; the user says it's vertical.
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 200, 60))
    app.page_service.update_region(segment.id, orientation=TextOrientation.VERTICAL_RL)

    fake = FakeOcr(OcrResult("縦書き", language="ja", confidence=0.9))
    app.page_service.ocr = fake
    app.page_service.ocr_segment(segment.id)

    assert fake.calls[-1]["orientation"] is TextOrientation.VERTICAL_RL


def test_geometry_supplies_the_initial_orientation(app):
    page = import_page(app)
    tall = app.page_service.create_region(page.id, BoundingBox(0, 0, 40, 200))
    assert app.repository.get_segment(tall.id).region.orientation is TextOrientation.VERTICAL_RL

    fake = FakeOcr(OcrResult("縦", language="ja"))
    app.page_service.ocr = fake
    app.page_service.ocr_segment(tall.id)
    assert fake.calls[-1]["orientation"] is TextOrientation.VERTICAL_RL


def test_orientation_survives_a_resize(app):
    page = import_page(app)
    segment = app.page_service.create_region(page.id, BoundingBox(0, 0, 200, 60))
    app.page_service.update_region(segment.id, orientation=TextOrientation.VERTICAL_RL)
    app.page_service.update_region(segment.id, box=BoundingBox(10, 10, 220, 70))
    assert app.repository.get_segment(segment.id).region.orientation is TextOrientation.VERTICAL_RL
