"""Page import → region → OCR → Segment → TranslationPipeline → SegmentView.

The fixture image is generated here, not committed: a few black rectangles and
some rendered text on white. No comic artwork is used.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from scanlate.api.app import create_app
from scanlate.core.types import SegmentKind
from scanlate.fixtures import build_services, seed_demo
from scanlate.imaging.layout import BoundingBox
from scanlate.imaging.ocr import OcrResult
from scanlate.storage.models import Chapter, Status
from tests.test_imaging_corrections import FakeDetector, FakeOcr, png
from scanlate.imaging.detection import DetectedRegion


def fixture_page(width=520, height=720) -> bytes:
    """A stand-in comic page: two bubbles with Latin text, one caption box."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 300, 130], outline="black", width=3)
    draw.text((70, 70), "HELLO THERE", fill="black")
    draw.rectangle([220, 300, 470, 400], outline="black", width=3)
    draw.text((250, 340), "GOODBYE", fill="black")
    draw.rectangle([40, 560, 460, 640], fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path):
    services = build_services(tmp_path / "db.sqlite", media_root=tmp_path / "media")
    seed_demo(services)
    with services.uow.transaction():
        services.repository.create_chapter(Chapter("demo-ch2", "demo", 2, "Imaging"))
    app = create_app(services, seed=False)
    with TestClient(app) as test_client:
        test_client.services = services
        yield test_client


def import_page(client, data=None, number=1, chapter="demo-ch2", replace=False):
    return client.post("/api/projects/demo/pages",
                       files={"file": ("page.png", data or fixture_page(), "image/png")},
                       data={"chapter_id": chapter, "number": str(number),
                             "replace": str(replace).lower()})


# --- import -------------------------------------------------------------
def test_import_stores_dimensions_and_a_stable_reference(client):
    response = import_page(client)
    assert response.status_code == 200
    page = response.json()
    assert (page["width"], page["height"]) == (520, 720)
    assert page["has_image"] is True

    image = client.get(page["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(image.content)) as served:
        assert served.size == (page["width"], page["height"])

    # The reference survives a fresh service reading the same database.
    stored = client.services.repository.get_page(page["id"])
    assert client.services.media.exists(stored.image_path)


def test_duplicate_page_number_is_refused_then_replaceable(client):
    first = import_page(client).json()
    conflict = import_page(client)
    assert conflict.status_code == 409
    assert "already exists" in conflict.json()["detail"]

    # Replacing keeps the logical page and anything attached to it.
    client.post(f"/api/pages/{first['id']}/regions",
                json={"x": 40, "y": 40, "width": 260, "height": 90, "kind": "dialogue"})
    replaced = import_page(client, data=fixture_page(400, 600), replace=True).json()
    assert replaced["id"] == first["id"]
    assert (replaced["width"], replaced["height"]) == (400, 600)
    assert len(client.get(f"/api/pages/{first['id']}").json()["regions"]) == 1


@pytest.mark.parametrize("data,fragment", [
    (b"not an image at all", "isn't an image"),
    (png()[:20], "couldn't be read"),
])
def test_corrupt_and_unsupported_images_are_rejected(client, data, fragment):
    response = client.post("/api/projects/demo/pages",
                           files={"file": ("bad.png", data, "image/png")},
                           data={"chapter_id": "demo-ch2", "number": "5"})
    assert response.status_code == 400
    assert fragment in response.json()["detail"]


# --- regions ------------------------------------------------------------
def test_region_coordinates_round_trip_in_image_space(client):
    page = import_page(client).json()
    created = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 37, "y": 41, "width": 263, "height": 91,
                                "kind": "dialogue", "language": "ja"}).json()

    region = client.get(f"/api/pages/{page['id']}").json()["regions"][0]["region"]
    assert (region["x"], region["y"], region["width"], region["height"]) == (37, 41, 263, 91)
    assert created["language"] == "ja"

    moved = client.patch(f"/api/segments/{created['id']}/region",
                         json={"x": 100, "y": 200, "width": 150, "height": 60})
    assert moved.status_code == 200
    region = client.get(f"/api/pages/{page['id']}").json()["regions"][0]["region"]
    assert (region["x"], region["y"], region["width"], region["height"]) == (100, 200, 150, 60)


def test_region_outside_the_page_is_rejected(client):
    page = import_page(client).json()
    response = client.post(f"/api/pages/{page['id']}/regions",
                           json={"x": 500, "y": 700, "width": 400, "height": 400})
    assert response.status_code == 400
    assert "outside the page" in response.json()["detail"]

    response = client.post(f"/api/pages/{page['id']}/regions",
                           json={"x": 10, "y": 10, "width": 0, "height": 0})
    assert response.status_code == 400


def test_manual_region_lifecycle(client):
    page = import_page(client).json()
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 40, "y": 40, "width": 260, "height": 90}).json()

    client.patch(f"/api/segments/{segment['id']}/region", json={"kind": "sfx"})
    assert client.get(f"/api/segments/{segment['id']}").json()["kind"] == "sfx"

    assert client.delete(f"/api/segments/{segment['id']}/region").json()["outcome"] == "deleted"
    assert client.get(f"/api/pages/{page['id']}").json()["regions"] == []
    assert client.get(f"/api/segments/{segment['id']}").status_code == 404


def test_deleting_an_approved_region_needs_confirmation(client):
    """The documented rule: approved work is detached, never silently discarded."""
    page = import_page(client).json()
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 40, "y": 40, "width": 260, "height": 90,
                                "language": "ja", "source_text": "行くぞ！"}).json()
    client.post(f"/api/segments/{segment['id']}/approve", json={"translation": "Let's go!"})

    refused = client.delete(f"/api/segments/{segment['id']}/region")
    assert refused.status_code == 409
    assert "approved" in refused.json()["detail"]

    forced = client.delete(f"/api/segments/{segment['id']}/region?force=true")
    assert forced.json()["outcome"] == "detached"
    assert client.get(f"/api/pages/{page['id']}").json()["regions"] == []
    kept = client.get(f"/api/segments/{segment['id']}").json()
    assert kept["status"] == "approved" and kept["candidate"] == "Let's go!"


def test_detector_false_positives_can_be_deleted(client):
    page = import_page(client).json()
    client.services.page_service.detector = FakeDetector([
        DetectedRegion(BoundingBox(40, 40, 260, 90)),
        DetectedRegion(BoundingBox(40, 560, 420, 80)),        # the solid black bar
    ])
    detected = client.post(f"/api/pages/{page['id']}/detect").json()
    assert detected["created"] == 2

    bogus = detected["segments"][1]["id"]
    assert client.delete(f"/api/segments/{bogus}/region").json()["outcome"] == "deleted"
    assert len(client.get(f"/api/pages/{page['id']}").json()["regions"]) == 1


def test_zero_detected_regions_is_valid(client):
    page = import_page(client, data=png(300, 300)).json()     # blank white page
    client.services.page_service.detector = FakeDetector([])
    result = client.post(f"/api/pages/{page['id']}/detect").json()
    assert result["available"] is True and result["created"] == 0
    assert "Draw them by hand" in result["message"]

    # Manual creation still works on that page.
    manual = client.post(f"/api/pages/{page['id']}/regions",
                         json={"x": 10, "y": 10, "width": 100, "height": 40})
    assert manual.status_code == 200


# --- OCR ----------------------------------------------------------------
def test_ocr_writes_source_text_onto_the_segment(client):
    page = import_page(client).json()
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 40, "y": 40, "width": 265, "height": 95}).json()

    response = client.post(f"/api/segments/{segment['id']}/ocr").json()
    assert response["result"]["ok"] is True
    assert "HELLO" in response["result"]["text"].upper()
    assert response["segment"]["source"] == response["result"]["text"]


def test_ocr_failure_leaves_the_manual_workflow_available(client):
    page = import_page(client).json()
    client.services.page_service.ocr = FakeOcr(error="engine exploded")
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 40, "y": 40, "width": 260, "height": 90}).json()

    failed = client.post(f"/api/segments/{segment['id']}/ocr").json()
    assert failed["result"]["ok"] is False
    assert "engine exploded" in failed["result"]["error"]

    typed = client.patch(f"/api/segments/{segment['id']}/source",
                         json={"source_text": "うるせぇな…今忙しいんだよ。", "language": "ja"}).json()
    assert typed["segment"]["source"] == "うるせぇな…今忙しいんだよ。"
    assert typed["segment"]["candidate"] == "Please be quiet, I am busy right now."


def test_low_confidence_is_flagged_not_failed(client):
    page = import_page(client).json()
    client.services.page_service.ocr = FakeOcr(OcrResult("うるせぇ", language="ja", confidence=0.2))
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 40, "y": 40, "width": 260, "height": 90}).json()

    result = client.post(f"/api/segments/{segment['id']}/ocr").json()["result"]
    assert result["ok"] is True and result["low_confidence"] is True
    assert result["needs_review"] is True
    assert client.get(f"/api/segments/{segment['id']}").json()["source"] == "うるせぇ"


def test_ocr_page_reads_every_empty_region(client):
    page = import_page(client).json()
    for box in ({"x": 40, "y": 40, "width": 265, "height": 95},
                {"x": 220, "y": 300, "width": 255, "height": 105}):
        client.post(f"/api/pages/{page['id']}/regions", json=box)
    results = client.post(f"/api/pages/{page['id']}/ocr").json()["results"]
    assert len(results) == 2
    assert all(r["ok"] for r in results)


# --- language routing ---------------------------------------------------
def test_japanese_and_korean_regions_reach_their_own_adapters(client):
    page = import_page(client).json()
    japanese = client.post(f"/api/pages/{page['id']}/regions",
                           json={"x": 10, "y": 10, "width": 200, "height": 60,
                                 "language": "ja", "source_text": "うるせぇな…今忙しいんだよ。"}).json()
    korean = client.post(f"/api/pages/{page['id']}/regions",
                         json={"x": 10, "y": 100, "width": 200, "height": 60,
                               "language": "ko", "source_text": "형, 진짜 괜찮아?"}).json()

    assert japanese["features"]["ja.politeness"] == "plain"
    assert japanese["features"]["register"] == "crude"
    assert korean["features"]["ko.speech_level"] == "haeche"
    assert "ja.politeness" not in korean["features"]
    # Mixed languages coexist on one page.
    regions = client.get(f"/api/pages/{page['id']}").json()["regions"]
    assert {r["language"] for r in regions} == {"ja", "ko"}


# --- source text changes ------------------------------------------------
def test_editing_source_text_reruns_the_pipeline(client):
    page = import_page(client).json()
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 10, "y": 10, "width": 200, "height": 60,
                                "language": "ja", "source_text": "行くぞ！"}).json()
    assert segment["candidate"] == "Let's go!"

    updated = client.patch(f"/api/segments/{segment['id']}/source",
                           json={"source_text": "うるせぇな…今忙しいんだよ。"}).json()
    assert updated["segment"]["candidate"] == "Please be quiet, I am busy right now."
    assert any(w["code"] == "register.mismatch" for w in updated["segment"]["warnings"])


def test_approved_translation_is_not_retained_after_a_source_change(client):
    page = import_page(client).json()
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 10, "y": 10, "width": 200, "height": 60,
                                "language": "ja", "source_text": "行くぞ！"}).json()
    client.post(f"/api/segments/{segment['id']}/approve", json={"translation": "Let's go!"})

    changed = client.patch(f"/api/segments/{segment['id']}/source",
                           json={"source_text": "何してるの？"}).json()
    assert changed["reopened"] is True
    assert "no longer applies" in changed["message"]
    assert changed["segment"]["status"] != "approved"
    assert changed["segment"]["candidate"] == "What are you doing?"
    # The old approval survives in memory as a variant, but not as this segment's answer.
    assert client.services.memory.current_variant_for(segment["id"]) is None
    assert any(e["target_text"] == "Let's go!"
               for e in client.get("/api/projects/demo/memory").json())


# --- persistence and end to end ----------------------------------------
def test_page_regions_and_segments_survive_a_restart(client, tmp_path):
    page = import_page(client).json()
    segment = client.post(f"/api/pages/{page['id']}/regions",
                          json={"x": 37, "y": 41, "width": 263, "height": 91,
                                "kind": "narration", "language": "ja",
                                "source_text": "その日、雨は止まなかった。"}).json()
    client.post(f"/api/segments/{segment['id']}/approve",
                json={"translation": "The rain didn't let up that day."})

    # A second process against the same database and media directory.
    reopened = build_services(tmp_path / "db.sqlite", media_root=tmp_path / "media")
    with TestClient(create_app(reopened, seed=False)) as second:
        reloaded = second.get(f"/api/pages/{page['id']}").json()
        assert (reloaded["width"], reloaded["height"]) == (520, 720)
        assert second.get(reloaded["image_url"]).status_code == 200

        region = reloaded["regions"][0]
        assert region["region"] == {"x": 37, "y": 41, "width": 263, "height": 91,
                                    "kind": "caption", "orientation": "horizontal",
                                    "reading_order": 0}
        assert region["kind"] == "narration" and region["language"] == "ja"

        view = second.get(f"/api/segments/{segment['id']}").json()
        assert view["status"] == "approved"
        assert view["candidate"] == "The rain didn't let up that day."
        assert view["page_id"] == page["id"]


def test_end_to_end_image_to_segment_view(client):
    """Image → detected region → OCR → corrected source → SegmentView."""
    page = import_page(client).json()

    detected = client.post(f"/api/pages/{page['id']}/detect").json()
    assert detected["available"] is True
    if not detected["created"]:
        pytest.skip("the detector found nothing on this synthetic page")

    segment_id = detected["segments"][0]["id"]
    ocr = client.post(f"/api/segments/{segment_id}/ocr").json()
    assert ocr["result"]["ok"] or ocr["result"]["error"]

    # Whatever OCR produced, the translator corrects it and the pipeline runs.
    corrected = client.patch(f"/api/segments/{segment_id}/source",
                             json={"source_text": "田中先輩、ちょっといいですか？",
                                   "language": "ja"}).json()["segment"]

    assert corrected["candidate"] == "Tanaka-senpai, do you have a second?"
    assert corrected["language"] == "ja"
    assert corrected["features"]["register"] == "polite"
    assert [(a["start"], a["end"], a["target"]) for a in corrected["annotations"]
            if a["type"] == "term"] == [(0, 2, "Tanaka"), (2, 4, "senpai")]
    assert corrected["page_id"] == page["id"]

    approved = client.post(f"/api/segments/{segment_id}/approve",
                           json={"translation": "Tanaka-senpai, got a sec?"}).json()
    assert approved["segment"]["status"] == "approved"

    # And it is an ordinary segment: it shows up in chapter-wide Review.
    review = client.get("/api/projects/demo/segments").json()["segments"]
    assert any(s["id"] == segment_id and s["status"] == "approved" for s in review)


# --- route thinness -----------------------------------------------------
def test_page_routes_delegate_rather_than_reimplement():
    """Routes resolve, call a service, serialize. No imaging or translation logic."""
    import inspect as py_inspect

    from scanlate.api import pages as routes

    source = py_inspect.getsource(routes)
    for forbidden in ("cv2", "pytesseract", "PIL", "MSER", "TranslationPipeline(",
                      "record_approved", "enforce(", "detect(image"):
        assert forbidden not in source, f"page routes should not contain {forbidden}"
    # Every mutation answers with a pipeline-produced view rather than building one.
    assert source.count("services.pipeline.process") == 2
