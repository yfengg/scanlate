"""Preview and export: service/API integration for rendering approved regions
onto the real page image, using the real HTTP API.

Distinct from tests/test_page_render.py, which exercises the stateless
rendering core (cleanup/typeset/SFX) with no database. Here the question is
wiring: does the service pick only approved segments, does preview leave no
file behind, does export write the right bytes to the right place without
touching the source, and does render-setting persistence round-trip.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from scanlate.api.app import create_app
from scanlate.fixtures import build_services, seed_demo
from scanlate.storage.models import Chapter

from tests.test_page_workflow import fixture_page, import_page

# fixture_page() is 520x720 with illustrated bubbles at y 40-130 and
# y 300-400, and a filled caption box at y 560-640. This strip is plain white
# background, clear of all of that — a safe area for regions meant to render
# successfully.
BLANK_AREA = (40, 440, 200, 60)
OTHER_BLANK_AREA = (260, 440, 200, 60)
# Small enough that a longer translation needs the font stepped down to fit
# after cleanup's inset shrinks the available area further.
SNUG_BLANK_AREA = (40, 440, 140, 50)
# The exact box fixture_page() draws its first bubble's outline and "HELLO
# THERE" text in — sampling this always finds visible ink, so cleanup must
# flag it rather than blank it.
BUBBLE_WITH_TEXT = (40, 40, 260, 90)


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


def create_region(client, page_id, box, kind="dialogue", source_text="placeholder"):
    body = {"x": box[0], "y": box[1], "width": box[2], "height": box[3],
            "kind": kind, "source_text": source_text}
    response = client.post(f"/api/pages/{page_id}/regions", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def approve(client, segment_id, translation):
    response = client.post(f"/api/segments/{segment_id}/approve",
                           json={"translation": translation})
    assert response.status_code == 200, response.text
    return response.json()


# --- preview -------------------------------------------------------------
def test_preview_png_renders_an_approved_region_over_a_blank_area(client):
    page = import_page(client).json()
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Hi!")

    response = client.get(f"/api/pages/{page['id']}/preview.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    image = Image.open(io.BytesIO(response.content))
    assert image.size == (520, 720)

    report = client.get(f"/api/pages/{page['id']}/preview").json()
    assert report["flagged"] == []
    assert any(r["segment_id"] == region["id"] and r["rendered"] for r in report["results"])


def test_preview_never_writes_an_export_file(client):
    page = import_page(client).json()
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Hi!")

    client.get(f"/api/pages/{page['id']}/preview.png")
    client.get(f"/api/pages/{page['id']}/preview")

    exports_dir = client.services.media.root / "demo" / "exports"
    assert not exports_dir.exists()


def test_preview_flags_a_region_that_still_has_visible_text(client):
    page = import_page(client).json()
    region = create_region(client, page["id"], BUBBLE_WITH_TEXT)
    approve(client, region["id"], "Hi there!")

    report = client.get(f"/api/pages/{page['id']}/preview").json()
    assert len(report["flagged"]) == 1
    assert report["flagged"][0]["segment_id"] == region["id"]
    assert report["flagged"][0]["reason"]


def test_only_approved_segments_are_rendered(client):
    page = import_page(client).json()
    approved = create_region(client, page["id"], BLANK_AREA)
    approve(client, approved["id"], "Hi!")
    unapproved = create_region(client, page["id"], OTHER_BLANK_AREA)

    report = client.get(f"/api/pages/{page['id']}/preview").json()
    ids = {r["segment_id"] for r in report["results"]}
    assert approved["id"] in ids
    assert unapproved["id"] not in ids


# --- export ----------------------------------------------------------------
def test_export_writes_a_png_without_touching_the_source(client):
    page = import_page(client).json()
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Hi!")

    source = client.services.repository.get_page(page["id"])
    source_before = client.services.media.read(source.image_path)

    response = client.post(f"/api/pages/{page['id']}/export")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report"]["flagged"] == []

    assert client.services.media.read(source.image_path) == source_before


def test_export_uses_a_predictable_filename_from_chapter_and_page_identity(client):
    page = import_page(client, number=3).json()  # chapter demo-ch2, number 2
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Hi!")

    response = client.post(f"/api/pages/{page['id']}/export")
    assert response.json()["reference"] == "demo/exports/chapter-2-page-3-translated.png"


def test_exported_png_matches_the_original_dimensions(client):
    page = import_page(client).json()
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Hi!")

    reference = client.post(f"/api/pages/{page['id']}/export").json()["reference"]
    exported_path = client.services.media.root / reference
    assert exported_path.exists()
    with Image.open(exported_path) as exported:
        assert exported.size == (520, 720)


def test_reexporting_after_an_edit_overwrites_only_its_own_previous_export(client):
    page = import_page(client).json()
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Hi!")
    first = client.post(f"/api/pages/{page['id']}/export").json()

    approve(client, region["id"], "A rather longer greeting that changes the rendered pixels!")
    second = client.post(f"/api/pages/{page['id']}/export").json()

    assert first["reference"] == second["reference"]


# --- render-setting persistence --------------------------------------------
def test_render_settings_persist_after_a_successful_preview(client):
    region = create_region_on_new_page(client, SNUG_BLANK_AREA)
    approve(client, region["id"], "This text needs to shrink a bit to fit.")
    page_id = region["page_id"]

    client.get(f"/api/pages/{page_id}/preview.png")
    segment = client.services.repository.get_segment(region["id"])
    assert segment.render is not None
    assert segment.render.font_id == "lato-regular"
    assert segment.render.font_size < 16.0

    fitted_size = segment.render.font_size
    client.get(f"/api/pages/{page_id}/preview.png")
    assert client.services.repository.get_segment(region["id"]).render.font_size == fitted_size


def create_region_on_new_page(client, box):
    page = import_page(client).json()
    return create_region(client, page["id"], box)


# --- unsupported target language -------------------------------------------
def test_unsupported_target_language_is_flagged_end_to_end(client):
    project = client.post("/api/projects", json={
        "name": "French test", "default_source_language": "ja", "target_language": "fr"}).json()
    chapter = client.post(f"/api/projects/{project['id']}/chapters", json={"number": 1}).json()
    page = client.post(f"/api/projects/{project['id']}/pages",
                       files={"file": ("p.png", fixture_page(), "image/png")},
                       data={"chapter_id": chapter["id"], "number": "1"}).json()
    region = create_region(client, page["id"], BLANK_AREA)
    approve(client, region["id"], "Bonjour !")

    report = client.get(f"/api/pages/{page['id']}/preview").json()
    assert len(report["flagged"]) == 1
    assert "fr" in report["flagged"][0]["reason"]
