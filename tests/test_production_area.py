"""Automatic production-area inference: growing a tight text box outward
into a safe layout/cleanup area, independent of BubbleCleaner's own
apply-to-a-given-box checks (see test_page_render.py for those).
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from scanlate.imaging.layout import BoundingBox, Polygon
from scanlate.imaging.production_area import _step, infer_production_area

WHITE = (250, 250, 250)
INK = (0, 0, 0)
DARK = (20, 20, 20)


def flat_canvas(size=(1000, 1000), fill=WHITE) -> Image.Image:
    return Image.new("RGB", size, fill)


def stipple(image: Image.Image, box: BoundingBox, ratio: float, background=WHITE, ink=INK) -> None:
    step = max(1, round(1 / ratio))
    i = 0
    for x in range(box.x, box.x + box.width):
        for y in range(box.y, box.y + box.height):
            image.putpixel((x, y), ink if i % step == 0 else background)
            i += 1


def test_step_scales_with_region_size_not_a_fixed_pixel_value():
    assert _step(100) > _step(20)
    assert _step(20) == max(2, round(0.08 * 20))
    assert _step(100) == max(2, round(0.08 * 100))


def test_no_confident_background_in_the_text_box_is_an_exception():
    image = flat_canvas()
    text_box = BoundingBox(450, 450, 100, 60)
    ImageDraw.Draw(image).rectangle(
        [text_box.x, text_box.y, text_box.x + text_box.width - 1, text_box.y + text_box.height - 1],
        fill=DARK)
    result = infer_production_area(image, text_box, 1000, 1000)
    assert result.ok is False
    assert result.box is None
    assert "background" in result.reason


def test_confident_light_bubble_grows_on_all_sides_up_to_the_relative_cap():
    image = flat_canvas()
    text_box = BoundingBox(450, 470, 100, 60)
    stipple(image, text_box, ratio=0.2)

    result = infer_production_area(image, text_box, 1000, 1000)
    assert result.ok is True
    # Each side may grow at most 2x the box's own dimension on that axis —
    # a flat page with nothing to stop it should reach exactly that cap.
    assert result.box == BoundingBox(450 - 200, 470 - 120, 100 + 400, 60 + 240)


def test_growth_stops_asymmetrically_at_a_dark_boundary_on_one_side_only():
    image = flat_canvas()
    text_box = BoundingBox(100, 450, 100, 60)
    stipple(image, text_box, ratio=0.2)
    # A dark strip immediately to the right of the text box, as if a bubble
    # border or neighbouring panel started right there.
    ImageDraw.Draw(image).rectangle([200, 0, 260, 999], fill=DARK)

    result = infer_production_area(image, text_box, 1000, 1000)
    assert result.ok is True
    assert result.box.x + result.box.width == 200          # right: no growth at all
    assert result.box.x < text_box.x                        # left: grew normally
    assert result.box.y < text_box.y                        # top: grew normally
    assert result.box.y + result.box.height > text_box.y + text_box.height  # bottom: grew normally


def test_growth_is_capped_at_the_page_edge_not_just_the_relative_cap():
    image = flat_canvas(size=(300, 300))
    text_box = BoundingBox(10, 10, 100, 60)  # near the top-left corner
    stipple(image, text_box, ratio=0.2)

    result = infer_production_area(image, text_box, 300, 300)
    assert result.ok is True
    assert result.box.x == 0    # can't grow past the page edge
    assert result.box.y == 0


def test_mask_based_background_estimate_succeeds_where_the_heuristic_fails():
    # Half the text box is solid ink, half is solid background -- enough ink
    # that the no-mask "brightest 55%" heuristic is forced to include some of
    # it (55% > the 40% that's actually background), while a real mask can
    # isolate the true background exactly.
    image = flat_canvas()
    text_box = BoundingBox(400, 400, 100, 100)
    ImageDraw.Draw(image).rectangle([400, 400, 459, 499], fill=INK)          # 60% ink
    ImageDraw.Draw(image).rectangle([460, 400, 499, 499], fill=WHITE)        # 40% background
    mask = Polygon([(400, 400), (460, 400), (460, 500), (400, 500)])

    without_mask = infer_production_area(image, text_box, 1000, 1000, mask=None)
    assert without_mask.ok is False

    with_mask = infer_production_area(image, text_box, 1000, 1000, mask=mask)
    assert with_mask.ok is True
    assert with_mask.box.width > text_box.width


def test_final_sanity_check_rejects_a_margin_that_drifts_from_the_background():
    # Each individual growth band is tested against the fixed original
    # background estimate, so a band-by-band pass can't itself drift -- but a
    # margin built from bands that are each barely light enough, alternating
    # just above and below the acceptance line, still needs to look like one
    # coherent background as a WHOLE. Simulate that by making the margin
    # region a checkerboard of two shades that are each individually inside
    # the per-band tolerance yet, sampled together, exceed the flatness
    # threshold as a combined population.
    image = flat_canvas()
    text_box = BoundingBox(450, 470, 100, 60)
    stipple(image, text_box, ratio=0.2)
    # A checkerboard immediately surrounding the box: alternating light
    # shades whose combined std-dev is well over the flatness threshold,
    # even though many individual thin bands might scrape by.
    for x in range(300, 700):
        for y in range(300, 700):
            if BoundingBox(450, 470, 100, 60).x <= x < 550 and 470 <= y < 530:
                continue
            image.putpixel((x, y), (250, 250, 250) if (x + y) % 2 == 0 else (180, 180, 180))

    result = infer_production_area(image, text_box, 1000, 1000)
    # Either individual bands already reject this (textured, not flat), or —
    # if some bands scraped by — the final whole-margin check must.
    assert result.ok is False
