"""Rendering core: bubble cleanup, deterministic typesetting, SFX captions.

Pure-image unit tests, no database or API — the service/API/frontend wiring
that turns this into "Preview"/"Export" is a later step. Fixture images are
generated here, not committed, same convention as test_page_workflow.py.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from scanlate.imaging.cleanup import BubbleCleaner
from scanlate.imaging.layout import BoundingBox, Polygon, RegionKind, RenderSettings
from scanlate.imaging.render import PageRenderer, RenderableRegion, _draw_lines
from scanlate.imaging.typeset import (DEFAULT_FONT_ID, FONT_FILES, FONTS_DIR,
                                     LatinHorizontalTypesetter, LayoutResult,
                                     TextLayoutEngine, load_font)

WHITE_BOX = BoundingBox(40, 40, 200, 100)


def assert_unchanged_outside(before: Image.Image, after: Image.Image, box: BoundingBox) -> None:
    """Every pixel outside ``box`` must be identical; ``box`` itself may differ."""
    w, h = before.size

    def strip(im, x0, y0, x1, y1):
        return list(im.crop((x0, y0, x1, y1)).getdata())

    assert strip(before, 0, 0, w, box.y) == strip(after, 0, 0, w, box.y)                       # above
    assert strip(before, 0, box.y + box.height, w, h) == strip(after, 0, box.y + box.height, w, h)  # below
    assert strip(before, 0, box.y, box.x, box.y + box.height) == \
           strip(after, 0, box.y, box.x, box.y + box.height)                                    # left
    assert strip(before, box.x + box.width, box.y, w, box.y + box.height) == \
           strip(after, box.x + box.width, box.y, w, box.y + box.height)                        # right


def stippled_page(box: BoundingBox, background=(250, 250, 250), ink=(15, 15, 15),
                  foreground_ratio=0.2, page_size=(400, 300)) -> Image.Image:
    """A light background with a deterministic, evenly-scattered fraction of
    dark "ink" pixels inside ``box`` — a stand-in for real glyph coverage
    without needing to render actual text into the fixture."""
    image = Image.new("RGB", page_size, (255, 255, 255))
    step = max(1, round(1 / foreground_ratio))
    i = 0
    for x in range(box.x, box.x + box.width):
        for y in range(box.y, box.y + box.height):
            image.putpixel((x, y), ink if i % step == 0 else background)
            i += 1
    return image


def flat_page(fill=(255, 255, 255), box=WHITE_BOX, box_fill=None) -> Image.Image:
    image = Image.new("RGB", (400, 300), fill)
    if box_fill is not None:
        for x in range(box.x, box.x + box.width):
            for y in range(box.y, box.y + box.height):
                image.putpixel((x, y), box_fill)
    return image


def checkerboard_page(box=WHITE_BOX) -> Image.Image:
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    for x in range(box.x, box.x + box.width):
        for y in range(box.y, box.y + box.height):
            image.putpixel((x, y), (240, 240, 240) if (x + y) % 2 == 0 else (10, 10, 10))
    return image


# --- bundled font --------------------------------------------------------
def test_bundled_font_file_is_present_and_loadable():
    path = FONT_FILES[DEFAULT_FONT_ID]
    assert path.exists(), f"bundled font missing: {path}"
    font = load_font(DEFAULT_FONT_ID, 16.0)
    assert font.getbbox("A")[2] > 0  # a real glyph, not a broken/empty font


def test_render_settings_default_to_the_bundled_font():
    assert RenderSettings().font_id == DEFAULT_FONT_ID


def test_unknown_font_id_falls_back_to_the_bundled_default_rather_than_erroring():
    font = load_font("does-not-exist", 16.0)
    assert font.getbbox("A") == load_font(DEFAULT_FONT_ID, 16.0).getbbox("A")


def test_license_is_bundled_alongside_the_font():
    license_text = (FONTS_DIR / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License" in license_text
    assert "Lato" in license_text


def test_layout_and_drawing_use_the_same_font_so_measured_width_is_accurate():
    # If layout measured with one font but rendering drew with another, a
    # line judged "fits" could still overflow visually. Same call, same font.
    box = BoundingBox(0, 0, 300, 100)
    render = RenderSettings(font_size=16.0)
    layout = LatinHorizontalTypesetter().layout("Hello there.", box, render)
    measured = load_font(render.font_id, layout.font_size).getbbox(layout.lines[0])
    assert measured[2] > 0


# --- BubbleCleaner -------------------------------------------------------
def test_flat_white_background_is_judged_safe():
    cleaner = BubbleCleaner()
    result = cleaner.inspect(flat_page(), WHITE_BOX)
    assert result.ok is True
    assert result.cleaned_box is not None
    # The inset never grows the box and never touches its edge.
    assert result.cleaned_box.x > WHITE_BOX.x
    assert result.cleaned_box.y > WHITE_BOX.y
    assert result.cleaned_box.width < WHITE_BOX.width
    assert result.fill_color == (255, 255, 255)


def test_flat_off_white_background_is_still_safe():
    cleaner = BubbleCleaner()
    result = cleaner.inspect(flat_page(box_fill=(250, 248, 245)), WHITE_BOX)
    assert result.ok is True
    assert result.fill_color == (250, 248, 245)


def test_textured_background_is_flagged_not_cleaned():
    cleaner = BubbleCleaner()
    result = cleaner.inspect(checkerboard_page(), WHITE_BOX)
    assert result.ok is False
    assert "flat" in result.reason
    assert result.cleaned_box is None


def test_uniform_dark_background_is_flagged_not_cleaned():
    cleaner = BubbleCleaner()
    result = cleaner.inspect(flat_page(box_fill=(20, 20, 20)), WHITE_BOX)
    assert result.ok is False
    assert "light" in result.reason


def test_region_too_small_to_inset_is_flagged():
    cleaner = BubbleCleaner()
    result = cleaner.inspect(flat_page(), BoundingBox(10, 10, 4, 4))
    assert result.ok is False
    assert "too small" in result.reason


def test_apply_only_touches_the_inset_not_the_whole_box():
    box = WHITE_BOX
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Red right up to the region's own edge, white only inside the inset —
    # if apply() blanked the whole box instead of just the inset, the red
    # margin would disappear too.
    draw.rectangle([box.x, box.y, box.x + box.width - 1, box.y + box.height - 1], fill=(255, 0, 0))
    cleaner = BubbleCleaner()
    inset = cleaner._inset(box)
    draw.rectangle([inset.x, inset.y, inset.x + inset.width - 1, inset.y + inset.height - 1],
                   fill=(250, 250, 250))

    result = cleaner.inspect(image, box)
    assert result.ok is True
    cleaner.apply(image, result)

    assert image.getpixel((box.x, box.y)) == (255, 0, 0)          # margin: untouched
    assert image.getpixel((inset.x + 1, inset.y + 1)) == result.fill_color  # inset: filled


def test_apply_is_a_no_op_when_inspect_said_unsafe():
    image = checkerboard_page()
    before = image.copy()
    cleaner = BubbleCleaner()
    result = cleaner.inspect(image, WHITE_BOX)
    cleaner.apply(image, result)
    assert list(image.getdata()) == list(before.getdata())


# --- BubbleCleaner: explicit production area (apply_inset=False) ---------
# The box is expected to contain real glyph ink here — that's the whole
# point of an explicit area — so these use the robust-background path
# instead of requiring the entire rectangle to already be flat.
def test_light_background_with_source_glyphs_succeeds_with_explicit_geometry():
    cleaner = BubbleCleaner()
    image = stippled_page(WHITE_BOX, background=(250, 250, 250), ink=(10, 10, 10),
                          foreground_ratio=0.2)
    result = cleaner.inspect(image, WHITE_BOX, apply_inset=False)
    assert result.ok is True
    # No inset applied on top of an explicit selection: the cleaned area is
    # exactly the box given, not a shrunk version of it.
    assert result.cleaned_box == WHITE_BOX
    # The fill colour reflects the true background, not tinted by the ink
    # that was included in the sample.
    assert result.fill_color == (250, 250, 250)


def test_textured_illustrated_area_still_flags_with_explicit_geometry():
    cleaner = BubbleCleaner()
    result = cleaner.inspect(checkerboard_page(), WHITE_BOX, apply_inset=False)
    assert result.ok is False


def test_excessive_non_background_coverage_still_flags_with_explicit_geometry():
    cleaner = BubbleCleaner()
    # Only a minority of a bubble's interior should ever be ink; a rectangle
    # that's mostly dark reads as art or a panel, not text on a flat fill.
    image = stippled_page(WHITE_BOX, background=(250, 250, 250), ink=(10, 10, 10),
                          foreground_ratio=0.6)
    result = cleaner.inspect(image, WHITE_BOX, apply_inset=False)
    assert result.ok is False
    assert "coverage" in result.reason


def test_dark_background_still_flags_with_explicit_geometry():
    cleaner = BubbleCleaner()
    image = stippled_page(WHITE_BOX, background=(20, 20, 20), ink=(0, 0, 0), foreground_ratio=0.1)
    result = cleaner.inspect(image, WHITE_BOX, apply_inset=False)
    assert result.ok is False
    assert "light" in result.reason


# --- BubbleCleaner: real segmentation mask (ground truth) -----------------
def test_mask_takes_precedence_and_is_not_inset():
    # Ink fills a strip on the left of the box (a minority of it, like real
    # glyphs in a bubble); the mask marks exactly that strip, so the rest
    # (pure background) is what gets sampled.
    image = flat_page(box_fill=(255, 255, 255))
    ink_edge = WHITE_BOX.x + round(WHITE_BOX.width * 0.2)
    for x in range(WHITE_BOX.x, ink_edge):
        for y in range(WHITE_BOX.y, WHITE_BOX.y + WHITE_BOX.height):
            image.putpixel((x, y), (10, 10, 10))
    mask = Polygon([(WHITE_BOX.x, WHITE_BOX.y), (ink_edge, WHITE_BOX.y),
                    (ink_edge, WHITE_BOX.y + WHITE_BOX.height), (WHITE_BOX.x, WHITE_BOX.y + WHITE_BOX.height)])

    cleaner = BubbleCleaner()
    result = cleaner.inspect(image, WHITE_BOX, apply_inset=True, mask=mask)  # mask wins over apply_inset
    assert result.ok is True
    assert result.cleaned_box == WHITE_BOX          # no inset on top of a mask, same as an explicit area
    assert result.fill_color == (255, 255, 255)      # true background, unaffected by the ink half


def test_mask_covering_too_much_of_the_box_still_flags():
    image = flat_page(box_fill=(255, 255, 255))
    # Mask covers nearly the whole box: almost nothing left to call background.
    mask = Polygon([(WHITE_BOX.x, WHITE_BOX.y), (WHITE_BOX.x + WHITE_BOX.width, WHITE_BOX.y),
                    (WHITE_BOX.x + WHITE_BOX.width, WHITE_BOX.y + WHITE_BOX.height - 5),
                    (WHITE_BOX.x, WHITE_BOX.y + WHITE_BOX.height - 5)])
    cleaner = BubbleCleaner()
    result = cleaner.inspect(image, WHITE_BOX, mask=mask)
    assert result.ok is False
    assert "coverage" in result.reason


def test_mask_over_a_dark_background_still_flags():
    image = flat_page(box_fill=(20, 20, 20))
    mask = Polygon([(WHITE_BOX.x, WHITE_BOX.y), (WHITE_BOX.x + 10, WHITE_BOX.y),
                    (WHITE_BOX.x + 10, WHITE_BOX.y + 10), (WHITE_BOX.x, WHITE_BOX.y + 10)])
    cleaner = BubbleCleaner()
    result = cleaner.inspect(image, WHITE_BOX, mask=mask)
    assert result.ok is False
    assert "light" in result.reason


def test_legacy_path_is_unaffected_by_apply_inset_default():
    # apply_inset defaults to True: existing callers that never pass it get
    # exactly the pre-existing behaviour.
    cleaner = BubbleCleaner()
    assert cleaner.inspect(flat_page(), WHITE_BOX).cleaned_box != WHITE_BOX  # still an inset


# --- TextLayoutEngine / LatinHorizontalTypesetter ------------------------
def test_short_text_fits_at_the_requested_size():
    layout = LatinHorizontalTypesetter().layout(
        "Hi there.", BoundingBox(0, 0, 300, 100), RenderSettings(font_size=16.0))
    assert layout.fits is True
    assert layout.font_size == 16.0
    assert layout.lines == ["Hi there."]


def test_long_text_wraps_and_shrinks_to_fit():
    text = "This translation is much too long to fit on a single line at full size."
    layout = LatinHorizontalTypesetter().layout(
        text, BoundingBox(0, 0, 120, 80), RenderSettings(font_size=16.0))
    assert layout.fits is True
    assert len(layout.lines) > 1
    assert layout.font_size < 16.0
    assert " ".join(layout.lines) == text


def test_text_that_never_fits_is_flagged_not_overflowed():
    text = "Absolutely nothing about this sentence is going to fit in here."
    layout = LatinHorizontalTypesetter().layout(
        text, BoundingBox(0, 0, 20, 15), RenderSettings(font_size=16.0))
    assert layout.fits is False
    assert layout.unsupported is False
    assert "minimum font size" in layout.reason


def test_empty_translation_lays_out_as_no_lines():
    layout = LatinHorizontalTypesetter().layout(
        "", BoundingBox(0, 0, 200, 100), RenderSettings())
    assert layout.fits is True
    assert layout.lines == []


def test_unsupported_target_language_is_flagged_not_typeset_as_english():
    engine = TextLayoutEngine()
    layout = engine.layout("何か", BoundingBox(0, 0, 200, 100), RenderSettings(),
                           target_language="ja")
    assert layout.fits is False
    assert layout.unsupported is True
    assert "ja" in layout.reason


def test_registering_a_new_strategy_is_how_another_language_is_added():
    class AlwaysFits(LatinHorizontalTypesetter):
        pass  # a stand-in for a future non-Latin strategy

    engine = TextLayoutEngine(strategies={"en": LatinHorizontalTypesetter(), "fr": AlwaysFits()})
    layout = engine.layout("Bonjour", BoundingBox(0, 0, 300, 100), RenderSettings(),
                           target_language="fr")
    assert layout.fits is True


# --- PageRenderer ----------------------------------------------------------
def region(text="Hello there.", kind=RegionKind.SPEECH_BUBBLE, box=WHITE_BOX,
          render=None, target_language="en", production_box=None) -> RenderableRegion:
    return RenderableRegion("seg-1", kind, box, text, render or RenderSettings(), target_language,
                            production_box)


def test_render_cleans_and_typesets_a_safe_bubble():
    image = flat_page()
    out, report = PageRenderer().render(image, [region()])
    assert report.flagged == []
    assert report.results[0].rendered is True
    # Something in the box changed (the drawn text), but the page is still
    # the same size and format.
    assert out.size == image.size
    assert list(out.crop((WHITE_BOX.x, WHITE_BOX.y, WHITE_BOX.x + WHITE_BOX.width,
                          WHITE_BOX.y + WHITE_BOX.height)).getdata()) != \
           list(image.crop((WHITE_BOX.x, WHITE_BOX.y, WHITE_BOX.x + WHITE_BOX.width,
                            WHITE_BOX.y + WHITE_BOX.height)).getdata())


def test_a_successful_render_reports_the_settings_that_actually_worked():
    tiny_box = BoundingBox(40, 40, 90, 60)
    long_text = "This one needs a smaller font to fit."
    out, report = PageRenderer().render(flat_page(box=tiny_box), [
        region(text=long_text, box=tiny_box, render=RenderSettings(font_size=16.0))])
    result = report.results[0]
    assert result.rendered is True
    assert result.applied_render is not None
    assert result.applied_render.font_size == result.layout.font_size
    assert result.applied_render.font_size < 16.0  # it had to shrink to fit


def test_a_flagged_region_reports_no_applied_render():
    out, report = PageRenderer().render(checkerboard_page(), [region()])
    assert report.results[0].applied_render is None


def test_render_leaves_an_unsafe_bubble_completely_untouched():
    image = checkerboard_page()
    before = image.copy()
    out, report = PageRenderer().render(image, [region(box=WHITE_BOX)])
    assert len(report.flagged) == 1
    assert report.flagged[0].reason is not None
    crop_before = before.crop((WHITE_BOX.x, WHITE_BOX.y, WHITE_BOX.x + WHITE_BOX.width,
                               WHITE_BOX.y + WHITE_BOX.height))
    crop_after = out.crop((WHITE_BOX.x, WHITE_BOX.y, WHITE_BOX.x + WHITE_BOX.width,
                           WHITE_BOX.y + WHITE_BOX.height))
    assert list(crop_before.getdata()) == list(crop_after.getdata())


def test_render_flags_text_that_does_not_fit_without_drawing_anything():
    image = flat_page(box=BoundingBox(40, 40, 20, 15), box_fill=(255, 255, 255))
    tiny_box = BoundingBox(40, 40, 20, 15)
    before = image.copy()
    long_text = "Absolutely nothing about this sentence is going to fit in here."
    out, report = PageRenderer().render(image, [region(text=long_text, box=tiny_box)])
    assert len(report.flagged) == 1
    # Cleanup itself may have been judged safe, but nothing is drawn/blanked
    # unless the text also fits — a flagged region stays pristine.
    assert list(before.getdata()) == list(out.getdata())


def test_render_never_cleans_sfx_and_draws_a_caption_below_it():
    sfx_box = BoundingBox(100, 100, 60, 40)
    image = flat_page(box=sfx_box, box_fill=(30, 30, 30))  # dark, textured-ish stand-in for art
    before = image.copy()
    out, report = PageRenderer().render(image, [region(text="CRASH", kind=RegionKind.SFX, box=sfx_box)])
    assert report.results[0].rendered is True
    assert report.results[0].cleanup is None  # cleanup never runs for SFX

    # Pixels inside the SFX's own box are bit-identical to the source.
    inside_before = before.crop((sfx_box.x, sfx_box.y, sfx_box.x + sfx_box.width,
                                 sfx_box.y + sfx_box.height))
    inside_after = out.crop((sfx_box.x, sfx_box.y, sfx_box.x + sfx_box.width,
                             sfx_box.y + sfx_box.height))
    assert list(inside_before.getdata()) == list(inside_after.getdata())

    # Something was drawn in a strip below the box (the caption).
    below = out.crop((sfx_box.x, sfx_box.y + sfx_box.height,
                      sfx_box.x + sfx_box.width, sfx_box.y + sfx_box.height + 40))
    assert list(below.getdata()) != [(255, 255, 255)] * below.width * below.height


def test_render_flags_unsupported_target_language_instead_of_guessing():
    image = flat_page()
    out, report = PageRenderer().render(image, [region(target_language="ko")])
    assert len(report.flagged) == 1
    assert report.flagged[0].layout.unsupported is True
    # Cleanup was still safe to compute, but nothing was drawn or blanked.
    assert list(image.getdata()) == list(out.getdata())


# --- PageRenderer: explicit production_box -------------------------------
def test_ocr_geometry_is_untouched_by_an_explicit_production_box():
    # region.box (source/OCR/selection geometry) is passed through unchanged
    # regardless of production_box; nothing in the renderer ever mutates it.
    prod = BoundingBox(30, 30, 220, 120)
    r = region(box=WHITE_BOX, production_box=prod)
    assert r.box == WHITE_BOX
    assert r.production_box == prod


def test_a_wider_production_box_gives_more_room_than_the_tight_ocr_box():
    long_text = "This translation is much too long to fit in the tight OCR box alone."
    tight_box = BoundingBox(40, 40, 90, 60)
    wide_production = BoundingBox(20, 20, 300, 150)

    tight_image = stippled_page(tight_box, foreground_ratio=0.15)
    out_tight, report_tight = PageRenderer().render(
        tight_image, [region(text=long_text, box=tight_box, production_box=None)])
    # The tight box alone is too small: flagged, or shrunk to a small font.
    tight_result = report_tight.results[0]

    wide_image = stippled_page(wide_production, foreground_ratio=0.15)
    out_wide, report_wide = PageRenderer().render(
        wide_image, [region(text=long_text, box=tight_box, production_box=wide_production)])
    wide_result = report_wide.results[0]

    assert wide_result.rendered is True
    if tight_result.rendered:
        # Both fit: the wider area should never need a smaller font than the
        # tight one to fit the same text.
        assert wide_result.applied_render.font_size >= tight_result.applied_render.font_size
    # Either way, the wide production box is what actually got used.
    assert wide_result.cleanup.cleaned_box == wide_production


def test_every_pixel_outside_production_box_is_bit_identical_after_render():
    prod = BoundingBox(60, 60, 160, 80)
    image = stippled_page(prod, background=(250, 250, 250), ink=(10, 10, 10), foreground_ratio=0.15)
    before = image.copy()
    out, report = PageRenderer().render(image, [region(box=WHITE_BOX, production_box=prod)])
    assert report.results[0].rendered is True
    assert_unchanged_outside(before, out, prod)


def test_clearing_production_box_reproduces_the_legacy_auto_inset_result():
    # A segment that once had an explicit production_box, once cleared
    # (production_box=None), must render byte-for-byte as if it had never
    # had one — the legacy auto-inset path, not some leftover trace of it.
    prod = BoundingBox(30, 30, 220, 120)
    with_explicit_box = region(box=WHITE_BOX, production_box=prod)
    after_clearing = region(box=WHITE_BOX, production_box=None)
    pure_legacy = region(box=WHITE_BOX, production_box=None)

    PageRenderer().render(flat_page(), [with_explicit_box])  # exercised once; discarded
    out_cleared, report_cleared = PageRenderer().render(flat_page(), [after_clearing])
    out_legacy, report_legacy = PageRenderer().render(flat_page(), [pure_legacy])

    assert list(out_cleared.getdata()) == list(out_legacy.getdata())
    assert report_cleared.results[0].cleanup.cleaned_box == report_legacy.results[0].cleanup.cleaned_box


def test_sfx_ignores_an_explicit_production_box_even_if_one_is_set():
    sfx_box = BoundingBox(100, 100, 60, 40)
    prod = BoundingBox(80, 80, 120, 100)
    image = flat_page(box=sfx_box, box_fill=(30, 30, 30))
    out, report = PageRenderer().render(
        image, [region(text="CRASH", kind=RegionKind.SFX, box=sfx_box, production_box=prod)])
    assert report.results[0].rendered is True
    assert report.results[0].cleanup is None  # cleanup never runs for SFX, production_box or not
    # The original art inside the SFX box is preserved regardless of the
    # (ignored) production_box.
    inside_before = flat_page(box=sfx_box, box_fill=(30, 30, 30)).crop(
        (sfx_box.x, sfx_box.y, sfx_box.x + sfx_box.width, sfx_box.y + sfx_box.height))
    inside_after = out.crop((sfx_box.x, sfx_box.y, sfx_box.x + sfx_box.width,
                             sfx_box.y + sfx_box.height))
    assert list(inside_before.getdata()) == list(inside_after.getdata())


def test_render_uses_a_real_mask_when_the_region_provides_one():
    # Same fixture as the BubbleCleaner mask test: only succeeds if the mask
    # is actually threaded through to inspect().
    image = flat_page(box_fill=(255, 255, 255))
    ink_edge = WHITE_BOX.x + round(WHITE_BOX.width * 0.2)
    for x in range(WHITE_BOX.x, ink_edge):
        for y in range(WHITE_BOX.y, WHITE_BOX.y + WHITE_BOX.height):
            image.putpixel((x, y), (10, 10, 10))
    mask = Polygon([(WHITE_BOX.x, WHITE_BOX.y), (ink_edge, WHITE_BOX.y),
                    (ink_edge, WHITE_BOX.y + WHITE_BOX.height), (WHITE_BOX.x, WHITE_BOX.y + WHITE_BOX.height)])

    out, report = PageRenderer().render(image, [region(box=WHITE_BOX, production_box=None)])
    assert report.results[0].rendered is False  # no mask given: legacy inset sees ink, flags it

    out, report = PageRenderer().render(image, [
        RenderableRegion("seg-1", RegionKind.SPEECH_BUBBLE, WHITE_BOX, "Hi", RenderSettings(), "en", mask=mask)])
    assert report.results[0].rendered is True
    assert report.results[0].cleanup.cleaned_box == WHITE_BOX


def test_draw_lines_is_clipped_even_with_an_adversarial_oversized_layout():
    # A layout claiming a huge font size and a line far wider than the box —
    # if anything the layout math got wrong, this is the shape it'd take.
    # _draw_lines must still be unable to touch pixels outside the box.
    box = BoundingBox(60, 60, 40, 20)
    adversarial = LayoutResult(fits=True, lines=["MUCH TOO WIDE FOR THIS BOX"], font_size=500.0)
    image = Image.new("RGB", (400, 300), (255, 255, 255))
    before = image.copy()
    _draw_lines(image, box, adversarial, RenderSettings())
    assert_unchanged_outside(before, image, box)
