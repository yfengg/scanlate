"""Rendering core: bubble cleanup, deterministic typesetting, SFX captions.

Pure-image unit tests, no database or API — the service/API/frontend wiring
that turns this into "Preview"/"Export" is a later step. Fixture images are
generated here, not committed, same convention as test_page_workflow.py.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

from scanlate.imaging.cleanup import BubbleCleaner
from scanlate.imaging.layout import BoundingBox, RegionKind, RenderSettings
from scanlate.imaging.render import PageRenderer, RenderableRegion
from scanlate.imaging.typeset import (DEFAULT_FONT_ID, FONT_FILES, FONTS_DIR,
                                     LatinHorizontalTypesetter, TextLayoutEngine, load_font)

WHITE_BOX = BoundingBox(40, 40, 200, 100)


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
          render=None, target_language="en") -> RenderableRegion:
    return RenderableRegion("seg-1", kind, box, text, render or RenderSettings(), target_language)


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
