"""Page workflow service.

Detection, OCR and region editing all end in the same place: ordinary
``Segment`` records with a ``region``. There is no parallel "OCR object" — a
segment created here is indistinguishable from a demo segment downstream and
runs through ``TranslationPipeline`` unchanged.

Two rules worth stating plainly, because they decide what happens to work the
user has already done:

**Changing source text.** New source means the existing translation was for a
different sentence. An unapproved candidate is discarded and regenerated. An
*approved* one is not silently kept: the segment is reopened, its translation
memory occurrence released, and it comes back needing review. The previously
approved text stays on record as a memory variant — it was approved once — but
it is no longer this segment's current answer.

**Deleting a region.** The region and its segment are one thing, so deleting
the region deletes the segment — unless that segment is approved, which is
finished work. Deleting an approved segment's region requires ``force=True``,
and then the segment is kept and detached from the page rather than destroyed;
it remains in Review with its translation and memory intact.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from ..core.types import SegmentKind
from ..languages import AdapterRegistry
from ..memory.store import TranslationMemory
from ..storage.models import Origin, Segment, SegmentState, Status
from ..storage.repository import ProjectRepository
from ..storage.unit_of_work import UnitOfWork
from .detection import DetectorError, DetectorUnavailable, TextRegionDetector
from .importer import ImageError, MediaStore
from .layout import BoundingBox, Region, RegionKind, RenderSettings, TextOrientation
from .ocr import OcrBackend, OcrError, OcrResult
from .render import PageRenderer, RenderableRegion, RenderReport

REGION_KIND_FOR_SEGMENT = {
    SegmentKind.DIALOGUE: RegionKind.SPEECH_BUBBLE,
    SegmentKind.THOUGHT: RegionKind.THOUGHT_BUBBLE,
    SegmentKind.NARRATION: RegionKind.CAPTION,
    SegmentKind.SFX: RegionKind.SFX,
    SegmentKind.SIGN: RegionKind.SIGN,
    SegmentKind.OTHER: RegionKind.FREE_TEXT,
}


@dataclass
class RegionOcr:
    """What OCR did to one region.

    ``low_confidence`` is about the recognition; ``reopened`` is about what
    writing the text did to an existing approval. They are unrelated — a
    perfectly confident OCR pass can reopen an approved segment, and a shaky
    one on a fresh region reopens nothing — so they are reported separately
    rather than folded into one flag.
    """
    segment_id: str
    text: str
    language: str | None = None
    confidence: float | None = None
    #: False when the engine reports no score at all. Distinct from a low one.
    confidence_available: bool = True
    low_confidence: bool = False
    reopened: bool = False
    orientation: str | None = None
    backend: str | None = None
    fell_back_from: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def needs_review(self) -> bool:
        return self.low_confidence or self.reopened or not self.ok


@dataclass
class DetectionResult:
    """Detection outcome, separating "couldn't run" from "found nothing"."""
    regions: list[Segment]
    available: bool = True
    message: str | None = None

    @property
    def empty(self) -> bool:
        return not self.regions


@dataclass
class SourceTextChange:
    segment_id: str
    reopened: bool = False       # an approved translation lost its authority
    message: str | None = None
    changed: str | None = None   # "source text", "source language", or both


class RegionError(Exception):
    """Invalid geometry, or a deletion that needs confirmation."""


class PageService:
    def __init__(self, repository: ProjectRepository, media: MediaStore,
                 detector: TextRegionDetector, ocr: OcrBackend,
                 adapters: AdapterRegistry, memory: TranslationMemory,
                 uow: UnitOfWork | None = None, renderer: PageRenderer | None = None):
        self.repository = repository
        self.media = media
        self.detector = detector
        self.ocr = ocr
        self.adapters = adapters
        self.memory = memory
        self.uow = uow or repository.uow
        self.renderer = renderer or PageRenderer()

    # --- regions -------------------------------------------------------
    def create_region(self, page_id: str, box: BoundingBox, kind: SegmentKind = SegmentKind.OTHER,
                      language: str | None = None, source_text: str = "",
                      segment_id: str | None = None) -> Segment:
        """Draw a region by hand. Always available, detector or not."""
        page = self._page(page_id)
        self._validate(box, page.width, page.height)
        segment = Segment(
            id=segment_id or f"seg-{uuid.uuid4().hex[:10]}",
            project_id=page.project_id, seq=self.repository.next_seq(page.project_id),
            kind=SegmentKind(kind), source_text=source_text, language=language,
            page_id=page_id,
            region=self._region(box, kind, language or self._default_language(page)))
        with self.uow.transaction():
            self.repository.add_segment(segment)
        return self.repository.get_segment(segment.id)

    def update_region(self, segment_id: str, box: BoundingBox | None = None,
                      kind: SegmentKind | None = None, language: str | None = None,
                      orientation: TextOrientation | None = None) -> Segment:
        """Move, resize, retype, relabel or reorient a region."""
        segment = self._segment(segment_id)
        with self.uow.transaction():
            if box is not None:
                page = self._page(segment.page_id) if segment.page_id else None
                self._validate(box, page.width if page else None,
                               page.height if page else None)
                region = self._region(box, kind or segment.kind,
                                      segment.language or self._default_language(
                                          self._page(segment.page_id) if segment.page_id else None))
                if orientation is not None:
                    region.orientation = orientation
                elif segment.region is not None:
                    region.orientation = segment.region.orientation
                self.repository.set_region(segment_id, region)
            if kind is not None:
                self.repository.set_kind(segment_id, SegmentKind(kind))
                if box is None and segment.region is not None:
                    # Changing the type must not silently reset the orientation
                    # the user chose.
                    region = self._region(segment.region.box, kind, segment.language)
                    region.orientation = segment.region.orientation
                    self.repository.set_region(segment_id, region)
            if language is not None and language != segment.language:
                # Not a plain field update: changing the language changes which
                # adapter reads the segment, so it invalidates the translation.
                self.set_source_text(segment_id, segment.source_text, language)
            if orientation is not None and box is None:
                current = self.repository.get_segment(segment_id)
                if current.region is not None:
                    region = self._region(current.region.box, current.kind)
                    region.orientation = orientation
                    self.repository.set_region(segment_id, region)
        return self.repository.get_segment(segment_id)

    def delete_region(self, segment_id: str, force: bool = False) -> str:
        """Remove a region. See the module docstring for the approved-segment rule."""
        segment = self._segment(segment_id)
        with self.uow.transaction():
            if segment.state.status is Status.APPROVED:
                if not force:
                    raise RegionError(
                        "This region's translation is approved. Deleting it would discard "
                        "finished work — confirm to detach it from the page instead.")
                self.memory.clear_occurrence(segment_id)
                self.repository.detach_from_page(segment_id)
                return "detached"
            self.repository.delete_segment(segment_id)
        return "deleted"

    # --- detection -----------------------------------------------------
    def detect_regions(self, page_id: str, replace: bool = False) -> DetectionResult:
        """Detect text and turn each box into a segment.

        Finding nothing is a valid outcome, not an error. Not being able to run
        at all is reported separately; either way the user draws regions by hand.
        """
        page = self._page(page_id)
        image = self.media.read(page.image_path)

        try:
            detected = self.detector.detect(image)
        except DetectorUnavailable as error:
            return DetectionResult([], available=False, message=str(error))
        except DetectorError:
            raise
        except Exception as error:
            raise DetectorError(f"Detection failed: {error}")

        created: list[Segment] = []
        with self.uow.transaction():
            if replace:
                for segment in self.repository.list_page_segments(page_id):
                    if segment.state.status is not Status.APPROVED and not segment.source_text:
                        self.repository.delete_segment(segment.id)

            # Boxes already on the page, plus every box accepted so far in this
            # run: two detections of the same bubble must not both be kept just
            # because neither existed when the run started.
            taken = [s.region.box for s in self.repository.list_page_segments(page_id)
                     if s.region and s.region.box]
            for region in detected:
                if any(_overlaps(region.box, box) for box in taken):
                    continue
                taken.append(region.box)
                created.append(self.create_region(page_id, region.box,
                                                  kind=region.suggested_kind,
                                                  language=None))
        message = None if created else "No text regions were found. Draw them by hand."
        return DetectionResult(created, available=True, message=message)

    # --- OCR -----------------------------------------------------------
    def ocr_segment(self, segment_id: str, language: str | None = None) -> RegionOcr:
        """Run OCR on one region and write the result as the segment's source."""
        segment = self._segment(segment_id)
        if segment.page_id is None or segment.region is None or segment.region.box is None:
            return RegionOcr(segment_id, segment.source_text,
                             error="This segment has no region on a page.")
        page = self._page(segment.page_id)
        hint = language or segment.language
        # The region carries the orientation the user last saved; only a region
        # that has never been set falls back to the engine's shape guess.
        orientation = segment.region.orientation if segment.region.orientation else None
        try:
            image = self.media.read(page.image_path)
            result = self.ocr.recognize(image, segment.region.box, hint, orientation)
        except (OcrError, ImageError) as error:
            # Never fatal: the region stays, the user types the text.
            return RegionOcr(segment_id, segment.source_text, error=str(error))

        detected = result.language or self.adapters.resolve(result.text, hint)
        change = self.set_source_text(segment_id, result.text,
                                      language=detected if detected != "und" else None)
        return RegionOcr(
            segment_id=segment_id, text=result.text, language=detected,
            confidence=result.confidence,
            confidence_available=result.metadata.get("confidence_available",
                                                     result.confidence_available),
            low_confidence=result.low_confidence, reopened=change.reopened,
            # The engine may not know which way the text ran; report the
            # region's own setting in that case rather than guessing.
            orientation=(result.orientation.value if result.orientation
                         else (orientation.value if orientation else None)),
            backend=result.metadata.get("backend"),
            fell_back_from=result.metadata.get("fell_back_from"))

    def ocr_page(self, page_id: str, language: str | None = None,
                 only_empty: bool = True) -> list[RegionOcr]:
        """OCR every region on the page. One failure doesn't stop the rest."""
        results = []
        for segment in self.repository.list_page_segments(page_id):
            if only_empty and segment.source_text.strip():
                continue
            if segment.state.status is Status.APPROVED:
                continue                        # don't disturb finished work in bulk
            try:
                results.append(self.ocr_segment(segment.id, language))
            except Exception as error:
                results.append(RegionOcr(segment.id, segment.source_text, error=str(error)))
        return results

    # --- source text ---------------------------------------------------
    def set_source_text(self, segment_id: str, source_text: str,
                        language: str | None = None) -> SourceTextChange:
        """Correct the OCR text, the source language, or both.

        A language change invalidates a translation exactly as a text change
        does: it routes the segment through a different adapter, so the register
        reading, the terminology and the memory key all change. The same
        approval rule applies — see the module docstring.
        """
        segment = self._segment(segment_id)
        text_changed = source_text != segment.source_text
        language_changed = language is not None and language != segment.language
        if not text_changed and not language_changed:
            return SourceTextChange(segment_id)

        reopened = False
        with self.uow.transaction():
            if text_changed:
                self.repository.set_source_text(segment_id, source_text)
            if language_changed:
                self.repository.set_language(segment_id, language)
            if segment.state.status is Status.APPROVED:
                # The approval was for a different sentence, or for the same
                # words read as a different language. Either way it stops being
                # this segment's answer and has to be looked at again.
                self.memory.clear_occurrence(segment_id)
                reopened = True
            # The old candidate was produced from the old source, so it goes
            # and the pipeline regenerates.
            self.repository.save_state(segment_id, SegmentState(
                None, Status.MACHINE, Origin.MACHINE_TRANSLATION))

        what = "source text" if text_changed else "source language"
        if text_changed and language_changed:
            what = "source text and language"
        message = (f"The {what} changed, so the approved translation no longer applies to "
                   f"this segment. It's back in review.") if reopened else None
        return SourceTextChange(segment_id, reopened, message, what)

    # --- rendering (preview/export) -------------------------------------
    def render_page(self, page_id: str) -> tuple[Image.Image, RenderReport]:
        """Composite every approved region's cleaned bubble/typeset translation
        (or SFX caption) onto a copy of the source page.

        Never writes to disk — this is what both Preview and Export call, but
        only Export goes on to persist the result. Regions belonging to a
        segment that isn't approved yet are left as original artwork; they
        are simply not in the list handed to the renderer.
        """
        page = self._page(page_id)
        project = self.repository.get_project(page.project_id)
        target_language = project.target_language if project else "en"
        image = Image.open(BytesIO(self.media.read(page.image_path)))
        regions = self._renderable_regions(page_id, target_language)
        rendered, report = self.renderer.render(image, regions)
        self._persist_applied_render(report)
        return rendered, report

    def export_page(self, page_id: str) -> tuple[str, RenderReport]:
        """Render and write a PNG at the source's exact dimensions.

        The filename comes from chapter/page identity, not a content hash —
        predictable and human-readable, and stable across re-exports of the
        same page after an edit (each overwrites only its own previous
        export). The imported source file is never touched: exports live
        under their own ``exports/`` path, never the source's.
        """
        page = self._page(page_id)
        rendered, report = self.render_page(page_id)
        buffer = BytesIO()
        rendered.save(buffer, format="PNG")
        reference = self.media.store_export(page.project_id, self._export_filename(page),
                                            buffer.getvalue())
        return reference, report

    def _renderable_regions(self, page_id: str, target_language: str) -> list[RenderableRegion]:
        regions = []
        for segment in self.repository.list_page_segments(page_id):
            if segment.state.status is not Status.APPROVED:
                continue                        # left as original artwork
            if segment.region is None or segment.region.box is None:
                continue
            regions.append(RenderableRegion(
                segment.id, segment.region.kind, segment.region.box,
                segment.state.candidate or "", segment.render or RenderSettings(),
                target_language))
        return regions

    def _persist_applied_render(self, report: RenderReport) -> None:
        """Save the font size that actually fit, so the next render starts
        there instead of re-searching from scratch. No mask is recorded —
        cleanup is cheap to recompute from the source image every time."""
        with self.uow.transaction():
            for result in report.results:
                if result.applied_render is not None:
                    self.repository.set_render(result.segment_id, result.applied_render)

    def _export_filename(self, page) -> str:
        chapter = self.repository.get_chapter(page.chapter_id) if page.chapter_id else None
        chapter_number = chapter.number if chapter else "x"
        return f"chapter-{chapter_number}-page-{page.number}-translated.png"

    # --- helpers -------------------------------------------------------
    def _region(self, box: BoundingBox, kind: SegmentKind,
                language: str | None = None) -> Region:
        return Region(kind=REGION_KIND_FOR_SEGMENT.get(SegmentKind(kind), RegionKind.FREE_TEXT),
                      box=box, reading_order=0,
                      orientation=self._initial_orientation(box, language))

    def _initial_orientation(self, box: BoundingBox, language: str | None) -> TextOrientation:
        """Guess which way the text runs, language first.

        Shape alone is a poor signal: a typical manga bubble is roughly square,
        so an aspect-ratio rule called most vertical Japanese horizontal. The
        adapter knows whether its language sets text vertically at all, which
        is the stronger prior; shape only overrides it for a box too wide to
        hold a vertical line.
        """
        adapter = self.adapters.get(language) if language else None
        profile = adapter.ocr_profile("") if adapter else None
        clearly_wide = box.width > box.height * 1.5
        if profile is not None and profile.vertical_text_likely and not clearly_wide:
            return TextOrientation.VERTICAL_RL
        if box.height > box.width * 1.6:
            return TextOrientation.VERTICAL_RL
        return TextOrientation.HORIZONTAL

    def _validate(self, box: BoundingBox, width: int | None, height: int | None) -> None:
        """Coordinates are in image space; the viewport never reaches here."""
        if box.width < 1 or box.height < 1:
            raise RegionError("A region needs a positive width and height.")
        if box.x < 0 or box.y < 0:
            raise RegionError("A region can't start outside the page.")
        if width and height and (box.x + box.width > width or box.y + box.height > height):
            raise RegionError(f"That region falls outside the page ({width}×{height}).")

    def _default_language(self, page) -> str | None:
        if page is None:
            return None
        project = self.repository.get_project(page.project_id)
        return project.default_source_language if project else None

    def _page(self, page_id: str):
        page = self.repository.get_page(page_id)
        if page is None:
            raise ImageError(f"No page {page_id}")
        return page

    def _segment(self, segment_id: str) -> Segment:
        segment = self.repository.get_segment(segment_id)
        if segment is None:
            raise RegionError(f"No segment {segment_id}")
        return segment


def _overlaps(a: BoundingBox, b: BoundingBox, threshold: float = 0.5) -> bool:
    overlap_x = max(0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    overlap_y = max(0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    intersection = overlap_x * overlap_y
    smaller = min(a.width * a.height, b.width * b.height)
    return smaller > 0 and intersection / smaller > threshold
