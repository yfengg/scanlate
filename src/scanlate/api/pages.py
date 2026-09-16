"""Page, region and OCR routes.

Thin by construction: each route resolves its arguments, calls ``PageService``
or the existing pipeline, and serializes. No detection, OCR or translation
logic lives here — the imaging services own it, and these routes would be the
wrong place to duplicate any of it.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..core.types import SegmentKind
from ..imaging.detection import DetectorError
from ..imaging.importer import ImageError, PageConflict
from ..imaging.layout import BoundingBox, TextOrientation
from ..imaging.service import RegionError
from ..storage.models import Chapter
from . import schemas


class RegionIn(BaseModel):
    x: int
    y: int
    width: int
    height: int
    kind: SegmentKind = SegmentKind.OTHER
    language: str | None = None
    source_text: str = ""


class RegionPatch(BaseModel):
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    kind: SegmentKind | None = None
    language: str | None = None
    orientation: TextOrientation | None = None


class SourceIn(BaseModel):
    source_text: str
    language: str | None = None


class ChapterIn(BaseModel):
    number: int
    title: str | None = None


def build_page_router(services) -> APIRouter:
    router = APIRouter()
    pages = services.page_service

    def page_or_404(page_id: str):
        page = services.repository.get_page(page_id)
        if page is None:
            raise HTTPException(404, f"No page {page_id}")
        return page

    def segment_or_404(segment_id: str):
        segment = services.repository.get_segment(segment_id)
        if segment is None:
            raise HTTPException(404, f"No segment {segment_id}")
        return segment

    def view(segment_id: str) -> dict:
        """Every mutation answers with the re-run SegmentView, so a region edit
        and a translation edit look the same to the client."""
        outcome = services.pipeline.process(segment_or_404(segment_id))
        return schemas.segment_view(outcome.view)

    # --- chapters and pages -------------------------------------------
    @router.post("/api/projects/{project_id}/chapters")
    def create_chapter(project_id: str, body: ChapterIn):
        if services.repository.get_project(project_id) is None:
            raise HTTPException(404, f"No project {project_id}")
        chapter = Chapter(f"{project_id}-ch{body.number}", project_id, body.number, body.title)
        try:
            with services.uow.transaction():
                services.repository.create_chapter(chapter)
        except Exception as error:
            raise HTTPException(409, f"Chapter {body.number} already exists: {error}")
        return {"id": chapter.id, "number": chapter.number, "title": chapter.title}

    @router.get("/api/projects/{project_id}/pages")
    def list_pages(project_id: str):
        if services.repository.get_project(project_id) is None:
            raise HTTPException(404, f"No project {project_id}")
        return [schemas.page(p) for p in services.repository.list_pages(project_id)]

    @router.post("/api/projects/{project_id}/pages")
    async def import_page(project_id: str, file: UploadFile = File(...),
                          chapter_id: str = Form(...), number: int = Form(...),
                          replace: bool = Form(False)):
        if services.repository.get_project(project_id) is None:
            raise HTTPException(404, f"No project {project_id}")
        data = await file.read()
        try:
            page = services.pages.import_page(project_id, chapter_id, number, data,
                                              replace=replace)
        except PageConflict as error:
            raise HTTPException(409, str(error))
        except ImageError as error:
            raise HTTPException(400, str(error))
        return schemas.page(page, services.repository.list_page_segments(page.id))

    @router.get("/api/pages/{page_id}")
    def get_page(page_id: str):
        page = page_or_404(page_id)
        return schemas.page(page, services.repository.list_page_segments(page_id))

    @router.get("/api/pages/{page_id}/image")
    def get_page_image(page_id: str):
        page = page_or_404(page_id)
        try:
            data = services.media.read(page.image_path)
        except ImageError as error:
            raise HTTPException(404, str(error))
        suffix = (page.image_path or "").rsplit(".", 1)[-1].lower()
        media_type = {"png": "image/png", "jpg": "image/jpeg",
                      "jpeg": "image/jpeg", "webp": "image/webp"}.get(suffix, "image/png")
        return Response(content=data, media_type=media_type,
                        headers={"Cache-Control": "private, max-age=60"})

    @router.get("/api/pages/{page_id}/segments")
    def page_segments(page_id: str):
        page = page_or_404(page_id)
        segments = services.repository.list_page_segments(page_id)
        return {"page": schemas.page(page, segments),
                "segments": [schemas.segment_view(services.pipeline.process(s).view)
                             for s in segments]}

    # --- detection and OCR --------------------------------------------
    @router.post("/api/pages/{page_id}/detect")
    def detect(page_id: str, replace: bool = False):
        page_or_404(page_id)
        try:
            result = pages.detect_regions(page_id, replace=replace)
        except DetectorError as error:
            # Detection failing never blocks manual regions.
            return {"available": False, "created": 0, "message": str(error), "segments": []}
        except ImageError as error:
            raise HTTPException(404, str(error))
        return {"available": result.available, "created": len(result.regions),
                "message": result.message,
                "segments": [view(s.id) for s in result.regions]}

    @router.post("/api/pages/{page_id}/ocr")
    def ocr_page(page_id: str, language: str | None = None, only_empty: bool = True):
        page_or_404(page_id)
        results = pages.ocr_page(page_id, language, only_empty)
        return {"results": [schemas.ocr_result(r) for r in results],
                "segments": [view(r.segment_id) for r in results if r.ok]}

    @router.post("/api/segments/{segment_id}/ocr")
    def ocr_segment(segment_id: str, language: str | None = None):
        segment_or_404(segment_id)
        result = pages.ocr_segment(segment_id, language)
        return {"result": schemas.ocr_result(result), "segment": view(segment_id)}

    # --- regions -------------------------------------------------------
    @router.post("/api/pages/{page_id}/regions")
    def create_region(page_id: str, body: RegionIn):
        page_or_404(page_id)
        try:
            segment = pages.create_region(
                page_id, BoundingBox(body.x, body.y, body.width, body.height),
                body.kind, body.language, body.source_text,
                segment_id=f"seg-{uuid.uuid4().hex[:10]}")
        except RegionError as error:
            raise HTTPException(400, str(error))
        return view(segment.id)

    @router.patch("/api/segments/{segment_id}/region")
    def update_region(segment_id: str, body: RegionPatch):
        segment = segment_or_404(segment_id)
        box = None
        if None not in (body.x, body.y, body.width, body.height):
            box = BoundingBox(body.x, body.y, body.width, body.height)
        elif any(v is not None for v in (body.x, body.y, body.width, body.height)):
            raise HTTPException(400, "A region move or resize needs x, y, width and height.")
        try:
            pages.update_region(segment_id, box, body.kind, body.language, body.orientation)
        except RegionError as error:
            raise HTTPException(400, str(error))
        return view(segment_id)

    @router.delete("/api/segments/{segment_id}/region")
    def delete_region(segment_id: str, force: bool = False):
        segment_or_404(segment_id)
        try:
            outcome = pages.delete_region(segment_id, force=force)
        except RegionError as error:
            # An approved segment needs confirmation, not a silent discard.
            raise HTTPException(409, str(error))
        return {"segment_id": segment_id, "outcome": outcome}

    @router.patch("/api/segments/{segment_id}/source")
    def set_source(segment_id: str, body: SourceIn):
        segment_or_404(segment_id)
        try:
            change = pages.set_source_text(segment_id, body.source_text, body.language)
        except RegionError as error:
            raise HTTPException(400, str(error))
        return {"segment": view(segment_id), "reopened": change.reopened,
                "changed": change.changed, "message": change.message}

    return router
