"""HTTP layer.

Routes do request handling and nothing else: they resolve the segment, call
``TranslationPipeline`` or ``ApprovalService``, and serialize. The translation
decision order lives in the pipeline and is not reproduced here.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..fixtures import DEMO_PROJECT_ID, build_services, seed_demo
from ..glossary.store import Category, GlossaryEntry
from ..memory.store import CanonicalConflict
from ..pipeline.approval import ApprovalError, GlossarySuggestion, TermResolution
from ..storage.models import Origin, Project, SegmentState, Status
from . import schemas
from .pages import build_page_router

STATIC = Path(__file__).resolve().parent.parent / "workbench" / "static"


class ProjectIn(BaseModel):
    name: str
    default_source_language: str = "ja"
    target_language: str = "en"
    id: str | None = None


class SegmentPatch(BaseModel):
    translation: str | None = None
    status: Status | None = None
    origin: Origin | None = None
    language: str | None = None


class ResolutionIn(BaseModel):
    source_term: str
    target_term: str


class ApprovalIn(BaseModel):
    translation: str | None = None
    term_resolutions: list[ResolutionIn] = Field(default_factory=list)


class DismissIn(BaseModel):
    fingerprint: str
    code: str
    resolution: str = "dismissed"


class GlossaryIn(BaseModel):
    source_language: str
    source_term: str
    target_language: str
    target_term: str
    category: Category = Category.OTHER
    locked: bool = False
    notes: str | None = None


class GlossaryPatch(BaseModel):
    target_term: str | None = None
    category: Category | None = None
    locked: bool | None = None
    notes: str | None = None


def create_app(services=None, seed: bool | None = None) -> FastAPI:
    """Build the app.

    A fresh install opens on an empty projects screen. The demo project is
    fixture data for tests and screenshots, so it is only created when asked
    for: ``SCANLATE_DEMO=1`` or ``seed=True``.
    """
    services = services or build_services(os.environ.get("SCANLATE_DB", ":memory:"))
    if seed is None:
        seed = os.environ.get("SCANLATE_DEMO", "").lower() in {"1", "true", "yes"}
    if seed and not services.repository.get_project(DEMO_PROJECT_ID):
        seed_demo(services)

    app = FastAPI(title="Scanlate", version="0.1.0")
    app.state.services = services
    app.include_router(build_page_router(services))

    def segment_or_404(segment_id: str):
        segment = services.repository.get_segment(segment_id)
        if segment is None:
            raise HTTPException(404, f"No segment {segment_id}")
        return segment

    def project_or_404(project_id: str):
        project = services.repository.get_project(project_id)
        if project is None:
            raise HTTPException(404, f"No project {project_id}")
        return project

    # --- projects ------------------------------------------------------
    @app.get("/api/projects")
    def list_projects():
        projects = []
        for project in services.repository.list_projects():
            chapters = services.repository.list_chapters(project.id)
            payload = schemas.project(project, chapters,
                                      services.repository.list_pages(project.id))
            payload["segment_count"] = len(services.repository.list_segments(project.id))
            projects.append(payload)
        return projects

    @app.post("/api/projects")
    def create_project(body: ProjectIn):
        name = body.name.strip()
        if not name:
            raise HTTPException(400, "A project needs a name.")
        project_id = body.id or _slug(name)
        if services.repository.get_project(project_id):
            raise HTTPException(409, f"A project called {name!r} already exists.")
        project = Project(project_id, name, body.default_source_language,
                          body.target_language)
        with services.uow.transaction():
            services.repository.create_project(project)
        return schemas.project(project)

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str):
        project = project_or_404(project_id)
        return schemas.project(project, services.repository.list_chapters(project_id),
                               services.repository.list_pages(project_id))

    @app.get("/api/projects/{project_id}/segments")
    def list_segments(project_id: str, page_id: str | None = None):
        project = project_or_404(project_id)
        segments = services.repository.list_segments(project_id)
        if page_id:
            segments = [s for s in segments if s.page_id == page_id]
        views = [services.pipeline.process(s, project).view for s in segments]
        return {
            "project": schemas.project(project, services.repository.list_chapters(project_id),
                                       services.repository.list_pages(project_id)),
            "segments": [schemas.segment_view(v) for v in views],
        }

    @app.post("/api/projects/{project_id}/translate")
    def translate_project(project_id: str, page_id: str | None = None):
        """Re-run the pipeline. User-owned text is left alone."""
        project = project_or_404(project_id)
        segments = services.repository.list_segments(project_id)
        if page_id:
            segments = [s for s in segments if s.page_id == page_id]
        views = [services.pipeline.process(s, project).view for s in segments]
        return {
            "translated": sum(1 for v in views if v.status is Status.MACHINE),
            "needs_review": sum(1 for v in views if v.needs_review),
            "segments": [schemas.segment_view(v) for v in views],
        }

    # --- segments ------------------------------------------------------
    @app.get("/api/segments/{segment_id}")
    def get_segment(segment_id: str):
        return schemas.segment_view(services.pipeline.process(segment_or_404(segment_id)).view)

    @app.patch("/api/segments/{segment_id}")
    def patch_segment(segment_id: str, patch: SegmentPatch):
        segment = segment_or_404(segment_id)
        with services.uow.transaction():
            if patch.language:
                services.repository.set_language(segment_id, patch.language)
            if patch.translation is not None or patch.status is not None:
                text = patch.translation if patch.translation is not None else segment.state.candidate
                edited = patch.translation is not None and patch.translation != segment.state.candidate
                status = patch.status or (Status.EDITED if edited else segment.state.status)
                origin = patch.origin or (Origin.MANUAL if edited else segment.state.origin)
                services.repository.save_state(segment_id, SegmentState(text, status, origin))
        return schemas.segment_view(
            services.pipeline.process(segment_or_404(segment_id)).view)

    @app.post("/api/segments/{segment_id}/approve")
    def approve_segment(segment_id: str, body: ApprovalIn = Body(default=ApprovalIn())):
        segment_or_404(segment_id)
        try:
            result = services.approvals.approve(
                segment_id, body.translation,
                [TermResolution(r.source_term, r.target_term) for r in body.term_resolutions])
        except ApprovalError as error:
            raise HTTPException(400, str(error))
        except CanonicalConflict as error:
            raise HTTPException(409, str(error))
        return {
            "segment": schemas.segment_view(services.pipeline.process(result.segment).view),
            "memory_entry_id": result.memory_entry_id,
            "suggestion": schemas.suggestion(result.suggestion),
        }

    @app.post("/api/segments/{segment_id}/reopen")
    def reopen_segment(segment_id: str):
        segment_or_404(segment_id)
        segment = services.approvals.reopen(segment_id)
        return schemas.segment_view(services.pipeline.process(segment).view)

    @app.post("/api/segments/{segment_id}/issues/resolve")
    def resolve_issue(segment_id: str, body: DismissIn):
        segment_or_404(segment_id)
        with services.uow.transaction():
            services.repository.resolve_issue(segment_id, body.fingerprint, body.code,
                                              body.resolution)
        return schemas.segment_view(
            services.pipeline.process(segment_or_404(segment_id)).view)

    # --- glossary ------------------------------------------------------
    @app.get("/api/projects/{project_id}/glossary")
    def list_glossary(project_id: str, source_language: str | None = None):
        project_or_404(project_id)
        return [schemas.glossary_entry(e)
                for e in services.glossary.list(project_id, source_language)]

    @app.post("/api/projects/{project_id}/glossary")
    def add_glossary(project_id: str, body: GlossaryIn):
        project_or_404(project_id)
        with services.uow.transaction():
            entry = services.glossary.add(GlossaryEntry(
                None, project_id, body.source_language, body.source_term,
                body.target_language, body.target_term, body.category,
                locked=body.locked, notes=body.notes))
        return schemas.glossary_entry(entry)

    @app.post("/api/projects/{project_id}/glossary/accept")
    def accept_suggestion(project_id: str, body: GlossaryIn):
        """Create the rule approval suggested, once the user has consented."""
        project_or_404(project_id)
        suggestion = GlossarySuggestion(
            source_term=body.source_term, target_term=body.target_term,
            category=body.category, reason="accepted by user",
            source_language=body.source_language, target_language=body.target_language,
            explicit=True)
        entry = services.approvals.accept_suggestion(project_id, suggestion, body.locked)
        return schemas.glossary_entry(entry)

    @app.patch("/api/glossary/{entry_id}")
    def patch_glossary(entry_id: int, patch: GlossaryPatch):
        fields = {k: v for k, v in patch.model_dump().items() if v is not None}
        with services.uow.transaction():
            entry = services.glossary.update(entry_id, **fields)
        if entry is None:
            raise HTTPException(404, f"No glossary entry {entry_id}")
        return schemas.glossary_entry(entry)

    # --- memory --------------------------------------------------------
    @app.get("/api/projects/{project_id}/memory")
    def list_memory(project_id: str, limit: int = 200):
        project_or_404(project_id)
        return [schemas.memory_entry(e) for e in services.memory.list(project_id, limit)]

    @app.post("/api/memory/{entry_id}/canonical")
    def set_canonical(entry_id: int, locked: bool = True):
        try:
            entry = services.memory.set_locked(entry_id, locked)
        except CanonicalConflict as error:
            raise HTTPException(409, str(error))
        if entry is None:
            raise HTTPException(404, f"No memory entry {entry_id}")
        return schemas.memory_entry(entry)

    # --- workbench -----------------------------------------------------
    @app.get("/")
    def workbench():
        """The workbench and the API share one origin.

        The page is served from this same FastAPI process, so its relative
        ``/api/...`` requests resolve to this server with no CORS configuration
        and no separate dev server. Nothing here is cross-origin by design.
        """
        return FileResponse(STATIC / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        """No icon yet; answer so it isn't a 404 in every browser session."""
        return Response(status_code=204)

    @app.get("/api/health")
    def health():
        ocr = services.page_service.ocr
        translation = services.backends.describe()
        return {
            "status": "ok",
            "languages": services.adapters.languages(),
            "translation": {
                "names": translation["names"],
                "by_pair": translation["by_pair"],
                # True when nothing but the deterministic test backend is
                # loaded, so the UI can say so instead of presenting glosses
                # as machine translation.
                "development": translation["development"],
            },
            "detector": {"name": services.page_service.detector.name,
                         "available": services.page_service.detector.available()},
            "ocr": {"name": ocr.name, "available": ocr.available(),
                    "languages": ocr.languages(),
                    "routing": ocr.describe() if hasattr(ocr, "describe") else None},
        }

    return app


# Deliberately a factory, not an instance: building the app at import time
# would open a database and seed the demo project as a side effect of importing
# this module. uvicorn auto-detects it, but prefer the explicit form:
#     uvicorn scanlate.api.app:create_app --factory
app = create_app


def _slug(name: str) -> str:
    import re
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"
    return base[:48]
