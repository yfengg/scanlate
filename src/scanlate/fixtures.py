"""The demo project and the standard service wiring.

One place that assembles repository, glossary, memory, adapters, backends and
validators into a working pipeline, used by both the tests and the running
server so they cannot drift apart.

The demo project is mixed-language on purpose: Japanese and Korean segments in
one project, which is what a real page with a foreign-language sign looks like.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .core.types import SegmentKind
from .glossary.store import Category, GlossaryEntry, GlossaryStore
from .languages import AdapterRegistry, default_registry
from .memory.store import TranslationMemory
from .imaging.detection import build_detector
from .imaging.importer import MediaStore, PageImporter
from .imaging.ocr import build_ocr
from .imaging.service import PageService
from .pipeline.approval import ApprovalService
from .pipeline.pipeline import TranslationPipeline
from .sfx.renderer import EnglishSfxRenderer, SfxRendererRegistry
from .storage.db import connect
from .storage.models import Chapter, Page, Project, Segment
from .storage.repository import ProjectRepository
from .storage.unit_of_work import UnitOfWork
from .translation.backends.lexicon import LexiconBackend
from .translation.backends.opus_mt import OpusMtBackend
from .translation.registry import BackendRegistry

DEMO_PROJECT_ID = "demo"
BACKEND_ENV = "SCANLATE_BACKEND"


def build_backends(name: str | None = None) -> BackendRegistry:
    """Choose the machine-translation backend at runtime.

    ``auto`` (the default for a running app) uses the real OPUS-MT model when
    it is installed and falls back to the deterministic phrase table otherwise,
    flagging that fallback as a development backend so the UI can say so rather
    than passing word glosses off as translation.

    ``opus`` demands the real model and fails loudly if it is missing. Tests
    pass ``lexicon`` explicitly, so a failure there means the surrounding logic
    broke rather than the model drifting.
    """
    name = (name or os.environ.get(BACKEND_ENV) or "auto").lower()
    registry = BackendRegistry()

    if name in {"opus", "opus-mt", "opus_mt", "auto"}:
        opus = OpusMtBackend()
        if opus.available():
            registry.register(opus)
        elif name != "auto":
            raise RuntimeError(
                f"SCANLATE_BACKEND=opus can't run: {opus.unavailable_reason()}. "
                f"Falling back silently would hide which model produced a translation.")
    elif name != "lexicon":
        raise ValueError(f"Unknown backend {name!r}; expected 'auto', 'opus' or 'lexicon'.")

    registry.register(LexiconBackend())
    registry.set_default(LexiconBackend())
    return registry


@dataclass
class Services:
    conn: object
    uow: UnitOfWork
    repository: ProjectRepository
    glossary: GlossaryStore
    memory: TranslationMemory
    adapters: AdapterRegistry
    backends: BackendRegistry
    pipeline: TranslationPipeline
    approvals: ApprovalService
    media: MediaStore
    pages: PageImporter
    page_service: PageService


def build_services(path: str | Path = ":memory:", backend=None,
                   backend_name: str | None = None, media_root: str | Path | None = None,
                   detector=None, ocr=None) -> Services:
    conn = connect(path)
    uow = UnitOfWork(conn)
    repository = ProjectRepository(conn, uow)
    glossary = GlossaryStore(conn, uow)
    memory = TranslationMemory(conn, uow)
    adapters = default_registry()

    if backend is not None:
        backends = BackendRegistry()
        backends.register(backend)
    else:
        backends = build_backends(backend_name)

    renderers = SfxRendererRegistry()
    renderers.register(EnglishSfxRenderer())

    pipeline = TranslationPipeline(repository, glossary, memory, adapters, backends, renderers,
                                   uow=uow)
    approvals = ApprovalService(repository, memory, glossary, adapters, uow=uow)

    media = MediaStore(media_root or os.environ.get("SCANLATE_MEDIA", "media"))
    pages = PageImporter(repository, media, uow)
    page_service = PageService(
        repository, media,
        detector if detector is not None else build_detector(os.environ.get("SCANLATE_DETECTOR")),
        ocr if ocr is not None else build_ocr(os.environ.get("SCANLATE_OCR")),
        adapters, memory, uow)
    return Services(conn, uow, repository, glossary, memory, adapters, backends,
                    pipeline, approvals, media, pages, page_service)


# id, seq, kind, language, source text
DEMO_SEGMENTS: list[tuple[str, int, SegmentKind, str, str]] = [
    ("p012-b1", 1, SegmentKind.DIALOGUE, "ja", "田中先輩、ちょっといいですか？"),
    ("p012-b2", 2, SegmentKind.DIALOGUE, "ja", "うるせぇな…今忙しいんだよ。"),
    ("p012-s1", 3, SegmentKind.SFX, "ja", "ドクン"),
    ("p013-b1", 4, SegmentKind.DIALOGUE, "ko", "형, 진짜 괜찮아?"),
    ("p013-b2", 5, SegmentKind.DIALOGUE, "ko", "저희가 처리하겠습니다."),
    ("p013-n1", 6, SegmentKind.NARRATION, "ja", "その日、雨は止まなかった。"),
    ("p013-b3", 7, SegmentKind.DIALOGUE, "ja", "兄貴、助けてくれ！"),
]


def seed_demo(services: Services, project_id: str = DEMO_PROJECT_ID) -> Project:
    """A small, deterministic project matching the workbench examples."""
    repo = services.repository
    project = Project(project_id, "Yoru no Kissaten", "ja", "en")
    with services.uow.transaction():
        repo.create_project(project)
        repo.create_chapter(Chapter(f"{project_id}-ch1", project_id, 1, "Closing Time"))
        repo.create_page(Page(f"{project_id}-p012", project_id, 12, f"{project_id}-ch1"))
        repo.create_page(Page(f"{project_id}-p013", project_id, 13, f"{project_id}-ch1"))
        for seg_id, seq, kind, language, text in DEMO_SEGMENTS:
            page = f"{project_id}-p012" if seg_id.startswith("p012") else f"{project_id}-p013"
            repo.add_segment(Segment(seg_id, project_id, seq, kind, text,
                                     language=language, page_id=page))

        for entry in (
            GlossaryEntry(None, project_id, "ja", "田中", "en", "Tanaka",
                          Category.CHARACTER, locked=True),
            GlossaryEntry(None, project_id, "ja", "先輩", "en", "senpai",
                          Category.TITLE, locked=True, notes="upperclassman|senior"),
            GlossaryEntry(None, project_id, "ja", "兄貴", "en", "Aniki",
                          Category.RELATIONSHIP, locked=True, notes="big bro|bro|brother"),
        ):
            services.glossary.add(entry)

        # One line already approved in a previous chapter, canonical.
        services.memory.record_approved(
            project_id=project_id, source_language="ko", target_language="en",
            normalized_source="저희가 처리하겠습니다.", source_text="저희가 처리하겠습니다.",
            target_text="We'll take care of it.", locked=True)
    return project
