"""Serialization for the HTTP layer.

The wire format is the SegmentView contract and nothing else: no row ids, no
SQL shapes, no backend internals the translator can't act on. Keeping it in one
module means the frontend has a single place to check against.
"""
from __future__ import annotations

from typing import Any

from ..glossary.store import GlossaryEntry
from ..memory.store import MemoryEntry
from ..pipeline.approval import GlossarySuggestion
from ..storage.models import Chapter, Page, Project, Segment
from ..workbench.view import Annotation, SegmentView, Warning_


def span(value) -> dict | None:
    return None if value is None else {"start": value.start, "end": value.end}


def annotation(item: Annotation) -> dict:
    return {
        "start": item.span.start,
        "end": item.span.end,
        "type": item.type.value,
        "target": item.target,
        "label": item.label,
        "locked": item.locked,
        "origin": item.origin.value if item.origin else None,
        "data": item.data,
    }


def warning(item: Warning_) -> dict:
    return {
        "code": item.code,
        "severity": item.severity.name,
        "message": item.message,
        "fingerprint": item.fingerprint,
        "span": span(item.span),
        "actions": [{"kind": a.kind, "label": a.label, "value": a.value} for a in item.actions],
        "data": item.data,
    }


def segment_view(view: SegmentView) -> dict:
    return {
        "id": view.id,
        "project_id": view.project_id,
        "page_id": view.page_id,
        "order": view.order,
        "kind": view.kind.value,
        "language": view.language,
        "target_language": view.target_language,
        "source": view.source,
        "candidate": view.candidate,
        "annotations": [annotation(a) for a in view.annotations],
        "alternatives": [{"text": a.text, "note": a.note, "reason": a.reason}
                         for a in view.alternatives],
        "features": _clean(view.features),
        "warnings": [warning(w) for w in view.warnings],
        "status": view.status.value,
        "origin": view.origin.value,
        "locked": view.locked,
        "needs_review": view.needs_review,
    }


def project(item: Project, chapters: list[Chapter] | None = None,
            pages: list[Page] | None = None) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "default_source_language": item.default_source_language,
        "target_language": item.target_language,
        "chapters": [{"id": c.id, "number": c.number, "title": c.title}
                     for c in (chapters or [])],
        "pages": [{"id": p.id, "number": p.number, "chapter_id": p.chapter_id,
                   "image_path": p.image_path} for p in (pages or [])],
    }


def glossary_entry(entry: GlossaryEntry) -> dict:
    return {
        "id": entry.id,
        "source_language": entry.source_language,
        "source_term": entry.source_term,
        "target_language": entry.target_language,
        "target_term": entry.target_term,
        "category": entry.category.value,
        "locked": entry.locked,
        "notes": entry.notes,
    }


def memory_entry(entry: MemoryEntry) -> dict:
    return {
        "id": entry.id,
        "source_language": entry.source_language,
        "target_language": entry.target_language,
        "source_text": entry.source_text,
        "target_text": entry.target_text,
        "locked": entry.locked,
        "canonical": entry.canonical,
        "occurrences": entry.occurrences,
        "updated_at": entry.updated_at,
    }


def suggestion(item: GlossarySuggestion | None) -> dict | None:
    if item is None:
        return None
    return {
        "source_term": item.source_term,
        "target_term": item.target_term,
        "category": item.category.value,
        "reason": item.reason,
        "source_language": item.source_language,
        "target_language": item.target_language,
        "explicit": item.explicit,
    }


def _clean(features: dict[str, Any]) -> dict[str, Any]:
    """Features are for the translator; drop anything that isn't."""
    return {k: v for k, v in features.items() if k != "backend" or v}


def region(segment: Segment) -> dict | None:
    """Region geometry in image coordinates — the only coordinates that persist."""
    if segment.region is None or segment.region.box is None:
        return None
    box = segment.region.box
    return {
        "x": box.x, "y": box.y, "width": box.width, "height": box.height,
        "kind": segment.region.kind.value,
        "orientation": segment.region.orientation.value,
        "reading_order": segment.region.reading_order,
    }


def page(item: Page, segments: list[Segment] | None = None) -> dict:
    return {
        "id": item.id,
        "project_id": item.project_id,
        "chapter_id": item.chapter_id,
        "number": item.number,
        "width": item.width,
        "height": item.height,
        "image_url": f"/api/pages/{item.id}/image",
        "has_image": bool(item.image_path),
        "regions": [{"segment_id": s.id, "seq": s.seq, "kind": s.kind.value,
                     "language": s.language, "source_text": s.source_text,
                     "status": s.state.status.value, "region": region(s)}
                    for s in (segments or []) if s.region and s.region.box],
    }


def region_render_result(item) -> dict:
    return {
        "segment_id": item.segment_id,
        "kind": item.kind.value,
        "rendered": item.rendered,
        "reason": item.reason,
    }


def render_report(report) -> dict:
    """Which regions rendered and which were left as original artwork —
    enough for the UI to say "N regions need manual handling" rather than
    silently presenting an incomplete page as finished."""
    return {
        "results": [region_render_result(r) for r in report.results],
        "flagged": [region_render_result(r) for r in report.flagged],
    }


def ocr_result(result) -> dict:
    return {
        "segment_id": result.segment_id,
        "text": result.text,
        "language": result.language,
        "confidence": result.confidence,
        "confidence_available": result.confidence_available,
        "low_confidence": result.low_confidence,
        "backend": result.backend,
        "fell_back_from": result.fell_back_from,
        "reopened": result.reopened,
        "orientation": result.orientation,
        "needs_review": result.needs_review,
        "ok": result.ok,
        "error": result.error,
    }
