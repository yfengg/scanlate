"""Imaging: geometry, page import, detection, OCR, and the page workflow service.

``layout`` is imported eagerly because ``storage.models`` depends on it for the
Region/RenderSettings fields. Everything else is resolved lazily: those modules
import ``storage`` themselves, and importing them here would close the loop.
"""
from __future__ import annotations

from .layout import (BoundingBox, MaskRef, Polygon, Region, RegionKind, RenderSettings,
                     TextOrientation, region_from_dict, region_to_dict,
                     render_from_dict, render_to_dict)

_LAZY = {
    "MediaStore": ".importer", "PageImporter": ".importer", "ImageInfo": ".importer",
    "ImageError": ".importer", "PageConflict": ".importer", "inspect": ".importer",
    "TextRegionDetector": ".detection", "DetectedRegion": ".detection",
    "DetectorError": ".detection", "DetectorUnavailable": ".detection",
    "OpenCvComicTextDetector": ".detection", "NullDetector": ".detection",
    "build_detector": ".detection",
    "OcrBackend": ".ocr", "OcrResult": ".ocr", "OcrError": ".ocr",
    "TesseractOcr": ".ocr", "NullOcr": ".ocr", "build_ocr": ".ocr",
    "MangaOcr": ".manga_ocr_backend", "OcrRouter": ".manga_ocr_backend",
    "PageService": ".service", "RegionOcr": ".service", "RegionError": ".service",
    "SourceTextChange": ".service", "DetectionResult": ".service",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(module_name, __name__), name)


__all__ = ["BoundingBox", "Polygon", "Region", "RegionKind", "RenderSettings",
           "TextOrientation", "MaskRef", "region_to_dict", "region_from_dict",
           "render_to_dict", "render_from_dict", *_LAZY]
