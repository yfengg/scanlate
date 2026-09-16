"""Page import.

Images live on disk in an application-managed media directory; the database
keeps a stable relative reference plus the dimensions everything else needs to
map region coordinates. Blobs stay out of SQLite — a chapter of pages is tens
of megabytes, and the database is queried on every keystroke.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ..storage.models import Page
from ..storage.repository import ProjectRepository
from ..storage.unit_of_work import UnitOfWork

SUPPORTED = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
MAX_PIXELS = 80_000_000          # a guard against decompression bombs
# Project ids and stored filenames are path components, so they are restricted
# to characters that cannot traverse or escape.
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ImageError(Exception):
    """Unsupported, corrupt, or unreadable image."""


class PageConflict(Exception):
    """A page already occupies that number in the chapter."""


@dataclass
class ImageInfo:
    format: str
    width: int
    height: int
    data: bytes = b""              # pixels as stored, after orientation is applied
    reoriented: bool = False


def inspect(data: bytes) -> ImageInfo:
    """Validate an image, normalize its orientation, and read its dimensions.

    A JPEG from a phone or scanner can carry an EXIF orientation flag, which
    browsers honour and raw pixel readers do not. Left alone, the page would
    display rotated relative to the pixels OCR crops from, and every region
    coordinate would be wrong. So the rotation is baked into the pixels once,
    here, and everything downstream — stored file, width/height, browser,
    OCR, region coordinates — refers to the same orientation.
    """
    from io import BytesIO

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()                     # catches truncation and bad headers
        with Image.open(BytesIO(data)) as image:
            fmt = image.format
            oriented = ImageOps.exif_transpose(image)
            width, height = oriented.size
            reoriented = oriented.size != image.size or _has_exif_rotation(image)
            if reoriented:
                buffer = BytesIO()
                save_format = fmt if fmt in SUPPORTED else "PNG"
                params = {"quality": 95} if save_format == "JPEG" else {}
                oriented.save(buffer, format=save_format, **params)
                data = buffer.getvalue()
    except UnidentifiedImageError:
        raise ImageError("That file isn't an image Scanlate can read.")
    except Exception as error:                 # PIL raises a wide spread here
        raise ImageError(f"The image couldn't be read: {error}")

    if fmt not in SUPPORTED:
        raise ImageError(f"{fmt} isn't supported. Use PNG, JPEG or WebP.")
    if width * height > MAX_PIXELS:
        raise ImageError(f"That image is too large ({width}×{height}).")
    if width < 1 or height < 1:
        raise ImageError("That image has no pixels.")
    return ImageInfo(fmt, width, height, data, reoriented)


def _has_exif_rotation(image) -> bool:
    try:
        exif = image.getexif()
    except Exception:
        return False
    return bool(exif) and exif.get(274, 1) not in (1, None)


class MediaStore:
    """Content-addressed page images under one managed directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _component(self, value: str, label: str) -> str:
        if not SAFE_COMPONENT.match(value or "") or value in {".", ".."}:
            raise ImageError(f"Unsafe {label}: {value!r}")
        return value

    def project_dir(self, project_id: str) -> Path:
        path = self.root / self._component(project_id, "project id")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def store(self, project_id: str, data: bytes, info: ImageInfo) -> str:
        """Write the image and return the reference kept in the database.

        Named by content hash so re-importing the same file is idempotent and
        two pages can never collide on a filename.
        """
        digest = hashlib.sha256(data).hexdigest()[:32]
        name = f"{digest}{SUPPORTED[info.format]}"
        path = self.project_dir(project_id) / name
        if not path.exists():
            path.write_bytes(data)
        return f"{project_id}/{name}"

    def path(self, reference: str) -> Path:
        """Resolve a stored reference, refusing anything outside the root.

        Containment is checked with real path semantics, not a string prefix:
        ``/media-evil`` starts with ``/media`` but is a different directory, and
        symlinks resolve before the check.
        """
        parts = (reference or "").replace("\\", "/").split("/")
        if len(parts) != 2:
            raise ImageError(f"Invalid image reference: {reference!r}")
        project_id, filename = parts
        self._component(project_id, "project id")
        self._component(filename, "filename")

        root = self.root.resolve()
        candidate = (root / project_id / filename).resolve()
        if not candidate.is_relative_to(root):
            raise ImageError("Invalid image reference.")
        return candidate

    def read(self, reference: str) -> bytes:
        path = self.path(reference)
        if not path.exists():
            raise ImageError(f"The image file for this page is missing ({reference}).")
        return path.read_bytes()

    def exists(self, reference: str | None) -> bool:
        try:
            return bool(reference) and self.path(reference).exists()
        except ImageError:
            return False


class PageImporter:
    def __init__(self, repository: ProjectRepository, media: MediaStore,
                 uow: UnitOfWork | None = None):
        self.repository = repository
        self.media = media
        self.uow = uow or repository.uow

    def import_page(self, project_id: str, chapter_id: str, number: int, data: bytes,
                    page_id: str | None = None, replace: bool = False) -> Page:
        """Import one image as a page.

        An existing page at that number is never overwritten silently: either
        pass ``replace=True``, which updates the artwork in place and keeps the
        page's segments, or pick another number.
        """
        info = inspect(data)
        existing = self.repository.find_page(chapter_id, number)
        if existing and not replace:
            raise PageConflict(
                f"Page {number} already exists in this chapter. Replace it or use "
                f"another number.")

        reference = self.media.store(project_id, info.data or data, info)
        with self.uow.transaction():
            if existing:
                # Same logical page, new artwork: segments and translations survive.
                self.repository.set_page_image(existing.id, reference, info.width, info.height)
                return self.repository.get_page(existing.id)
            page = Page(page_id or f"{chapter_id}-p{number:03d}", project_id, number,
                        chapter_id, reference, info.width, info.height)
            self.repository.create_page(page)
        return page
