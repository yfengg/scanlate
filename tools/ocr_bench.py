"""Generate a vertical-Japanese comic-style page and compare OCR backends on it.

The page is drawn here rather than taken from a published comic: no manga is
legally redistributable for this purpose, so the alternative would be reporting
numbers from material I can't ship. It is real vertical Japanese typeset in
bubbles with a CJK font, which is the property that matters for OCR, but it is
*not* scanned artwork — no screentone, no hand lettering, no JPEG noise, no
skew. Treat the numbers as a floor, not as manga accuracy.

    python tools/ocr_bench.py            # build the page and compare backends
    python tools/ocr_bench.py --page out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from scanlate.imaging.layout import BoundingBox, TextOrientation  # noqa: E402
from scanlate.imaging.manga_ocr_backend import MangaOcr, OcrRouter  # noqa: E402
from scanlate.imaging.ocr import NullOcr, OcrError, TesseractOcr  # noqa: E402

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]

# (text, bubble box, vertical?) — the text is the ground truth.
BUBBLES: list[tuple[str, tuple[int, int, int, int], bool]] = [
    ("田中先輩、ちょっといいですか？", (60, 60, 250, 330), True),
    ("うるせぇな…今忙しいんだよ。", (330, 90, 500, 340), True),
    ("その日、雨は止まなかった。", (70, 420, 480, 500), False),
    ("兄貴、助けてくれ！", (80, 560, 230, 800), True),
    ("何してるの？", (300, 600, 420, 790), True),
    ("行くぞ！", (450, 620, 540, 780), True),
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise SystemExit("No CJK font found; install fonts-noto-cjk.")


def draw_vertical(draw: ImageDraw.ImageDraw, text: str, box, font) -> None:
    """Japanese vertical setting: top to bottom, columns right to left."""
    x0, y0, x1, y1 = box
    size = font.size
    per_column = max(1, (y1 - y0 - 20) // (size + 4))
    columns = [text[i:i + per_column] for i in range(0, len(text), per_column)]
    x = x1 - size - 14
    for column in columns:
        y = y0 + 14
        for character in column:
            draw.text((x, y), character, font=font, fill="black")
            y += size + 4
        x -= size + 8


def draw_horizontal(draw: ImageDraw.ImageDraw, text: str, box, font) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0 + 16, (y0 + y1) // 2 - font.size // 2), text, font=font, fill="black")


def build_page(path: Path, width: int = 600, height: int = 850) -> Path:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(26)

    draw.rectangle([20, 20, width - 20, height - 20], outline="black", width=3)
    draw.line([(20, 400), (width - 20, 400)], fill="black", width=3)
    draw.line([(270, 520), (270, height - 20)], fill="black", width=3)

    for text, box, vertical in BUBBLES:
        x0, y0, x1, y1 = box
        draw.ellipse([x0 - 12, y0 - 12, x1 + 12, y1 + 12], fill="white", outline="black", width=3)
        (draw_vertical if vertical else draw_horizontal)(draw, text, box, font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def compare(page_path: Path) -> int:
    data = page_path.read_bytes()
    tesseract = TesseractOcr()
    manga = MangaOcr()
    router = OcrRouter(tesseract if tesseract.available() else NullOcr(), {"ja": manga})

    print(f"page: {page_path} ({Image.open(page_path).size[0]}×{Image.open(page_path).size[1]})")
    print(f"tesseract available: {tesseract.available()}  "
          f"langs={tesseract.languages()}")
    print(f"manga-ocr available: {manga.available()}"
          + (f"  ({manga.failure})" if manga.failure else ""))
    print(f"router picks for ja: {router.backend_for('ja').name}\n")

    header = f"{'ground truth':32} | {'tesseract':32} | {'manga-ocr':32}"
    print(header)
    print("-" * len(header))

    exact = {"tesseract": 0, "manga-ocr": 0}
    for text, box, vertical in BUBBLES:
        x0, y0, x1, y1 = box
        region = BoundingBox(x0 - 14, y0 - 14, (x1 - x0) + 28, (y1 - y0) + 28)
        orientation = TextOrientation.VERTICAL_RL if vertical else TextOrientation.HORIZONTAL
        outputs = {}
        for name, backend in (("tesseract", tesseract), ("manga-ocr", manga)):
            if not backend.available():
                outputs[name] = "(unavailable)"
                continue
            try:
                result = backend.recognize(data, region, "ja", orientation)
                outputs[name] = result.text
                if result.text.replace(" ", "") == text:
                    exact[name] += 1
            except OcrError as error:
                outputs[name] = f"(failed: {error})"
        print(f"{text:32} | {outputs['tesseract'][:32]:32} | {outputs['manga-ocr'][:32]:32}")

    print()
    for name in ("tesseract", "manga-ocr"):
        print(f"{name}: {exact[name]}/{len(BUBBLES)} exact matches")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, default=Path("build/vertical_ja_page.png"))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    path = build_page(args.page)
    print(f"wrote {path}")
    return 0 if args.build_only else compare(path)


if __name__ == "__main__":
    raise SystemExit(main())
