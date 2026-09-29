"""Bridge script for comic-text-detector, run OUTSIDE Scanlate's own process.

This file is not part of the ``scanlate`` package and Scanlate never imports
it. It exists to be copied into a *separate* checkout of comic-text-detector
(https://github.com/dmMaze/comic-text-detector, GPL-3.0) and run with THAT
checkout's own Python environment. Scanlate's ``CtdComicTextDetector``
(imaging/detection.py) talks to it purely by launching it as a subprocess
and reading back the JSON file it writes — never by importing it. This keeps
comic-text-detector's GPL code, and its dependencies, entirely out of
Scanlate's own repository and process.

That said: this is a cleaner separation, not a complete legal analysis. It
does not by itself resolve every licensing question a real deployment might
raise — treat it as the practical boundary for now, and revisit if Scanlate's
own licensing posture changes.

Setup (once, outside the Scanlate repo):

    git clone https://github.com/dmMaze/comic-text-detector.git
    cd comic-text-detector
    python -m venv venv
    venv/Scripts/pip install torch torchvision opencv-python numpy shapely ^
        pyclipper tqdm wandb torchsummary
    # Download the weights this script defaults to looking for:
    #   https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt
    #   -> save as data/comictextdetector.pt inside this checkout
    copy <scanlate_repo>/tools/ctd_bridge.py .

Then point Scanlate at it:

    SCANLATE_CTD_PYTHON=<checkout>/venv/Scripts/python.exe
    SCANLATE_CTD_SCRIPT=<checkout>/ctd_bridge.py
    SCANLATE_DETECTOR=auto   # or ctd, to require it

Numpy 2.0 removed a few aliases (``np.bool8``, ``np.float_``, ``np.int0``)
that comic-text-detector's ``utils/io_utils.py``/``utils/imgproc_utils.py``
still reference as of this writing; if inference fails with an
``AttributeError`` naming one of those, patch that one line in your checkout
(not in Scanlate) or pin ``numpy<2`` in the checkout's own venv.

Usage: python ctd_bridge.py --input page.png --output result.json

Output JSON contract (this is Scanlate's own contract, not comic-text-detector's
native format — CtdComicTextDetector.detect() parses exactly this shape):

    {"page_size": [w, h],
     "regions": [{"box": {"x":.., "y":.., "width":.., "height":..},
                  "confidence": null,
                  "vertical": true,
                  "mask_polygon": [[x, y], ...] or null}, ...]}

Confidence is reported as null: comic-text-detector's own reference
``inference.py``/``group_output()`` does not carry the model's per-box score
through to the ``TextBlock`` objects it returns (``TextBlock.prob`` is a
hardcoded ``1``, not the real detection confidence), and reconstructing it
would mean duplicating internal logic that could drift from upstream. Boxes
and masks are unaffected by this; only the confidence number is absent.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "data" / "comictextdetector.pt")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from inference import TextDetector  # comic-text-detector's own module

    img = cv2.imread(str(args.input))
    if img is None:
        print(f"couldn't read {args.input}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]

    detector = TextDetector(model_path=str(args.model), input_size=1024, device="cpu")
    _mask, mask_refined, blk_list = detector(img, keep_undetected_mask=True)

    regions = []
    for blk in blk_list:
        x0, y0, x1, y1 = [int(v) for v in blk.xyxy]
        polygon = _mask_to_polygon(mask_refined, x0, y0, x1, y1)
        regions.append({
            "box": {"x": x0, "y": y0, "width": max(1, x1 - x0), "height": max(1, y1 - y0)},
            "confidence": None,
            "vertical": bool(blk.vertical),
            "mask_polygon": polygon,
        })

    args.output.write_text(json.dumps({"page_size": [w, h], "regions": regions}), encoding="utf8")
    return 0


def _mask_to_polygon(mask_refined: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """The convex hull over ALL of the block's mask contours, in page
    coordinates -- a lightweight vector approximation of the raster mask.

    ``Region.polygon`` (Scanlate's mask contract) is a single simple contour,
    not a multi-part shape, so a block with several disconnected ink
    components (separate characters, or -- notably -- separate vertical
    columns with a real gap between them) can only be represented as the
    hull over all of them. That is fine when the components are close
    together, but a hull bridging a wide gap between two text columns would
    claim most of that gap as "foreground" too, starving Scanlate's
    foreground/background split of the real background sample it needs.

    Rather than guess at a smarter multi-part representation today (a real
    data-contract change, out of scope here), this abstains: if the hull's
    area is much bigger than the ink it actually contains (low "solidity"),
    the hull is a poor single-polygon stand-in for this block, and it's
    safer to emit no mask at all than a misleading one -- Scanlate already
    has a tested statistical fallback for exactly that case.
    """
    crop = mask_refined[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= 4]
    if not valid:
        return None
    points = np.concatenate(valid)
    if len(points) < 3:
        return None
    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    solidity = sum(cv2.contourArea(c) for c in valid) / hull_area
    if solidity < 0.5:
        return None
    return [[int(x) + x0, int(y) + y0] for [[x, y]] in hull]


if __name__ == "__main__":
    raise SystemExit(main())
