"""
PaddleOCR engine wrapper.

Uses PaddleOCR with angle classification enabled (handles rotated text
common in handwritten prescriptions).

Install:
    pip install paddlepaddle paddleocr

Notes:
    - Runs on CPU by default (use_gpu=False).
    - Returns word-level bounding boxes as polygon points [[x,y], ...].
    - Confidence score is per-word.
    - English only initially (lang='en').
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)


def _poly_to_bbox(points: list) -> dict:
    """Convert a 4-point polygon [[x,y],...] to left/top/width/height."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        "left": float(x_min),
        "top": float(y_min),
        "width": float(x_max - x_min),
        "height": float(y_max - y_min),
        "polygon": [[float(p[0]), float(p[1])] for p in points],
    }


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine with angle classification."""

    name = "paddleocr"
    supports_bounding_boxes = True

    def __init__(self) -> None:
        super().__init__()
        self._ocr = None

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            return True
        except ImportError:
            self.logger.warning(
                "PaddleOCR not installed. Run: pip install paddlepaddle paddleocr"
            )
            return False

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
            )
        return self._ocr

    def _process_page(
        self,
        img: np.ndarray,
        page_num: int,
        preprocessed_array=None,
    ) -> tuple[str, list[WordBox], list[LineBox]]:
        ocr = self._get_ocr()
        source = preprocessed_array if preprocessed_array is not None else img
        
        try:
            # First attempt with angle classification
            result = ocr.ocr(source, cls=True)
        except TypeError as e:
            if "cls" in str(e) or "predict" in str(e):
                self.logger.warning(f"PaddleOCR.ocr() does not support 'cls' argument in this version. Retrying without it.")
                result = ocr.ocr(source)
            else:
                raise e

        words: list[WordBox] = []
        line_texts: list[str] = []

        # PaddleOCR result: list of pages, each page is list of
        # [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, conf)]
        if not result or result[0] is None:
            return "", [], []

        for line in result[0]:
            if line is None:
                continue
            polygon, (text, conf) = line
            bb = _poly_to_bbox(polygon)
            words.append(WordBox(
                text=text,
                confidence=float(conf),
                left=bb["left"],
                top=bb["top"],
                width=bb["width"],
                height=bb["height"],
                polygon=bb["polygon"],
                page=page_num,
            ))
            line_texts.append(text)

        # Sort words top-to-bottom for natural reading order
        words.sort(key=lambda w: (w.top or 0, w.left or 0))
        line_texts_sorted = [w.text for w in words]

        # Aggregate into line-level boxes (each PaddleOCR result row = one line)
        lines = [
            LineBox(
                text=w.text,
                confidence=w.confidence,
                left=w.left,
                top=w.top,
                width=w.width,
                height=w.height,
                polygon=w.polygon,
                page=page_num,
            )
            for w in words
        ]

        page_text = "\n".join(line_texts_sorted)
        return page_text, words, lines

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        all_words: list[WordBox] = []
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        confidences: list[float] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                pre = (
                    preprocessed_arrays[i - 1]
                    if preprocessed_arrays and len(preprocessed_arrays) >= i
                    else None
                )
                pt, ws, ls = self._process_page(img, i, pre)
                page_texts.append(pt)
                all_words.extend(ws)
                all_lines.extend(ls)
                confidences.extend(
                    w.confidence for w in ws if w.confidence is not None
                )
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        try:
            import paddleocr
            version = getattr(paddleocr, "__version__", "unknown")
        except Exception:
            version = "unknown"

        avg_conf = (sum(confidences) / len(confidences)) if confidences else None

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=all_words,
            lines=all_lines,
            engine_version=version,
            overall_confidence=avg_conf,
            errors=errors,
            model_id="paddleocr:en:angle_cls",
        )
