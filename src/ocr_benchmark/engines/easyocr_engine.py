"""
EasyOCR engine wrapper.

Uses the easyocr package for text detection and recognition.

Install:
    pip install easyocr

Notes:
    - Supports word-level bounding boxes.
    - GPU is recommended but it falls back to CPU.
    - English language is used by default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine wrapper."""

    name = "easyocr"
    supports_bounding_boxes = True

    def __init__(self) -> None:
        super().__init__()
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            self.logger.warning("easyocr not installed. Run: pip install easyocr")
            return False

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self.logger.info("Loading EasyOCR model...")
            # We initialize for English
            self._reader = easyocr.Reader(['en'], gpu=False) # Default to CPU for stability in benchmark unless configured
        return self._reader

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        reader = self._get_reader()
        
        all_words: list[WordBox] = []
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        confidences: list[float] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                # EasyOCR can take image path, cv2 image or bytes.
                # If we have preprocessed array, use it.
                source = (
                    preprocessed_arrays[i - 1]
                    if preprocessed_arrays and len(preprocessed_arrays) >= i
                    else str(img_path)
                )
                
                # result is a list of (bbox, text, confidence)
                # bbox is [[x,y], [x,y], [x,y], [x,y]]
                results = reader.readtext(source)
                
                page_word_texts: list[str] = []
                for bbox, text, conf in results:
                    # Convert bbox to left, top, width, height
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    wb = WordBox(
                        text=text,
                        confidence=float(conf),
                        left=float(x_min),
                        top=float(y_min),
                        width=float(x_max - x_min),
                        height=float(y_max - y_min),
                        polygon=[[float(p[0]), float(p[1])] for p in bbox],
                        page=i
                    )
                    all_words.append(wb)
                    page_word_texts.append(text)
                    confidences.append(float(conf))
                    
                    # We treat each result as a 'line' since EasyOCR tends to group words on lines
                    all_lines.append(LineBox(
                        text=text,
                        confidence=float(conf),
                        left=wb.left,
                        top=wb.top,
                        width=wb.width,
                        height=wb.height,
                        polygon=wb.polygon,
                        page=i
                    ))

                page_texts.append(" ".join(page_word_texts))

            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        avg_conf = (sum(confidences) / len(confidences)) if confidences else None

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=all_words,
            lines=all_lines,
            engine_version="unknown", # easyocr doesn't easily expose version
            overall_confidence=avg_conf,
            errors=errors,
            model_id="easyocr:en",
        )
