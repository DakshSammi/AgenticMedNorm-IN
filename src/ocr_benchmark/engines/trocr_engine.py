"""
TrOCR engine wrapper.

Uses Microsoft's TrOCR handwritten model from HuggingFace.

Install:
    pip install transformers torch accelerate Pillow

Model:
    microsoft/trocr-large-handwritten

Strategy:
    TrOCR operates on single-line image crops. This wrapper:
    1. Segments the image into text-line crops using a horizontal
       projection profile heuristic.
    2. Runs TrOCR on each crop.
    3. Concatenates results in reading order.

Approximate line bounding boxes are derived from the segmentation step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..config import TROCR_MODEL_ID
from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)

# Minimum number of rows a line region must have (filters noise)
_MIN_LINE_HEIGHT_PX = 10
# Horizontal projection threshold: rows with fewer dark pixels = gap
_PROJECTION_THRESHOLD = 0.02  # fraction of image width


def _segment_lines(gray: np.ndarray) -> list[tuple[int, int]]:
    """
    Segment a grayscale image into (y_start, y_end) row ranges
    using a horizontal projection profile.

    Works by finding contiguous bands of rows that contain text pixels
    (rows with projection sum > threshold).
    """
    # Normalise: 0=background, 1=text
    norm = 1.0 - (gray.astype(np.float32) / 255.0)
    projection = norm.mean(axis=1)  # per-row mean dark-pixel density

    threshold = _PROJECTION_THRESHOLD
    in_text = False
    segments: list[tuple[int, int]] = []
    start = 0

    for y, val in enumerate(projection):
        if not in_text and val > threshold:
            in_text = True
            start = y
        elif in_text and val <= threshold:
            in_text = False
            if y - start >= _MIN_LINE_HEIGHT_PX:
                segments.append((start, y))

    # Close last segment if image ends in text
    if in_text and len(gray) - start >= _MIN_LINE_HEIGHT_PX:
        segments.append((start, len(gray)))

    return segments


def _crop_line(pil_img: Image.Image, y_start: int, y_end: int) -> Image.Image:
    """Crop a single line from the full-page PIL image."""
    w, h = pil_img.size
    # Add small vertical padding
    pad = 4
    return pil_img.crop((0, max(0, y_start - pad), w, min(h, y_end + pad)))


class TrOCREngine(BaseOCREngine):
    """TrOCR handwritten model engine with line-segmentation preprocessing."""

    name = "trocr"
    supports_bounding_boxes = False  # approximate only from segmentation

    def __init__(self, model_id: str = TROCR_MODEL_ID) -> None:
        super().__init__()
        self.model_id = model_id
        self._processor = None
        self._model = None

    def is_available(self) -> bool:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # noqa
            return True
        except ImportError:
            self.logger.warning(
                "transformers not installed. Run: pip install transformers torch accelerate"
            )
            return False

    def _load_model(self):
        if self._processor is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self.logger.info("Loading TrOCR model: %s", self.model_id)
            self._processor = TrOCRProcessor.from_pretrained(self.model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)

    def _ocr_line(self, crop: Image.Image) -> str:
        """Run TrOCR inference on a single-line crop."""
        import torch
        pixel_values = self._processor(images=crop, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _process_page(
        self, img_path: Path, page_num: int, preprocessed_array: Optional[np.ndarray]
    ) -> tuple[str, list[LineBox]]:
        pil_img = Image.open(img_path).convert("RGB")
        img_w, img_h = pil_img.size

        # Use preprocessed grayscale for segmentation if available
        if preprocessed_array is not None:
            if len(preprocessed_array.shape) == 3:
                import cv2
                gray = cv2.cvtColor(preprocessed_array, cv2.COLOR_BGR2GRAY)
            else:
                gray = preprocessed_array
        else:
            gray = np.array(pil_img.convert("L"))

        segments = _segment_lines(gray)
        if not segments:
            # Fallback: treat entire image as one line
            segments = [(0, img_h)]

        self.logger.debug(
            "TrOCR: page %d — found %d line segments", page_num, len(segments)
        )

        lines: list[LineBox] = []
        page_text_parts: list[str] = []

        for y_start, y_end in segments:
            crop = _crop_line(pil_img, y_start, y_end)
            # Skip very thin crops (noise)
            if crop.size[1] < _MIN_LINE_HEIGHT_PX:
                continue
            try:
                text = self._ocr_line(crop)
            except Exception as exc:
                self.logger.warning("TrOCR line inference failed: %s", exc)
                text = ""

            if text.strip():
                page_text_parts.append(text)
                lines.append(LineBox(
                    text=text,
                    top=float(y_start) / img_h,
                    left=0.0,
                    width=1.0,
                    height=float(y_end - y_start) / img_h,
                    page=page_num,
                ))

        return "\n".join(page_text_parts), lines

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()

        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                pre = (
                    preprocessed_arrays[i - 1]
                    if preprocessed_arrays and len(preprocessed_arrays) >= i
                    else None
                )
                pt, ls = self._process_page(img_path, i, pre)
                page_texts.append(pt)
                all_lines.extend(ls)
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        try:
            import transformers
            version = transformers.__version__
        except Exception:
            version = "unknown"

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=[],  # TrOCR does not produce word-level boxes
            lines=all_lines,
            engine_version=version,
            errors=errors,
            warnings=["Bounding boxes are approximate from line segmentation."],
            model_id=self.model_id,
        )
