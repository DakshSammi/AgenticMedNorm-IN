"""
Surya OCR engine wrapper.

Uses the official surya-ocr package for line-level OCR.

Install:
    pip install surya-ocr

Notes:
    - Surya performs text detection + recognition in one pass.
    - Returns line-level text with bounding boxes as polygon points.
    - Confidence scores are available per line.
    - GPU recommended for speed; falls back to CPU automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)


class SuryaEngine(BaseOCREngine):
    """Surya OCR engine — line-level text with detection boxes."""

    name = "surya"
    supports_bounding_boxes = True

    def __init__(self) -> None:
        super().__init__()
        self._rec_predictor = None
        self._det_predictor = None

    def is_available(self) -> bool:
        try:
            from surya.recognition import RecognitionPredictor  # noqa: F401
            from surya.detection import DetectionPredictor      # noqa: F401
            return True
        except ImportError:
            self.logger.warning(
                "surya-ocr not installed or incomplete. Run: pip install surya-ocr"
            )
            return False

    def _load_models(self):
        if self._rec_predictor is None:
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.settings import settings

            self.logger.info("Loading Surya detection + recognition models...")
            # Following the pattern in surya.models.load_predictors
            self._det_predictor = DetectionPredictor()
            self._rec_predictor = RecognitionPredictor(
                FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
            )

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_models()
        from surya.common.surya.schema import TaskNames

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]
        # Surya 0.17.x uses predictors directly
        # task_names= [TaskNames.ocr_with_boxes] * len(pil_images)
        
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        confidences: list[float] = []

        try:
            # Predict recognition (which internally handles detection if det_predictor passed)
            results = self._rec_predictor(
                pil_images,
                task_names=[TaskNames.ocr_with_boxes] * len(pil_images),
                det_predictor=self._det_predictor,
                math_mode=True,
                return_words=False # We want line-level as per previous implementation
            )

            for page_num, page_result in enumerate(results, start=1):
                line_texts: list[str] = []
                for line in page_result.text_lines:
                    text = line.text
                    conf = getattr(line, "confidence", None)
                    if conf is not None:
                        confidences.append(float(conf))

                    # Surya polygon is a list of [x, y]
                    poly = getattr(line, "polygon", None)
                    if poly:
                        img_w, img_h = pil_images[page_num - 1].size
                        # Convert to relative coordinates [0, 1]
                        rel_poly = [[p[0] / img_w, p[1] / img_h] for p in poly]
                        
                        # Calculate bbox from poly
                        xs = [p[0] for p in rel_poly]
                        ys = [p[1] for p in rel_poly]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        
                        lb = LineBox(
                            text=text,
                            confidence=float(conf) if conf is not None else None,
                            left=x_min,
                            top=y_min,
                            width=x_max - x_min,
                            height=y_max - y_min,
                            polygon=rel_poly,
                            page=page_num,
                        )
                    else:
                        lb = LineBox(text=text, page=page_num)

                    all_lines.append(lb)
                    line_texts.append(text)

                page_texts.append("\n".join(line_texts))

        except Exception as exc:
            err = f"Surya predictor failed: {type(exc).__name__}: {exc}"
            errors.append(err)
            self.logger.error(err)
            page_texts = [""] * len(image_paths)

        try:
            import surya
            version = getattr(surya, "__version__", "unknown")
        except Exception:
            version = "unknown"

        avg_conf = (sum(confidences) / len(confidences)) if confidences else None

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=[],  # Surya returns line-level primarily
            lines=all_lines,
            engine_version=version,
            overall_confidence=avg_conf,
            errors=errors,
            model_id="surya-ocr:0.17.1:multilingual",
        )
