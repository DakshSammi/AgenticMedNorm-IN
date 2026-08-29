"""
docTR OCR engine wrapper.

Uses the doctr.models.ocr_predictor with pretrained weights.

Install:
    pip install "python-doctr[torch]"
    # OR for TensorFlow backend:
    pip install "python-doctr[tf]"

Notes:
    - Returns word-level bounding boxes as (left, top, right, bottom)
      relative coordinates [0, 1].
    - Confidence score per word.
    - Multi-page handled by passing all images to the predictor at once.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)


class DocTREngine(BaseOCREngine):
    """docTR OCR engine using the ocr_predictor."""

    name = "doctr"
    supports_bounding_boxes = True

    def __init__(self) -> None:
        super().__init__()
        self._predictor = None

    def is_available(self) -> bool:
        try:
            from doctr.models import ocr_predictor  # noqa: F401
            return True
        except ImportError:
            self.logger.warning(
                "docTR not installed. Run: pip install python-doctr[torch]"
            )
            return False

    def _get_predictor(self):
        if self._predictor is None:
            from doctr.models import ocr_predictor
            self._predictor = ocr_predictor(pretrained=True)
        return self._predictor

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        try:
            from doctr.io import DocumentFile
        except ImportError as exc:
            return EngineResult(errors=[str(exc)])

        predictor = self._get_predictor()
        all_words: list[WordBox] = []
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        confidences: list[float] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                doc = DocumentFile.from_images([str(img_path)])
                result = predictor(doc)

                page_word_texts: list[str] = []
                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_word_texts: list[str] = []
                            line_words: list[WordBox] = []

                            for word in line.words:
                                geo = word.geometry  # ((x0,y0),(x1,y1))
                                x0, y0 = geo[0]
                                x1, y1 = geo[1]
                                conf = float(word.confidence) if word.confidence is not None else None
                                wb = WordBox(
                                    text=word.value,
                                    confidence=conf,
                                    left=float(x0),
                                    top=float(y0),
                                    width=float(x1 - x0),
                                    height=float(y1 - y0),
                                    page=i,
                                )
                                all_words.append(wb)
                                line_words.append(wb)
                                line_word_texts.append(word.value)
                                if conf is not None:
                                    confidences.append(conf)

                            line_text = " ".join(line_word_texts)
                            page_word_texts.append(line_text)

                            if line_words:
                                # Aggregate line box from word boxes
                                lft = min(w.left for w in line_words if w.left is not None)
                                tp = min(w.top for w in line_words if w.top is not None)
                                rgt = max(
                                    (w.left or 0) + (w.width or 0) for w in line_words
                                )
                                btm = max(
                                    (w.top or 0) + (w.height or 0) for w in line_words
                                )
                                all_lines.append(LineBox(
                                    text=line_text,
                                    left=lft,
                                    top=tp,
                                    width=rgt - lft,
                                    height=btm - tp,
                                    page=i,
                                ))

                page_texts.append("\n".join(page_word_texts))

            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        try:
            import doctr
            version = getattr(doctr, "__version__", "unknown")
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
            model_id="doctr:ocr_predictor:pretrained",
        )
