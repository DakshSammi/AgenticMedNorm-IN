"""
Tesseract OCR engine wrapper.

Uses the pytesseract package (wrapper for Google's Tesseract-OCR).

Install:
    pip install pytesseract
    # ALSO requires Tesseract-OCR binary installed on the system.

Notes:
    - Supports word-level bounding boxes via image_to_data.
    - Performance depends on Tesseract version and trained data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine wrapper."""

    name = "tesseract"
    supports_bounding_boxes = True

    def is_available(self) -> bool:
        try:
            import pytesseract
            # Verify if tesseract binary is actually found
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, Exception):
            self.logger.warning(
                "Tesseract not available. Ensure 'pytesseract' is installed and "
                "Tesseract-OCR binary is in your PATH."
            )
            return False

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        import pytesseract
        from pytesseract import Output
        
        all_words: list[WordBox] = []
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        confidences: list[float] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                # Load image
                if preprocessed_arrays and len(preprocessed_arrays) >= i:
                    source = preprocessed_arrays[i - 1]
                else:
                    source = Image.open(img_path).convert("RGB")
                
                # Get detailed data including bounding boxes
                # config='--psm 3' for fully automatic page segmentation
                data = pytesseract.image_to_data(source, output_type=Output.DICT, config='--psm 3')
                
                page_full_text = pytesseract.image_to_string(source, config='--psm 3')
                page_texts.append(page_full_text.strip())

                n_boxes = len(data['text'])
                for j in range(n_boxes):
                    text = data['text'][j].strip()
                    if not text:
                        continue
                    
                    conf = data['conf'][j]
                    if conf == -1: # Tesseract uses -1 for non-text blocks
                        continue
                        
                    conf_norm = conf / 100.0
                    confidences.append(conf_norm)
                    
                    wb = WordBox(
                        text=text,
                        confidence=conf_norm,
                        left=float(data['left'][j]),
                        top=float(data['top'][j]),
                        width=float(data['width'][j]),
                        height=float(data['height'][j]),
                        page=i
                    )
                    all_words.append(wb)
                    
                    # Tesseract provides line numbers, but for simplicity here we treat 
                    # each word block as an individual entry in words. 
                    # LineBox could be aggregated by 'line_num' if needed.

            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        avg_conf = (sum(confidences) / len(confidences)) if confidences else None
        
        version = "unknown"
        try:
            version = str(pytesseract.get_tesseract_version())
        except:
            pass

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=all_words,
            lines=[], # Aggregating lines from Tesseract data is possible but omitted for brevity
            engine_version=version,
            overall_confidence=avg_conf,
            errors=errors,
            model_id="tesseract",
        )
