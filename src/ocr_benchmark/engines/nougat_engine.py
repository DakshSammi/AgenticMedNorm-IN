"""
Nougat OCR engine wrapper.

Uses Facebook's Nougat model for document OCR.

Install:
    pip install nougat-ocr
    # OR via HuggingFace transformers:
    pip install transformers torch accelerate

Model:
    facebook/nougat-base

Notes:
    - Nougat is designed for academic/scientific PDF documents.
    - It produces Markdown-formatted text output.
    - Including it in this benchmark to quantify its limitations on
      handwritten prescriptions vs its intended domain.
    - No bounding boxes (generative model).
    - Input: PIL image → output: Markdown text string.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from ..config import NOUGAT_MODEL_ID
from .base_engine import BaseOCREngine, EngineResult

logger = logging.getLogger(__name__)


class NougatEngine(BaseOCREngine):
    """Facebook Nougat document OCR engine."""

    name = "nougat"
    supports_bounding_boxes = False

    def __init__(self, model_id: str = NOUGAT_MODEL_ID) -> None:
        super().__init__()
        self.model_id = model_id
        self._processor = None
        self._model = None

    def is_available(self) -> bool:
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel  # noqa
            return True
        except ImportError:
            self.logger.warning(
                "transformers not installed. Run: pip install transformers torch accelerate"
            )
            return False

    def _load_model(self):
        if self._processor is None:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            self.logger.info("Loading Nougat model: %s", self.model_id)
            self._processor = NougatProcessor.from_pretrained(self.model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)

    def _ocr_page(self, pil_img: Image.Image) -> str:
        """Run Nougat inference on a single page image."""
        import torch
        pixel_values = self._processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=3096,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
            )

        text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Remove Nougat's repeated-sequence artifacts
        text = self._processor.post_process_generation(text, fix_markdown=False)
        return text

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()

        page_texts: list[str] = []
        errors: list[str] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                pil_img = Image.open(img_path).convert("RGB")
                text = self._ocr_page(pil_img)
                page_texts.append(text)
                self.logger.debug("Nougat: page %d — %d chars", i, len(text))
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
            words=[],
            lines=[],
            engine_version=version,
            errors=errors,
            warnings=[
                "Nougat is designed for academic documents. "
                "Performance on handwritten prescriptions may be very limited."
            ],
            model_id=self.model_id,
        )
