"""
Donut OCR engine wrapper.

Uses NAVER Clova's Donut model in document parsing mode.

Install:
    pip install transformers torch accelerate sentencepiece

Model:
    naver-clova-ix/donut-base-finetuned-cord-v2

Notes:
    - Donut is an end-to-end document understanding model that skips
      traditional OCR — it directly generates structured JSON from images.
    - The cord-v2 fine-tune produces receipts/document structure; its raw
      text output is still useful for benchmarking on prescriptions.
    - No bounding boxes (generative model).
    - Output is decoded token text — may include JSON-like structure tags.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from PIL import Image

from ..config import DONUT_MODEL_ID
from .base_engine import BaseOCREngine, EngineResult

logger = logging.getLogger(__name__)

# Prompt for Donut document parsing mode
_DONUT_TASK_PROMPT = "<s_cord-v2>"


def _clean_donut_output(text: str) -> str:
    """
    Remove XML-like Donut tags to expose the raw text content.
    E.g.: '<s_menu><s_nm>Aspirin 100mg</s_nm></s_menu>' → 'Aspirin 100mg'
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class DonutEngine(BaseOCREngine):
    """Donut document understanding engine."""

    name = "donut"
    supports_bounding_boxes = False

    def __init__(self, model_id: str = DONUT_MODEL_ID) -> None:
        super().__init__()
        self.model_id = model_id
        self._processor = None
        self._model = None

    def is_available(self) -> bool:
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel  # noqa
            return True
        except ImportError:
            self.logger.warning(
                "transformers not installed. Run: pip install transformers torch accelerate"
            )
            return False

    def _load_model(self):
        if self._processor is None:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            self.logger.info("Loading Donut model: %s", self.model_id)
            self._processor = DonutProcessor.from_pretrained(self.model_id)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)

    def _ocr_page(self, pil_img: Image.Image) -> tuple[str, str]:
        """
        Run Donut on a single page.
        Returns (raw_model_output, cleaned_text).
        """
        import torch
        pixel_values = self._processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values

        decoder_input_ids = self._processor.tokenizer(
            _DONUT_TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._model.decoder.config.max_position_embeddings,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        raw = self._processor.batch_decode(
            outputs.sequences, skip_special_tokens=False
        )[0]
        cleaned = _clean_donut_output(raw)
        return raw, cleaned

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()

        page_texts: list[str] = []
        errors: list[str] = []
        raw_outputs: list[str] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                pil_img = Image.open(img_path).convert("RGB")
                raw, cleaned = self._ocr_page(pil_img)
                page_texts.append(cleaned)
                raw_outputs.append(raw)
                self.logger.debug("Donut: page %d — %d chars cleaned", i, len(cleaned))
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")
                raw_outputs.append("")

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
                "Donut is fine-tuned on receipt documents. "
                "Structured tags in raw output are stripped — only text content is preserved.",
                "Raw model output (with tags) is not saved; only cleaned text.",
            ],
            model_id=self.model_id,
        )
