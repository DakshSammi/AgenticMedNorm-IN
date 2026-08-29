"""
Gemini / MedGemma VLM engine wrapper.

Uses Google Generative AI SDK (API-based).
"""

import logging
from pathlib import Path
from PIL import Image
import time

from .base_engine import BaseOCREngine, EngineResult
from ..config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class GeminiEngine(BaseOCREngine):
    """Gemini Vision engine for prescription transcription."""

    name = "gemini"
    supports_bounding_boxes = False  # Standard Gemini doesn't return boxes in simple OCR mode

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def is_available(self) -> bool:
        if not GOOGLE_API_KEY:
            self.logger.warning("GOOGLE_API_KEY not found in environment.")
            return False
        try:
            import google.generativeai as genai
            return True
        except ImportError:
            self.logger.warning("google-generativeai package not installed.")
            return False

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            # Use gemini-1.5-flash for speed or gemini-1.5-pro for better accuracy
            self._model = genai.GenerativeModel('gemini-1.5-flash')
        return self._model

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        model = self._get_model()
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        # Prompt for raw transcription
        prompt = (
            "You are a medical OCR specialist. "
            "Transcribe ALL text from this medical prescription image exactly as written. "
            "Maintain the layout structure if possible. "
            "Do not interpret or correct spelling. Only provide the raw transcription."
        )

        for i, img_path in enumerate(image_paths, start=1):
            try:
                img = Image.open(img_path).convert("RGB")
                response = model.generate_content([prompt, img])
                page_texts.append(response.text)
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=[],
            lines=[],
            engine_version="gemini-1.5-flash",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id="gemini-1.5-flash",
        )
