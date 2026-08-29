"""
Ollama VLM engine wrapper.

Uses local Ollama API for models like llama3-vision, deepseek-vl, etc.
"""

import logging
import base64
import requests
import time
from pathlib import Path

from .base_engine import BaseOCREngine, EngineResult
from ..config import OLLAMA_BASE_URL, OLLAMA_MODEL_ID

logger = logging.getLogger(__name__)

class OllamaEngine(BaseOCREngine):
    """Local Ollama VLM engine."""

    name = "ollama"
    supports_bounding_boxes = False

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            return resp.status_code == 200
        except:
            self.logger.warning(f"Ollama server not found at {OLLAMA_BASE_URL}")
            return False

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        prompt = (
            "Transcribe ALL text from this medical prescription image exactly as written. "
            "Do not interpret or correct. Just raw text."
        )

        for i, img_path in enumerate(image_paths, start=1):
            try:
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")

                payload = {
                    "model": OLLAMA_MODEL_ID,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False
                }
                
                resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                resp.raise_for_status()
                page_texts.append(resp.json().get("response", ""))
                
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            engine_version=f"ollama:{OLLAMA_MODEL_ID}",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id=OLLAMA_MODEL_ID,
        )
