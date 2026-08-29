"""
Moondream VLM engine wrapper.
Very fast and capable tiny VLM (1.6B params).
"""

import logging
import time
from pathlib import Path
from PIL import Image
import torch

from .base_engine import BaseOCREngine, EngineResult
from ..config import MOONDREAM_MODEL_ID, HF_TOKEN

logger = logging.getLogger(__name__)

class MoondreamEngine(BaseOCREngine):
    """Moondream VLM engine."""

    name = "moondream"
    supports_bounding_boxes = False

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import login
            
            if HF_TOKEN:
                login(token=HF_TOKEN)
            
            self.logger.info(f"Loading Moondream model: {MOONDREAM_MODEL_ID}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                MOONDREAM_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float32 # CPU friendly
            ).eval()
            
            # Fix for 'HfMoondream' object has no attribute 'all_tied_weights_keys'
            if not hasattr(self._model, "all_tied_weights_keys"):
                self._model.all_tied_weights_keys = []
            self._tokenizer = AutoTokenizer.from_pretrained(MOONDREAM_MODEL_ID)

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()
        
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        for i, img_path in enumerate(image_paths, start=1):
            try:
                image = Image.open(img_path).convert("RGB")
                enc_image = self._model.encode_image(image)
                
                # Moondream specific call
                text = self._model.answer_question(
                    enc_image, 
                    "Transcribe all text from this medical prescription image exactly as written. Just the raw text.", 
                    self._tokenizer
                )
                page_texts.append(text)
                
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            engine_version="moondream2",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id=MOONDREAM_MODEL_ID,
        )
