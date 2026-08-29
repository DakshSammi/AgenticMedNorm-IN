"""
Microsoft Florence-2 engine wrapper.
Extremely fast and accurate for its size (232M - 700M params).
"""

import logging
import time
from pathlib import Path
from PIL import Image
import torch

from .base_engine import BaseOCREngine, EngineResult
from ..config import FLORENCE_MODEL_ID, HF_TOKEN

logger = logging.getLogger(__name__)

class FlorenceEngine(BaseOCREngine):
    """Florence-2 VLM engine."""

    name = "florence"
    supports_bounding_boxes = True # Florence can return boxes, but we'll focus on text for now

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoProcessor
            from huggingface_hub import login
            
            if HF_TOKEN:
                login(token=HF_TOKEN)
            
            self.logger.info(f"Loading Florence-2 model: {FLORENCE_MODEL_ID}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                FLORENCE_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float32 # CPU friendly
            )
            # Fix for 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
            # Check both main config and text_config if it exists
            configs_to_check = [self._model.config]
            if hasattr(self._model.config, "text_config"):
                configs_to_check.append(self._model.config.text_config)
            
            for cfg in configs_to_check:
                if not hasattr(cfg, "forced_bos_token_id"):
                    try:
                        setattr(cfg, "forced_bos_token_id", None)
                    except Exception:
                        pass
            
            self._model = self._model.eval()
            self._processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_ID, trust_remote_code=True)

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()
        
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        # Florence-2 specific task
        task_prompt = "<OCR>" 

        for i, img_path in enumerate(image_paths, start=1):
            try:
                image = Image.open(img_path).convert("RGB")
                
                inputs = self._processor(text=task_prompt, images=image, return_tensors="pt")
                
                generated_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
                
                generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self._processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
                
                # Florence returns a dict like {'<OCR>': 'text...'}
                text = parsed_answer.get(task_prompt, "")
                page_texts.append(text)
                
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            engine_version="florence-2",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id=FLORENCE_MODEL_ID,
        )
