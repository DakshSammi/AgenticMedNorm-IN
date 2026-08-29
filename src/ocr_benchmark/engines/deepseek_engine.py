"""
DeepSeek-VL engine wrapper.
Optimized for lightweight vision tasks.
"""

import logging
import time
from pathlib import Path
from PIL import Image
import torch

from .base_engine import BaseOCREngine, EngineResult
from ..config import DEEPSEEK_VL_MODEL_ID, HF_TOKEN

logger = logging.getLogger(__name__)

class DeepSeekVLEngine(BaseOCREngine):
    """DeepSeek-VL engine."""

    name = "deepseek_vl"
    supports_bounding_boxes = False

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._tokenizer = None

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
            
            self.logger.info(f"Loading DeepSeek-VL model: {DEEPSEEK_VL_MODEL_ID}...")
            # For CPU, we avoid half-precision if it's not supported, but stay with float32/bfloat16
            self._model = AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_VL_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self._processor = AutoProcessor.from_pretrained(DEEPSEEK_VL_MODEL_ID, trust_remote_code=True)

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()
        
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        for i, img_path in enumerate(image_paths, start=1):
            try:
                image = Image.open(img_path).convert("RGB")
                
                # DeepSeek-VL prompt format
                prompt = "<image_placeholder>Transcribe ALL text from this medical prescription image exactly as written. Just raw text."
                
                # This is a simplified version of the DeepSeek-VL inference
                # DeepSeek-VL usually requires specific conversation templates
                from transformers import DynamicCache
                
                inputs = self._processor(prompt=prompt, images=[image], return_tensors="pt").to(self._model.device)
                
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )
                
                output_text = self._processor.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                page_texts.append(output_text)
                
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            engine_version="deepseek-vl",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id=DEEPSEEK_VL_MODEL_ID,
        )
