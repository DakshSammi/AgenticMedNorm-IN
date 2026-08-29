"""
HuggingFace Transformers VLM engine wrapper.
Supports models like Qwen2-VL, Moondream, etc.
"""

import logging
import time
from pathlib import Path
from PIL import Image

from .base_engine import BaseOCREngine, EngineResult
from ..config import QWEN_VL_MODEL_ID

logger = logging.getLogger(__name__)

class TransformersVLMEngine(BaseOCREngine):
    """VLM engine using HuggingFace Transformers."""

    name = "qwen_vl"
    supports_bounding_boxes = False

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    def is_available(self) -> bool:
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            from huggingface_hub import login
            from ..config import HF_TOKEN
            
            if HF_TOKEN:
                login(token=HF_TOKEN)
            
            self.logger.info(f"Loading VLM model: {QWEN_VL_MODEL_ID}...")
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_VL_MODEL_ID, 
                torch_dtype="auto", 
                device_map="auto"
            )
            self._processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_ID)

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        self._load_model()
        import torch
        
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        for i, img_path in enumerate(image_paths, start=1):
            try:
                image = Image.open(img_path).convert("RGB")
                
                # Prepare prompt for Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Transcribe ALL text from this medical prescription image exactly as written. Just raw text."},
                        ],
                    }
                ]
                
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = None, None # process_vision_info not implemented here for simplicity
                
                inputs = self._processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(self._model.device)

                generated_ids = self._model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self._processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                page_texts.append(output_text[0])
                
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            engine_version="transformers:qwen2-vl",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id=QWEN_VL_MODEL_ID,
        )
