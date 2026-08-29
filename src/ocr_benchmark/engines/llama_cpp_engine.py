"""
Llama.cpp VLM engine wrapper.

Uses llama-cpp-python for local GGUF models with vision support.
Requires a model GGUF and a CLIP mmproj GGUF.
"""

import logging
import base64
import time
from pathlib import Path

from .base_engine import BaseOCREngine, EngineResult
from ..config import LLAMA_CPP_MODEL_PATH, LLAMA_CPP_CLIP_PATH

logger = logging.getLogger(__name__)

class LlamaCppEngine(BaseOCREngine):
    """Local Llama.cpp VLM engine."""

    name = "llama_cpp"
    supports_bounding_boxes = False

    def __init__(self) -> None:
        super().__init__()
        self._llava = None

    def is_available(self) -> bool:
        if not LLAMA_CPP_MODEL_PATH:
            return False
        try:
            # pyrefly: ignore [missing-import]
            from llama_cpp import Llama  # noqa: F401
            # pyrefly: ignore [missing-import]
            from llama_cpp.llama_chat_format import Llava15ChatHandler # noqa: F401
            return True
        except ImportError:
            self.logger.warning("llama-cpp-python not installed.")
            return False

    def _get_llava(self):
        if self._llava is None:
            # pyrefly: ignore [missing-import]
            from llama_cpp import Llama
            # pyrefly: ignore [missing-import]
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            
            chat_handler = Llava15ChatHandler(clip_model_path=LLAMA_CPP_CLIP_PATH)
            self._llava = Llama(
                model_path=LLAMA_CPP_MODEL_PATH,
                chat_handler=chat_handler,
                n_ctx=2048, # Increased context for images
                logits_all=True,
                n_gpu_layers=-1 # Use all GPU layers if available
            )
        return self._llava

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        llava = self._get_llava()
        page_texts: list[str] = []
        errors: list[str] = []
        start_time = time.time()

        for i, img_path in enumerate(image_paths, start=1):
            try:
                with open(img_path, "rb") as f:
                    img_base64 = base64.b64encode(f.read()).decode("utf-8")
                
                data_url = f"data:image/jpeg;base64,{img_base64}"
                
                response = llava.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a medical OCR specialist."},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Transcribe ALL text from this medical prescription image exactly as written. Just raw text."},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ]
                )
                
                text = response["choices"][0]["message"]["content"]
                page_texts.append(text)
                
            except Exception as exc:
                err = f"Page {i} ({img_path.name}): {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            engine_version="llama-cpp-python",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
            model_id=Path(LLAMA_CPP_MODEL_PATH).name,
        )
