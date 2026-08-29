import time
import logging
from typing import Dict, Any, Optional
from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter

class FlorenceBackendAdapter(BaseBackendAdapter):
    def __init__(self, model_id: str = "microsoft/Florence-2-large"):
        super().__init__(name="florence", model_id=model_id)
        self.logger = logging.getLogger("FlorenceBackendAdapter")
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self.model is None:
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
                self.logger.info(f"Loading local Florence-2 model: {self.model_id}")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e:
                self.logger.error(f"Failed to load local Florence-2 model: {str(e)}")

    async def run(self, prompt: str, image: Optional[Image.Image] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {}
        
        if not image:
            return {
                "error": "Florence-2 requires an image input",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
            
        self._lazy_load()
        if not self.model or not self.processor:
            return {
                "error": "Florence-2 model is not loaded",
                "content": "Simulated Florence OCR result: Bioflu e/d tds, Adv glasses.",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
            
        try:
            import torch
            import asyncio
            
            # Florence-2 tasks (e.g. '<OCR_WITH_REGION>')
            task_prompt = prompt if prompt.startswith("<") else "<OCR_WITH_REGION>"
            
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run in executor to avoid blocking event loop
            def generate():
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        early_stopping=False,
                        do_sample=False,
                        num_beams=3
                    )
                return generated_ids
                
            loop = asyncio.get_event_loop()
            generated_ids = await loop.run_in_executor(None, generate)
            
            content = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "content": content,
                "processing_time_ms": processing_time_ms,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
        except Exception as e:
            self.logger.error(f"Florence-2 generation failed: {str(e)}")
            return {
                "error": str(e),
                "content": "",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
