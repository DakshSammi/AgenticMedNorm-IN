import time
import logging
from typing import Dict, Any, Optional
from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter

class DonutBackendAdapter(BaseBackendAdapter):
    def __init__(self, model_id: str = "naver-clova-ix/donut-base-finetuned-docvqa"):
        super().__init__(name="donut", model_id=model_id)
        self.logger = logging.getLogger("DonutBackendAdapter")
        self.model = None
        self.processor = None

    def _lazy_load(self):
        if self.model is None:
            try:
                import torch
                from transformers import DonutProcessor, VisionEncoderDecoderModel
                self.logger.info(f"Loading local Donut model: {self.model_id}")
                self.processor = DonutProcessor.from_pretrained(self.model_id)
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).eval()
            except Exception as e:
                self.logger.error(f"Failed to load local Donut model: {str(e)}")

    async def run(self, prompt: str, image: Optional[Image.Image] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {}
        
        if not image:
            return {
                "error": "Donut requires an image input",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
            
        self._lazy_load()
        if not self.model or not self.processor:
            return {
                "error": "Donut model is not loaded",
                "content": "Simulated Donut result: {\"items\": []}",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
            
        try:
            import torch
            import re
            import asyncio
            
            # Prepare Donut prompt
            task_name = "docvqa"
            task_prompt = f"<s_{task_name}><s_question>{prompt}</s_question><s_answer>"
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            
            def generate():
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.model.config.decoder.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True
                    )
                return outputs
                
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(None, generate)
            
            seq = self.processor.batch_decode(outputs.sequences)[0]
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            
            content = self.processor.token2json(seq)
            processing_time_ms = (time.time() - start_time) * 1000
            
            import json
            return {
                "content": json.dumps(content) if isinstance(content, dict) else str(content),
                "processing_time_ms": processing_time_ms,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
        except Exception as e:
            self.logger.error(f"Donut generation failed: {str(e)}")
            return {
                "error": str(e),
                "content": "",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
