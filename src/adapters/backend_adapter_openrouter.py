import os
import time
import base64
import io
import logging
import aiohttp
from typing import Dict, Any, Optional, List, Union
from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter

class OpenRouterBackendAdapter(BaseBackendAdapter):
    def __init__(self, model_id: str, api_key: Optional[str] = None, max_image_dim: int = 1024):
        super().__init__(name="openrouter", model_id=model_id)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.logger = logging.getLogger("OpenRouterBackendAdapter")
        self.max_image_dim = max_image_dim
        if not self.api_key:
            self.logger.warning("OPENROUTER_API_KEY environment variable is not set!")

    @property
    def supports_structured_output(self) -> bool:
        return True

    def _encode_image(self, image: Image.Image) -> str:
        # Aspect-ratio-preserving in-memory image resizing to limit max dimension to max_image_dim
        if max(image.size) > self.max_image_dim:
            w, h = image.size
            scale = self.max_image_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def run(self, prompt: str, image: Optional[Union[Image.Image, List[Image.Image]]] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        if not self.api_key:
            return {
                "error": "OPENROUTER_API_KEY is not configured",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }

        # Build messages payload
        messages = []
        user_content = []
        
        # Text prompt is always included
        user_content.append({"type": "text", "text": prompt})
        
        # Handle images
        if image:
            images_list = image if isinstance(image, list) else [image]
            for img in images_list:
                base64_str = self._encode_image(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}"
                    }
                })
                
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        payload = {
            "model": self.model_id,
            "messages": messages,
            **decoding_params
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
            "X-Title": "Clinical Prescription Pipeline"
        }
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        self.logger.info(f"Querying OpenRouter model '{self.model_id}'...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=300) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    if response.status == 200:
                        resp_json = await response.json()
                        choices = resp_json.get("choices", [])
                        if not choices:
                            raise Exception(f"No choices returned in OpenRouter response: {resp_json}")
                            
                        content = choices[0]["message"]["content"]
                        usage = resp_json.get("usage", {})
                        
                        # Extract usage details
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        
                        # OpenRouter sometimes returns estimated cost in usage or choices, or we can check usage.get("cost", 0.0)
                        estimated_cost = resp_json.get("usage", {}).get("cost", 0.0)
                        if estimated_cost is None:
                            estimated_cost = 0.0
                            
                        return {
                            "content": content,
                            "processing_time_ms": latency_ms,
                            "model_name": self.model_id,
                            "backend_name": self.name,
                            "decoding_parameters": decoding_params,
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens,
                                "estimated_cost": estimated_cost
                            },
                            "raw_response": resp_json
                        }
                    else:
                        err_text = await response.text()
                        self.logger.error(f"OpenRouter HTTP Error {response.status}: {err_text}")
                        raise Exception(f"HTTP {response.status}: {err_text}")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"OpenRouter inference failed: {str(e)}")
            return {
                "error": str(e),
                "content": "",
                "processing_time_ms": latency_ms,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
