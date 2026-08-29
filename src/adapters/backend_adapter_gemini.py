import time
import os
import logging
from typing import Dict, Any, Optional
from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter

try:
    from google import genai
except ImportError:
    genai = None

class GeminiBackendAdapter(BaseBackendAdapter):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="gemini", model_id="gemini-2.5-flash")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.logger = logging.getLogger("GeminiBackendAdapter")
        self.client = None
        if genai and self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    @property
    def supports_structured_output(self) -> bool:
        return True

    async def run(self, prompt: str, image: Optional[Image.Image] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_output_tokens": kwargs.get("max_tokens", 4096)
        }
        
        if not self.client:
            return {
                "error": "google-genai library or GOOGLE_API_KEY is not configured",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
            
        contents = [prompt]
        if image:
            contents.append(image)
            
        try:
            # Running client call in thread pool to prevent blocking event loop
            import asyncio
            loop = asyncio.get_event_loop()
            
            def make_call():
                return self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents
                )
                
            response = await loop.run_in_executor(None, make_call)
            processing_time_ms = (time.time() - start_time) * 1000
            
            return {
                "content": response.text,
                "processing_time_ms": processing_time_ms,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
        except Exception as e:
            self.logger.error(f"Gemini API request failed: {str(e)}")
            return {
                "error": str(e),
                "content": "",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }
