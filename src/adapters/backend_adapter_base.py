import time
from typing import Dict, Any, Optional
from PIL import Image

class BaseBackendAdapter:
    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    @property
    def supports_image_input(self) -> bool:
        return True

    @property
    def supports_text_input(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return False

    async def run(self, prompt: str, image: Optional[Image.Image] = None, **kwargs) -> Dict[str, Any]:
        """
        Executes inference. Returns dict containing:
        - content: raw generated text string
        - processing_time_ms: latency of model execution
        - model_name: exact model name
        - backend_name: name of this backend
        - decoding_parameters: dict of settings used (temp, etc)
        """
        raise NotImplementedError
