import io
import base64
import time
import aiohttp
import logging
from typing import Dict, Any, Optional, List, Union
from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter

class QwenVLBackendAdapter(BaseBackendAdapter):
    def __init__(self, endpoint_url: str = "http://localhost:8090/v1", model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        super().__init__(name="qwen25_vl_7b", model_id=model_id)
        self.endpoint_url = endpoint_url
        self.logger = logging.getLogger("QwenVLBackendAdapter")
        self._local_model = None
        self._local_processor = None

    @property
    def supports_structured_output(self) -> bool:
        return True

    def _encode_image(self, image: Image.Image) -> str:
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
        
        # Build images list and resize large images to prevent OOM and slow inference
        images_list = []
        if image:
            raw_list = image if isinstance(image, list) else [image]
            for img in raw_list:
                max_dim = 1024
                if max(img.size) > max_dim:
                    w, h = img.size
                    scale = max_dim / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                images_list.append(img)

        # Attempt A: Local vLLM OpenAI-compatible endpoint
        try:
            self.logger.info(f"Attempting inference via local vLLM at {self.endpoint_url}")
            
            # Prepare payload matching OpenAI chat completion spec
            messages = []
            user_content = [{"type": "text", "text": prompt}]
            
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
                "model": self.model_id, # model weight loaded in docker
                "messages": messages,
                **decoding_params
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.endpoint_url}/chat/completions", json=payload, timeout=120) as response:
                    if response.status == 200:
                        resp_json = await response.json()
                        content = resp_json["choices"][0]["message"]["content"]
                        processing_time_ms = (time.time() - start_time) * 1000
                        return {
                            "content": content,
                            "processing_time_ms": processing_time_ms,
                            "model_name": self.model_id,
                            "backend_name": self.name,
                            "decoding_parameters": decoding_params
                        }
                    else:
                        err_text = await response.text()
                        self.logger.warning(f"vLLM server returned error status {response.status}: {err_text}")
                        raise Exception(f"HTTP {response.status}: {err_text}")
                        
        except Exception as e:
            self.logger.warning(f"Local vLLM request failed: {e}. Falling back to local transformers loading...")
            
            # Attempt B: Local transformers fallback (Lazy initialized)
            try:
                import torch
                from transformers import AutoProcessor
                from qwen_vl_utils import process_vision_info
                
                # Import robust Qwen model class dynamically
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as QwenModelClass
                except ImportError:
                    try:
                        from transformers import Qwen2VLForConditionalGeneration as QwenModelClass
                    except ImportError:
                        from transformers import AutoModelForCausalLM as QwenModelClass
                
                loaded_and_run = False
                force_cpu = False
                content = ""
                
                while not loaded_and_run:
                    try:
                        if self._local_model is None:
                            self.logger.info(f"Lazy loading local HuggingFace model: {self.model_id}")
                            use_cuda = torch.cuda.is_available() and not force_cpu
                            
                            self._local_processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                            
                            if use_cuda:
                                try:
                                    self.logger.info("Attempting to load model on GPU...")
                                    self._local_model = QwenModelClass.from_pretrained(
                                        self.model_id, 
                                        torch_dtype=torch.bfloat16, 
                                        device_map="auto",
                                        trust_remote_code=True
                                    )
                                    # Log GPU IDs used
                                    device_map_info = getattr(self._local_model, "hf_device_map", {})
                                    self.logger.info(f"Successfully loaded model on GPU. Device Map: {device_map_info}")
                                except Exception as gpu_err:
                                    self.logger.warning(f"Failed to load model on GPU: {gpu_err}. Falling back to CPU...")
                                    use_cuda = False
                                    
                            if not use_cuda:
                                self.logger.info("Loading model on CPU...")
                                self._local_model = QwenModelClass.from_pretrained(
                                    self.model_id, 
                                    torch_dtype=torch.float32, 
                                    device_map="cpu",
                                    trust_remote_code=True
                                )
                                self.logger.info("Successfully loaded model on CPU.")
                            
                        # Prepare messages for local processor
                        messages = []
                        user_content = []
                        for img in images_list:
                            # In qwen-vl-utils, we can pass raw PIL images
                            user_content.append({"type": "image", "image": img})
                            
                        user_content.append({"type": "text", "text": prompt})
                        messages.append({"role": "user", "content": user_content})
                        
                        # Preprocess inputs
                        text = self._local_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self._local_processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(self._local_model.device)
                        
                        # Run inference
                        with torch.no_grad():
                            generated_ids = self._local_model.generate(**inputs, max_new_tokens=decoding_params["max_tokens"])
                            generated_ids_trimmed = [
                                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                            ]
                            content = self._local_processor.batch_decode(
                                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                            )[0]
                        
                        loaded_and_run = True
                        
                    except Exception as run_err:
                        if torch.cuda.is_available() and not force_cpu:
                            self.logger.warning(f"Failed during GPU model initialization or generation: {run_err}. Clearing GPU memory and forcing CPU fallback...")
                            # Clear GPU memory
                            self._local_model = None
                            self._local_processor = None
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            # Force CPU on next iteration
                            force_cpu = True
                        else:
                            raise run_err
                
                processing_time_ms = (time.time() - start_time) * 1000
                return {
                    "content": content,
                    "processing_time_ms": processing_time_ms,
                    "model_name": self.model_id,
                    "backend_name": self.name,
                    "decoding_parameters": decoding_params
                }
                
            except Exception as le:
                self.logger.error(f"Local HuggingFace fallback loader failed: {le}")
                # Final return with error details
                return {
                    "error": f"vLLM and HuggingFace loaders both failed. vLLM: {e}; HF: {le}",
                    "content": "",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "model_name": self.model_id,
                    "backend_name": self.name,
                    "decoding_parameters": decoding_params
                }
