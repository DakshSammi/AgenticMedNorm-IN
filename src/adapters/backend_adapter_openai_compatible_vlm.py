import os
import time
import base64
import io
import re
import json
import logging
import aiohttp
from typing import Dict, Any, Optional, List, Union
from PIL import Image
from src.adapters.backend_adapter_base import BaseBackendAdapter
from src.schemas.raw_extraction import CanonicalRawDoc

class OpenAICompatibleVLMBackendAdapter(BaseBackendAdapter):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        max_image_dim: int = 1024,
        jpeg_quality: int = 85,
        timeout: int = 300,
        stream: bool = True,
        api_key_env_var: str = "INTERNAL_QWEN3_API_KEY",
        base_url_env_var: str = "INTERNAL_QWEN3_BASE_URL",
        model_env_var: str = "INTERNAL_QWEN3_MODEL"
    ):
        resolved_base_url = base_url or os.getenv(base_url_env_var) or "http://127.0.0.1:4000/v1"
        resolved_model = model_id or os.getenv(model_env_var) or "qwen3-27b"
        resolved_api_key = api_key or os.getenv(api_key_env_var)
        
        super().__init__(name="openai_compatible_vlm", model_id=resolved_model)
        
        # Clean trailing slash from base_url if present
        if resolved_base_url.endswith("/"):
            resolved_base_url = resolved_base_url[:-1]
            
        self.base_url = resolved_base_url
        self.api_key = resolved_api_key
        self.max_image_dim = max_image_dim
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.stream = stream
        self.logger = logging.getLogger("OpenAICompatibleVLMBackendAdapter")
        
        if not self.api_key:
            self.logger.warning(f"API key environment variable '{api_key_env_var}' is not set!")

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
            self.logger.info(f"Resized image from {w}x{h} to {new_size[0]}x{new_size[1]}")

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def run(self, prompt: str, image: Optional[Union[Image.Image, List[Image.Image]]] = None, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        decoding_params = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 0.9)
        }
        
        if not self.api_key:
            self.logger.error("API key is not configured for OpenAICompatibleVLMBackendAdapter.")
            return {
                "error": "API key is not configured",
                "content": "",
                "processing_time_ms": 0.0,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }

        # Build messages payload
        messages = []
        user_content = []
        
        # Text prompt
        user_content.append({"type": "text", "text": prompt})
        
        # Handle images
        if image:
            images_list = image if isinstance(image, list) else [image]
            self.logger.info(f"Encoding {len(images_list)} images for input.")
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
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        self.logger.info(f"Querying OpenAI-compatible VLM endpoint '{url}' with model '{self.model_id}'...")
        
        payload["stream"] = self.stream
        
        try:
            accumulated_content = []
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        if not self.stream:
                            response_json = await response.json()
                            latency_ms = (time.time() - start_time) * 1000
                            choices = response_json.get("choices", [])
                            content = ""
                            if choices:
                                content = ((choices[0].get("message") or {}).get("content") or "")
                            usage = response_json.get("usage") or {}
                            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)
                            self.logger.info(f"Inference complete in {latency_ms:.2f} ms with status 200")
                            return {
                                "content": content,
                                "processing_time_ms": latency_ms,
                                "model_name": self.model_id,
                                "backend_name": self.name,
                                "decoding_parameters": decoding_params,
                                "usage": {
                                    "input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "total_tokens": total_tokens
                                },
                                "raw_response": response_json
                            }

                        async for line in response.content:
                            line_str = line.decode("utf-8").strip()
                            if not line_str:
                                continue
                            if line_str.startswith("data: "):
                                data_content = line_str[6:]
                                if data_content == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_content)
                                    choices = chunk_json.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        if "content" in delta and delta["content"]:
                                            accumulated_content.append(delta["content"])
                                    if "usage" in chunk_json and chunk_json["usage"]:
                                        usage = chunk_json["usage"]
                                        input_tokens = usage.get("prompt_tokens", 0)
                                        output_tokens = usage.get("completion_tokens", 0)
                                        total_tokens = usage.get("total_tokens", 0)
                                except Exception:
                                    pass
                                    
                        latency_ms = (time.time() - start_time) * 1000
                        content = "".join(accumulated_content)
                        self.logger.info(f"Inference complete in {latency_ms:.2f} ms with status 200")
                        
                        return {
                            "content": content,
                            "processing_time_ms": latency_ms,
                            "model_name": self.model_id,
                            "backend_name": self.name,
                            "decoding_parameters": decoding_params,
                            "usage": {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": total_tokens
                            },
                            "raw_response": {
                                "choices": [{"message": {"role": "assistant", "content": content}}],
                                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": total_tokens}
                            }
                        }
                    else:
                        err_text = await response.text()
                        self.logger.error(f"HTTP Error {response.status}: {err_text}")
                        raise Exception(f"HTTP {response.status}: {err_text}")
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Inference failed: {str(e)}")
            return {
                "error": str(e),
                "content": "",
                "processing_time_ms": latency_ms,
                "model_name": self.model_id,
                "backend_name": self.name,
                "decoding_parameters": decoding_params
            }

    def clean_and_repair_json(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            return text
        json_block = text[start_idx:end_idx + 1]
        json_block = re.sub(r",\s*([\]}])", r"\1", json_block)
        return json_block

    def prune_placeholder_items(self, parsed_dict: dict, warnings: Optional[list] = None) -> dict:
        if not isinstance(parsed_dict, dict):
            return parsed_dict
        if warnings is None:
            warnings = []
        def is_empty(val):
            if val is None:
                return True
            if isinstance(val, str):
                return val.strip().lower() in ("", "null", "none", "[]", "{}")
            return False

        entity_keys = ["complaints_or_diagnosis", "observations", "procedures", "advice", "allergy_mentions", "other_notes"]
        for k in entity_keys:
            if k in parsed_dict and isinstance(parsed_dict[k], list):
                cleaned = []
                for idx, item in enumerate(parsed_dict[k]):
                    if isinstance(item, dict):
                        if not is_empty(item.get("raw_text")):
                            cleaned.append(item)
                        elif not is_empty(item.get("evidence_text")):
                            item["raw_text"] = item["evidence_text"]
                            cleaned.append(item)
                            warnings.append(f"{k}[{idx}]: raw_text was empty; populated from evidence_text")
                        else:
                            warnings.append(f"{k}[{idx}]: pruned empty RawEntityItem")
                    elif isinstance(item, str) and not is_empty(item):
                        cleaned.append({"raw_text": item, "evidence_text": item, "page_number": 1})
                        warnings.append(f"{k}[{idx}]: converted raw string to RawEntityItem dict")
                parsed_dict[k] = cleaned

        if "medications" in parsed_dict and isinstance(parsed_dict["medications"], list):
            cleaned_meds = []
            for idx, item in enumerate(parsed_dict["medications"]):
                if isinstance(item, dict):
                    raw_line = item.get("raw_line_text")
                    raw_name = item.get("raw_name")
                    if is_empty(raw_line) and not is_empty(raw_name):
                        item["raw_line_text"] = raw_name
                        raw_line = raw_name
                        warnings.append(f"medications[{idx}]: raw_line_text empty; set to raw_name")
                    if not is_empty(raw_line):
                        cleaned_meds.append(item)
                    else:
                        warnings.append(f"medications[{idx}]: pruned empty RawMedicationItem")
                elif isinstance(item, str) and not is_empty(item):
                    cleaned_meds.append({"raw_line_text": item, "raw_name": item, "evidence_text": item, "page_number": 1})
                    warnings.append(f"medications[{idx}]: converted string to RawMedicationItem dict")
            parsed_dict["medications"] = cleaned_meds

        if "lab_observations" in parsed_dict and isinstance(parsed_dict["lab_observations"], list):
            cleaned_labs = []
            for idx, item in enumerate(parsed_dict["lab_observations"]):
                if isinstance(item, dict):
                    test_name = item.get("test_name")
                    result = item.get("result")
                    if not is_empty(test_name) or not is_empty(result):
                        if is_empty(item.get("raw_line_text")):
                            parts = [x for x in [test_name, result, item.get("unit")] if x]
                            item["raw_line_text"] = " ".join(parts) if parts else "Lab Observation"
                            warnings.append(f"lab_observations[{idx}]: raw_line_text empty; constructed from parts")
                        cleaned_labs.append(item)
                    else:
                        warnings.append(f"lab_observations[{idx}]: pruned empty RawLabObservationItem")
                elif isinstance(item, str) and not is_empty(item):
                    cleaned_labs.append({"raw_line_text": item, "test_name": item, "evidence_text": item, "page_number": 1})
                    warnings.append(f"lab_observations[{idx}]: converted string to RawLabObservationItem dict")
            parsed_dict["lab_observations"] = cleaned_labs

        if "follow_up" in parsed_dict:
            fup = parsed_dict["follow_up"]
            if isinstance(fup, dict):
                if is_empty(fup.get("raw_text")) and is_empty(fup.get("date")) and is_empty(fup.get("review_after")):
                    parsed_dict["follow_up"] = None
                    warnings.append("follow_up: pruned empty RawFollowUp")
            elif is_empty(fup):
                parsed_dict["follow_up"] = None
                warnings.append("follow_up: pruned empty follow_up element")

        return parsed_dict

    def validate_and_parse_doc(self, raw_content: str, document_id: str) -> Dict[str, Any]:
        warnings = []
        coercions = []
        repaired_json = self.clean_and_repair_json(raw_content)
        
        try:
            parsed_dict = json.loads(repaired_json)
        except Exception as je:
            return {"error": f"JSON parse error: {str(je)}", "valid": False}
            
        parsed_dict = self.prune_placeholder_items(parsed_dict, warnings=warnings)
        
        # Coercions
        if isinstance(parsed_dict, dict) and "patient_information" in parsed_dict and isinstance(parsed_dict["patient_information"], dict):
            pinfo = parsed_dict["patient_information"]
            for k in ["name", "age", "gender", "address", "phone", "patient_identifier", "abha_id"]:
                if k in pinfo and pinfo[k] is not None and not isinstance(pinfo[k], str):
                    old = pinfo[k]
                    pinfo[k] = str(old)
                    coercions.append(f"patient_information.{k}: coerced {type(old).__name__} to string")
                    
        if isinstance(parsed_dict, dict) and "encounter_information" in parsed_dict and isinstance(parsed_dict["encounter_information"], dict):
            einfo = parsed_dict["encounter_information"]
            for k in ["date", "department", "hospital_name", "doctor_name", "visit_type", "fees", "room_or_queue_no"]:
                if k in einfo and einfo[k] is not None and not isinstance(einfo[k], str):
                    old = einfo[k]
                    einfo[k] = str(old)
                    coercions.append(f"encounter_information.{k}: coerced {type(old).__name__} to string")

        try:
            parsed_dict["document_id"] = document_id
            parsed_dict["schema_version"] = "raw_rx_v2"
            doc = CanonicalRawDoc(**parsed_dict)
            return {
                "valid": True,
                "doc": doc,
                "warnings": warnings,
                "coercions": coercions
            }
        except Exception as ve:
            return {
                "error": f"Pydantic validation error: {str(ve)}",
                "valid": False,
                "warnings": warnings,
                "coercions": coercions
            }
