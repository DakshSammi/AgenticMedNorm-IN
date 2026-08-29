"""
Unified LLM service for semantic disambiguation, normalization, and reranking.
"""

import logging
import requests
from ..config import GEMINI_API_KEY, HUGGINGFACE_API_KEY

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.gemini_key = GEMINI_API_KEY
        self.hf_key = HUGGINGFACE_API_KEY
        self._gemini_model = None

    def _get_gemini_model(self):
        if self._gemini_model is None and self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self._gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except ImportError:
                logger.warning("google-generativeai not installed.")
        return self._gemini_model

    def call_gemini(self, prompt: str) -> str:
        model = self._get_gemini_model()
        if not model:
            return ""
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return ""

    def call_huggingface(self, model_id: str, payload: dict) -> dict:
        """Calls HuggingFace Inference API."""
        if not self.hf_key:
            return {}
        
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {self.hf_key}"}
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"HuggingFace API call failed for {model_id}: {e}")
            return {}

    def normalize_clinical_text(self, raw_text: str) -> str:
        """Uses Gemini to normalize noisy clinical text and expand abbreviations."""
        prompt = (
            "You are a medical informatics expert. "
            "Normalize the following noisy OCR text from an Indian medical prescription. "
            "Expand abbreviations (e.g., TDS -> three times daily, OD -> once daily). "
            "Correct OCR typos while maintaining clinical meaning. "
            "Return ONLY the normalized text.\n\n"
            f"Raw Text: {raw_text}\n"
            "Normalized Text:"
        )
        return self.call_gemini(prompt)

    def rerank_ontology_candidates(self, term: str, candidates: list[dict]) -> dict:
        """Uses LLM to pick the best ontology candidate based on context."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        candidate_str = "\n".join([f"{i+1}. {c['term']} (Ontology: {c['ontology_name']}, ID: {c['ontology_id']})" for i, c in enumerate(candidates)])
        
        prompt = (
            f"For the clinical term '{term}', which of the following ontology mappings is the most accurate?\n"
            f"{candidate_str}\n\n"
            "Return only the number of the best candidate."
        )
        
        choice = self.call_gemini(prompt)
        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except:
            pass
        return candidates[0]
