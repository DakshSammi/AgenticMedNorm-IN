"""
Normalization orchestrator.
"""

import logging
from .rule_based import RuleBasedNormalizer
from ..config import USE_LLM_NORMALIZATION

logger = logging.getLogger(__name__)

class SemanticNormalizer:
    def __init__(self):
        self.rule_based = RuleBasedNormalizer()
        self.llm_normalizer = None
        
        if USE_LLM_NORMALIZATION:
            try:
                # Lazy import for LLM components
                from .llm_normalizer import LLMNormalizer
                self.llm_normalizer = LLMNormalizer()
            except ImportError:
                logger.warning("LLMNormalizer requested but dependencies not met. Falling back to rule-based.")

    def normalize_text(self, text: str) -> str:
        """Normalizes a single string."""
        # Rule-based normalization is always the first step
        norm = self.rule_based.normalize(text)
        
        # LLM can further refine if enabled
        if self.llm_normalizer:
            norm = self.llm_normalizer.refine(norm)
            
        return norm

    def process_ocr_output(self, ocr_json: dict) -> dict:
        """
        Processes a full OCR JSON output and returns a normalized version.
        Maintains both raw and normalized values.
        """
        normalized_data = ocr_json.copy()
        
        # Normalize entities in RawEntities section
        if "raw_entities" in ocr_json:
            entities = ocr_json["raw_entities"]
            
            # Normalize medications
            if "medications" in entities:
                normalized_data["raw_entities"]["medications"] = [
                    self.rule_based.normalize_medication(m) for m in entities["medications"]
                ]
            
            # Normalize other flat fields
            for field in ["patient_information", "encounter_information"]:
                if field in entities:
                    for key, val in entities[field].items():
                        if isinstance(val, str) and val:
                            normalized_data["raw_entities"][field][f"normalized_{key}"] = self.normalize_text(val)
                            
            # Normalize list fields (complaints, observations, etc.)
            for field in ["complaints_or_diagnosis", "observations", "procedures"]:
                if field in entities:
                    normalized_data["raw_entities"][f"normalized_{field}"] = [
                        self.normalize_text(item) if isinstance(item, str) else item for item in entities[field]
                    ]
                    
        return normalized_data
