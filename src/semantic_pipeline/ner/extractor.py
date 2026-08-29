"""
NER Orchestrator.
"""

import logging
from .scispacy_wrapper import SciSpacyExtractor

logger = logging.getLogger(__name__)

class BiomedicalNER:
    def __init__(self, backend: str = "scispacy"):
        self.backend = backend
        self.extractor = None
        
        if backend == "scispacy":
            self.extractor = SciSpacyExtractor()
        elif backend == "transformers":
            try:
                from .transformer_wrapper import TransformerExtractor
                self.extractor = TransformerExtractor()
            except ImportError:
                logger.error("Transformers requested but dependencies/wrapper not found.")
                self.extractor = SciSpacyExtractor()
        else:
            logger.warning(f"Unknown NER backend: {backend}. Defaulting to scispacy.")
            self.extractor = SciSpacyExtractor()

    def extract(self, text: str) -> list[dict]:
        """Extracts entities from raw text."""
        return self.extractor.extract_entities(text)

    def enrich_json(self, ocr_json: dict) -> dict:
        """Adds NER results to the OCR JSON."""
        full_text = ocr_json.get("raw_text", {}).get("full_text", "")
        if not full_text:
            return ocr_json
            
        entities = self.extract(full_text)
        ocr_json["semantic_entities"] = entities
        return ocr_json
