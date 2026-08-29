"""
Wrapper for HuggingFace Transformers biomedical NER.
"""

import logging
from transformers import pipeline
from ..config import TRANSFORMER_BIOMEDICAL_MODEL, HUGGINGFACE_API_KEY

logger = logging.getLogger(__name__)

class TransformerExtractor:
    def __init__(self, model_name: str = TRANSFORMER_BIOMEDICAL_MODEL):
        self.model_name = model_name
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            logger.info(f"Loading Transformers NER pipeline: {self.model_name}...")
            # Use 'aggregation_strategy' to merge subword tokens into entities
            self._pipe = pipeline(
                "ner", 
                model=self.model_name, 
                aggregation_strategy="simple",
                # device_map="auto" # Uncomment if GPU is available and desired
            )
        return self._pipe

    def extract_entities(self, text: str) -> list[dict]:
        if not text:
            return []
            
        try:
            results = self.pipe(text)
            entities = []
            for res in results:
                entities.append({
                    "raw": res["word"],
                    "label": res["entity_group"],
                    "start": res["start"],
                    "end": res["end"],
                    "confidence": float(res["score"])
                })
            return entities
        except Exception as e:
            logger.error(f"Transformers NER extraction failed: {e}")
            return []
