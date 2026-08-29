"""
Wrapper for scispaCy biomedical NER.
"""

import logging
import spacy
from ..config import SCISPACY_MODEL

logger = logging.getLogger(__name__)

class SciSpacyExtractor:
    def __init__(self, model_name: str = SCISPACY_MODEL):
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded scispaCy model: {model_name}")
        except OSError:
            logger.warning(f"scispaCy model {model_name} not found. Attempting to download...")
            import subprocess
            subprocess.run(["python", "-m", "pip", "install", f"https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz"])
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> list[dict]:
        if not text:
            return []
            
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "raw": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0  # spacy-sci doesn't provide confidence by default
            })
        return entities
