"""
Bodhi Resolver: Indian-context semantic mapping and terminology harmonization.
"""

import logging
from ..utils.llm_service import LLMService
from .bioportal_client import BioPortalClient # To be implemented

logger = logging.getLogger(__name__)

class BodhiResolver:
    """
    Standardizes biomedical and healthcare terminology across Indian healthcare ecosystems.
    Handles Indian-specific shorthand, medication variations, and multilingual clinical concepts.
    """
    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or LLMService()
        self.local_dictionary = {
            # Indian clinical abbreviations
            "ra": "review after",
            "sos": "as needed (si opus sit)",
            "e/d": "eye drops",
            "tds": "three times daily",
            "bd": "twice daily",
            "od": "once daily",
            "pc": "after meals",
            "ac": "before meals",
            "bbf": "before breakfast",
            "hs": "at bedtime",
        }

    def resolve_indian_context(self, term: str) -> dict:
        """
        Resolves terms specific to Indian medical context.
        Uses LLM for complex semantic disambiguation if needed.
        """
        term_lower = term.lower().strip()
        
        # 1. Check local Indian-context dictionary
        if term_lower in self.local_dictionary:
            return {
                "normalized": self.local_dictionary[term_lower],
                "context": "Indian Healthcare Shorthand",
                "source": "Bodhi"
            }

        # 2. Use LLM for Indian context resolution
        prompt = (
            "You are a medical informatics expert specialized in the Indian healthcare system. "
            f"Interpret the following clinical shorthand or term in an Indian context: '{term}'. "
            "Expand abbreviations, clarify local medication naming variations, and harmonize the terminology. "
            "Return a JSON with 'normalized' and 'context_notes'.\n\n"
            "Result:"
        )
        
        response = self.llm.call_gemini(prompt)
        try:
            import json
            # Heuristic for JSON extraction
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                data = json.loads(response[start:end])
                data["source"] = "Bodhi-LLM"
                return data
        except:
            pass

        return {"normalized": term, "source": "None"}

    def harmonize_terminology(self, normalized_entities: list[dict]) -> list[dict]:
        """Apply Bodhi harmonization over a list of normalized entities."""
        harmonized = []
        for ent in normalized_entities:
            resolution = self.resolve_indian_context(ent.get("normalized", ent.get("raw", "")))
            ent["bodhi_normalized"] = resolution.get("normalized")
            ent["indian_context"] = resolution.get("context", resolution.get("context_notes", ""))
            harmonized.append(ent)
        return harmonized
