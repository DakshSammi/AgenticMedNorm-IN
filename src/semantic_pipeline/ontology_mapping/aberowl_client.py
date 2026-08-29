"""
Client for AberOWL API.
"""

import requests
import logging
from ..config import ABEROWL_API_URL

logger = logging.getLogger(__name__)

class AberOWLClient:
    def __init__(self, base_url: str = ABEROWL_API_URL):
        self.base_url = base_url.rstrip('/')

    def search_term(self, term: str, ontology: str = None) -> list[dict]:
        """
        Search for a term in AberOWL.
        Example: http://aber-owl.net/api/class/_search?term=Diabetes&ontology=SNOMEDCT
        """
        endpoint = f"{self.base_url}/class/_search"
        params = {"term": term}
        if ontology:
            params["ontology"] = ontology
            
        try:
            response = requests.get(endpoint, params=params, timeout=5)
            response.raise_for_status()
            results = response.json()
            
            # Format results for internal use
            formatted = []
            for item in results:
                formatted.append({
                    "term": item.get("label", [term])[0],
                    "ontology_name": item.get("ontology", "Unknown"),
                    "ontology_id": item.get("oboid", ""),
                    "iri": item.get("iri", ""),
                    "mapping_confidence": 0.8  # Placeholder for ranking
                })
            return formatted
        except Exception as e:
            logger.error(f"AberOWL search failed for '{term}': {e}")
            return []
            
    def map_to_snomed(self, term: str) -> dict:
        results = self.search_term(term, ontology="SNOMEDCT")
        return results[0] if results else None
