"""
Client for BioPortal API.
"""

import requests
import logging
from ..config import BIOPORTAL_API_KEY, BIOPORTAL_API_URL

logger = logging.getLogger(__name__)

class BioPortalClient:
    def __init__(self, api_key: str = BIOPORTAL_API_KEY):
        self.api_key = api_key
        self.base_url = BIOPORTAL_API_URL

    def search_term(self, term: str, ontologies: list[str] = None) -> list[dict]:
        """
        Search for a term in BioPortal.
        Example: http://data.bioontology.org/search?q=melanoma&ontologies=SNOMEDCT,ICD10
        """
        if not self.api_key:
            logger.warning("BioPortal API key not found. Skipping search.")
            return []

        params = {
            "q": term,
            "apikey": self.api_key,
            "display_context": "false"
        }
        if ontologies:
            params["ontologies"] = ",".join(ontologies)

        try:
            response = requests.get(f"{self.base_url}/search", params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("collection", [])
            
            formatted = []
            for item in results:
                formatted.append({
                    "term": item.get("prefLabel"),
                    "ontology_name": item.get("links", {}).get("ontology", "").split("/")[-1],
                    "ontology_id": item.get("@id"),
                    "mapping_confidence": 0.7 # BioPortal search relevance
                })
            return formatted
        except Exception as e:
            logger.error(f"BioPortal search failed for '{term}': {e}")
            return []
