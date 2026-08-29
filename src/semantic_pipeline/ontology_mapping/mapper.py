"""
Ontology Mapping Orchestrator with Bodhi and LLM Integration.
"""

import logging
from .aberowl_client import AberOWLClient
from .bioportal_client import BioPortalClient
from .bodhi_resolver import BodhiResolver
from ..utils.llm_service import LLMService

logger = logging.getLogger(__name__)

class OntologyMapper:
    def __init__(self):
        self.llm = LLMService()
        self.aberowl = AberOWLClient()
        self.bioportal = BioPortalClient()
        self.bodhi = BodhiResolver(llm_service=self.llm)

    def map_term_with_enrichment(self, term: str) -> list[dict]:
        """
        Maps a term to multiple ontologies and enriches it with Indian context (Bodhi).
        Workflow:
        1. Standard Mapping (AberOWL / BioPortal).
        2. Bodhi Enrichment.
        3. LLM Reranking.
        """
        if not term:
            return []
            
        all_candidates = []
        
        # 1. Standard Ontology Mapping
        # AberOWL (SNOMED, RxNorm)
        all_candidates.extend(self.aberowl.search_term(term))
        
        # BioPortal (ICD-10, UMLS, MeSH)
        all_candidates.extend(self.bioportal.search_term(term, ontologies=["ICD10", "UMLS", "MESH"]))
        
        # 2. LLM Reranking of standard candidates
        best_candidate = self.llm.rerank_ontology_candidates(term, all_candidates)
        
        # 3. Bodhi Indian-Context Enrichment
        bodhi_result = self.bodhi.resolve_indian_context(term)
        
        final_results = []
        if best_candidate:
            best_candidate["indian_context_enrichment"] = bodhi_result
            final_results.append(best_candidate)
        else:
            # If no ontology mapping found, still provide Bodhi enrichment
            final_results.append({
                "normalized_term": term,
                "ontology_name": "None",
                "indian_context_enrichment": bodhi_result
            })
            
        return final_results

    def enrich_json(self, ocr_json: dict) -> dict:
        """Enriches the OCR JSON with ontology mappings and Bodhi context."""
        # 1. Process extracted semantic entities (if any from NER stage)
        if "semantic_entities" in ocr_json:
            enriched_entities = []
            for ent in ocr_json["semantic_entities"]:
                term = ent.get("normalized", ent.get("raw", ""))
                mappings = self.map_term_with_enrichment(term)
                ent["ontology_mappings"] = mappings
                enriched_entities.append(ent)
            ocr_json["semantic_entities"] = enriched_entities
            
        # 2. Also process the structured fields in RawEntities
        if "raw_entities" in ocr_json:
            entities = ocr_json["raw_entities"]
            # Process medications
            if "medications" in entities:
                for med in entities["medications"]:
                    term = med.get("normalized_medication_text", med.get("raw_medication_text", ""))
                    med["ontology_mappings"] = self.map_term_with_enrichment(term)
                    
            # Process conditions/diagnosis
            for field in ["normalized_complaints_or_diagnosis", "normalized_observations"]:
                if field in entities:
                    mapped_field = field.replace("normalized_", "mapped_")
                    entities[mapped_field] = []
                    for term in entities[field]:
                        entities[mapped_field].append({
                            "term": term,
                            "mappings": self.map_term_with_enrichment(term)
                        })
                        
        return ocr_json
