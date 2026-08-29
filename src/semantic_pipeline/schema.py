"""
Standardised schema for semantic enrichment outputs.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class NormalizedEntity(BaseModel):
    raw: str
    normalized: str
    entity_type: str
    confidence: float = 1.0
    source_field: Optional[str] = None  # Which field in OCROutput it came from

class OntologyMapping(BaseModel):
    normalized_term: str
    ontology_name: str
    ontology_id: str
    mapping_confidence: float = 1.0
    iri: Optional[str] = None

class EnrichedPrescription(BaseModel):
    document_id: str
    ocr_engine: str
    raw_text: str
    normalized_entities: List[NormalizedEntity] = Field(default_factory=list)
    ontology_mappings: List[OntologyMapping] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()
