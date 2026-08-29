from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceStatus(StrEnum):
    MATCH = "MATCH"
    CONFLICT = "CONFLICT"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNKNOWN = "UNKNOWN"


class IngredientComponent(BaseModel):
    ingredient_id: str | None = None
    raw_name: str | None = None
    canonical_name: str | None = None
    strength_value: str | None = None
    strength_unit: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    order: int | None = None


class IngredientSet(BaseModel):
    components: list[IngredientComponent] = Field(default_factory=list)


class LexicalEvidence(BaseModel):
    evidence_id: str
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    observed_text: str | None = None
    candidate_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticEvidence(BaseModel):
    evidence_id: str
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)


class FormulationEvidence(BaseModel):
    evidence_id: str
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    ingredient_set: IngredientSet | None = None
    component_count: int | None = None
    dosage_form: str | None = None
    release_modifier: str | None = None
    fdc_structure: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextEvidence(BaseModel):
    evidence_id: str
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    implemented: bool = False
    neighboring_document_entities: list[dict[str, Any]] = Field(default_factory=list)
    possible_graph_concepts: list[dict[str, Any]] = Field(default_factory=list)
    relationship_evidence: list[dict[str, Any]] = Field(default_factory=list)


class ProvenanceEvidence(BaseModel):
    evidence_id: str
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    source_ids: list[str] = Field(default_factory=list)
    provenance_metadata: dict[str, Any] = Field(default_factory=dict)
