"""Evidence-backed semantic extraction schema for Stage 1C."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SemanticType = Literal[
    "medication",
    "dosage",
    "frequency",
    "duration",
    "diagnosis",
    "complaint",
    "observation",
    "vital",
    "lab_result",
    "procedure",
    "advice",
    "follow_up",
]


class SemanticEntity(BaseModel):
    semantic_type: SemanticType
    normalized_name: str
    raw_evidence_text: str
    source_raw_field: str
    source_page_or_image: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    normalization_method: str
    evidence_supported: bool


class SemanticRelation(BaseModel):
    relation_type: str
    source_entity: str
    target_entity: str
    raw_evidence_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_supported: bool = False


class UnsupportedInference(BaseModel):
    inferred_claim: str
    reason: str
    raw_evidence_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SemanticMetadata(BaseModel):
    schema_version: str = "semantic_rx_v1"
    model_name: str
    backend_name: str = "ollama"
    source_stage1b_path: str
    processing_time_ms: float = 0.0
    timestamp: Optional[str] = None
    prompt_version: str = "stage1c_evidence_v1"
    extra: Dict[str, Any] = Field(default_factory=dict)


class SemanticExtractionDoc(BaseModel):
    document_id: str
    source_system: str
    semantic_entities: List[SemanticEntity] = Field(default_factory=list)
    semantic_relations: List[SemanticRelation] = Field(default_factory=list)
    unsupported_inferences: List[UnsupportedInference] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: SemanticMetadata

