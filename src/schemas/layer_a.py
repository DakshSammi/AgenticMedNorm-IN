from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.provenance import SourceFieldValue


LAYER_A_SCHEMA_VERSION = "layer_a_schema_v1"

LineageStatus = Literal[
    "VERIFIED_EXACT_METADATA",
    "VERIFIED_VISUAL_HIGH",
    "VERIFIED_VISUAL_MEDIUM",
    "AMBIGUOUS",
    "UNVERIFIED_HEURISTIC",
    "UNMATCHED",
]

WorkStatus = Literal["PENDING", "RUNNING", "SUCCESS", "NEEDS_REVIEW", "FAILED", "SKIPPED_DUPLICATE", "BLOCKED"]


class CanonicalDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LAYER_A_SCHEMA_VERSION
    document_uid: str
    source_document_id: str
    collection_date: str
    source_json_relpath: str
    source_json_sha256: str
    total_pages: int | None = None
    language: list[str] = Field(default_factory=list)
    source_type: str | None = None
    duplicate_group_id: str | None = None
    canonical_duplicate_representative: str | None = None


class CanonicalPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LAYER_A_SCHEMA_VERSION
    page_uid: str
    document_uid: str
    page_number: int | None = None
    raw_image_relpath: str | None = None
    raw_image_sha256: str | None = None
    anonymized_image_relpath: str | None = None
    anonymized_image_sha256: str | None = None
    lineage_status: LineageStatus = "UNVERIFIED_HEURISTIC"
    deidentification_status: WorkStatus = "PENDING"
    annotation_status: WorkStatus = "SUCCESS"
    duplicate_group_id: str | None = None
    canonical_duplicate_representative: str | None = None
    evaluation_lineage_eligible: bool = False


class LayerAMedicationMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LAYER_A_SCHEMA_VERSION
    mention_id: str
    document_uid: str
    page_uid: str
    collection_date: str

    raw_medication_text: str | None = None
    lexical_surface_normalized: str | None = None

    raw_strength_text: str | None = None
    raw_dosage_text: str | None = None
    raw_frequency_text: str | None = None
    raw_duration_text: str | None = None
    raw_route_text: str | None = None
    raw_timing_text: str | None = None
    raw_instruction_text: str | None = None
    raw_notes: str | None = None

    source_json_path: str
    source_object_index: int
    source_json_relpath: str
    source_json_sha256: str

    annotation_model: str = "UNKNOWN_LEGACY_GENERATION"
    annotation_model_settings: dict[str, Any] | None = None
    annotation_prompt_version: str | None = None

    annotation_status: WorkStatus = "SUCCESS"
    annotation_review_status: str = "UNKNOWN"
    source_schema_variant: str

    context_bundle_id: str
    source_fields: list[SourceFieldValue] = Field(default_factory=list)
    deduplication_reason: str | None = None
    duplicate_source_paths: list[str] = Field(default_factory=list)


class DocumentContextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LAYER_A_SCHEMA_VERSION
    context_bundle_id: str
    document_uid: str
    page_uid: str | None = None
    chief_complaints: list[Any] = Field(default_factory=list)
    diagnoses: list[Any] = Field(default_factory=list)
    clinical_history: list[Any] = Field(default_factory=list)
    clinical_findings: list[Any] = Field(default_factory=list)
    investigations: list[Any] = Field(default_factory=list)
    lab_observations: list[Any] = Field(default_factory=list)
    procedures: list[Any] = Field(default_factory=list)
    advice: list[Any] = Field(default_factory=list)
    follow_up: list[Any] = Field(default_factory=list)
    other_context: list[Any] = Field(default_factory=list)
