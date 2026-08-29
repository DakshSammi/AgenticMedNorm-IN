from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW", "UNREADABLE"]
ReviewReasonCode = Literal[
    "IMAGE_UNCLEAR",
    "MEDICATION_TEXT_UNCERTAIN",
    "STRENGTH_UNCERTAIN",
    "HANDWRITING_AMBIGUOUS",
    "SCHEMA_FAILURE",
    "OTHER"
]


class DocumentMetadata(BaseModel):
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    patient_weight: Optional[str] = None
    prescriber_name: Optional[str] = None
    prescriber_qualification: Optional[str] = None
    prescriber_registration: Optional[str] = None
    clinic_hospital: Optional[str] = None
    prescription_date: Optional[str] = None
    prescription_number: Optional[str] = None


class MedicationEntry(BaseModel):
    raw_medication_text: str
    raw_strength: Optional[str] = None
    raw_dosage_form: Optional[str] = None
    raw_route: Optional[str] = None
    raw_frequency: Optional[str] = None
    raw_duration: Optional[str] = None
    raw_timing: Optional[str] = None
    raw_instructions: Optional[str] = None
    confidence: ConfidenceLevel
    review_reason_codes: list[ReviewReasonCode] = Field(default_factory=list)


class NonMedicationFields(BaseModel):
    diagnosis: Optional[str] = None
    investigations: Optional[str] = None
    follow_up: Optional[str] = None
    general_advice: Optional[str] = None


class AnnotationOutput(BaseModel):
    document_metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    medications: list[MedicationEntry] = Field(default_factory=list)
    non_medication: NonMedicationFields = Field(default_factory=NonMedicationFields)
    page_level_confidence: ConfidenceLevel = "HIGH"
    page_review_reason_codes: list[ReviewReasonCode] = Field(default_factory=list)


class AnnotationArtifact(BaseModel):
    artifact_id: str
    page_uid: str
    inference_group_id: str
    document_uid: Optional[str] = None
    page_number: Optional[int] = None
    collection_date: str
    source_type: str
    deidentified_image_path: str
    deidentified_sha256: str
    annotation: AnnotationOutput
    annotation_status: Literal["SUCCESS", "NEEDS_REVIEW", "FAILED", "PENDING"]
    annotation_model: str
    reasoning_effort: str
    prompt_version: str
    prompt_sha256: str
    duplicate_group_id: Optional[str] = None
    canonical_inference_page_uid: Optional[str] = None
    derived_from_duplicate_representative: Optional[str] = None
    tool_version: str
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_code: Optional[str] = None
    provenance: Optional[Any] = None