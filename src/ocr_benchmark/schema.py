"""
Standardised output schema for the OCR benchmarking pipeline.

This schema is a superset of all observed ground-truth schemas:
    - p1/p2/p3  : ophthalmology (observations, medications, procedures)
    - p25       : general medicine (vitals, clinical_history, neurological_exam)
    - p36       : endocrinology (diagnosis, lab_observations, instructions)

Rules:
    - All fields are Optional with sensible empty defaults (None / [] / "").
    - NO normalization, NO correction, NO ontology mapping.
    - OCR errors are preserved verbatim.
    - Missing fields → None or [] — never hallucinated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Normalised [0,1] or pixel coordinates depending on engine."""
    left: Optional[float] = None
    top: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    # Some engines return polygon points instead of rect
    polygon: Optional[list[list[float]]] = None


class WordCoordinate(BaseModel):
    text: str
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None
    page: int = 1


class LineCoordinate(BaseModel):
    text: str
    confidence: Optional[float] = None
    bounding_box: Optional[BoundingBox] = None
    page: int = 1


class OCRCoordinates(BaseModel):
    """Word- and line-level bounding boxes. Empty if engine doesn't support."""
    words: list[WordCoordinate] = Field(default_factory=list)
    lines: list[LineCoordinate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    document_id: str
    source_type: str = "prescription"
    language: list[str] = Field(default_factory=list)
    # source_image is a string for single-page, list for multi-page
    source_image: Union[str, list[str]] = ""
    page_number: Union[int, list[int]] = 1
    total_pages: int = 1
    ocr_engine: str = ""
    processed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pipeline_version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Document layout
# ---------------------------------------------------------------------------

class DocumentLayout(BaseModel):
    hospital_header: str = ""
    sections_detected: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Raw text
# ---------------------------------------------------------------------------

class RawText(BaseModel):
    """Per-page raw text plus concatenated full text."""
    full_text: str = ""
    pages: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Raw entities — mirroring ground-truth structure
# ---------------------------------------------------------------------------

class PatientInformation(BaseModel):
    name: str = ""
    age: str = ""
    gender: str = ""
    address: str = ""
    phone: str = ""
    patient_identifier: str = ""
    abha_id: str = ""
    # Additional demographic fields seen in ground truths
    occupation: str = ""
    w_o: str = ""   # Wife Of / Ward Of
    extra_fields: dict[str, str] = Field(default_factory=dict)


class EncounterInformation(BaseModel):
    date: str = ""
    department: str = ""
    hospital_name: str = ""
    doctor_name: str = ""
    visit_type: str = ""
    fees: str = ""
    room_queue_no: str = ""
    extra_fields: dict[str, str] = Field(default_factory=dict)


class ComplaintOrDiagnosis(BaseModel):
    """Raw complaint / diagnosis text — NO normalization."""
    raw_text: str = ""
    duration: str = ""


class Observation(BaseModel):
    """
    Can be a simple string (ophthalmology style) or structured
    (vitals with type/value). Stored as raw_text to be safe.
    """
    raw_text: str = ""


class Vital(BaseModel):
    type: str = ""
    value: str = ""


class LabObservationRow(BaseModel):
    """A single row from a tabular glucose/BP monitoring log."""
    date: str = ""
    fbs: str = ""
    pl: str = ""
    bp: str = ""
    pp: str = ""
    other: str = ""
    extra_fields: dict[str, str] = Field(default_factory=dict)


class Medication(BaseModel):
    """
    All subfields prefixed with 'raw_' to signal no normalization.
    Stores exactly what OCR produced — typos included.
    """
    raw_medication_text: str = ""
    raw_dosage_text: str = ""        # Tablet / Injection / e/d / Syrup
    raw_dose_text: str = ""          # 50 mg / 8-8-8
    raw_route_text: str = ""         # PO / IV / SC / topical
    raw_frequency_text: str = ""     # od / bd / tds / qid / HS / sos
    raw_duration_text: str = ""      # 5 days / x10 days / 2 weeks
    raw_timing_text: str = ""        # 9 pm / BBF
    raw_instruction_text: str = ""   # take with lukewarm water / खाली पेट
    raw_notes: str = ""


class FollowUp(BaseModel):
    date: str = ""
    review_after: str = ""
    day: str = ""
    appointment_time: str = ""


class RawEntities(BaseModel):
    """
    Superset of all entity types seen across ground-truth files.
    All fields are optional — empty means not found (never hallucinated).
    """
    patient_information: PatientInformation = Field(
        default_factory=PatientInformation
    )
    encounter_information: EncounterInformation = Field(
        default_factory=EncounterInformation
    )
    complaints_or_diagnosis: list[Union[ComplaintOrDiagnosis, str]] = Field(
        default_factory=list
    )
    diagnosis: list[ComplaintOrDiagnosis] = Field(default_factory=list)
    clinical_history: list[str] = Field(default_factory=list)
    observations: list[Union[Observation, str]] = Field(default_factory=list)
    vitals: list[Vital] = Field(default_factory=list)
    neurological_exam: list[str] = Field(default_factory=list)
    lab_observations: list[LabObservationRow] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    follow_up: Union[FollowUp, str] = Field(default_factory=FollowUp)
    other_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine metadata
# ---------------------------------------------------------------------------

class EngineMetadata(BaseModel):
    engine_name: str = ""
    engine_version: str = ""
    model_id: str = ""
    processing_time_seconds: Optional[float] = None
    overall_confidence: Optional[float] = None
    supports_bounding_boxes: bool = False
    preprocessing_applied: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level output schema
# ---------------------------------------------------------------------------

class OCROutput(BaseModel):
    """
    Standardised output JSON produced for every prescription+engine pair.

    Saved to: data/outputs/<engine_name>/<patient_id>.json
    """
    document_metadata: DocumentMetadata
    document_layout: DocumentLayout = Field(default_factory=DocumentLayout)
    raw_text: RawText = Field(default_factory=RawText)
    raw_entities: RawEntities = Field(default_factory=RawEntities)
    ocr_coordinates: OCRCoordinates = Field(default_factory=OCRCoordinates)
    ocr_engine_metadata: EngineMetadata = Field(default_factory=EngineMetadata)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
