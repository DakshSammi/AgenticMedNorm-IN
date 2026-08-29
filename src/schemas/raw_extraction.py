from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    backend_name: str
    processing_time_ms: float = 0.0
    decoding_parameters: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "raw_rx_v2"
    timestamp: Optional[str] = None
    confidence_score: Optional[float] = None
    uncertainty_notes: Optional[str] = None
    pages: List[Dict[str, Any]] = Field(default_factory=list)
    document_type: Optional[str] = None
    validation_warnings: List[str] = Field(default_factory=list)
    type_coercions: List[str] = Field(default_factory=list)


class PatientInformation(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    patient_identifier: Optional[str] = None
    abha_id: Optional[str] = None

class EncounterInformation(BaseModel):
    date: Optional[str] = None
    department: Optional[str] = None
    hospital_name: Optional[str] = None
    doctor_name: Optional[str] = None
    visit_type: Optional[str] = None
    fees: Optional[str] = None
    room_or_queue_no: Optional[str] = None

class RawEntityItem(BaseModel):
    raw_text: str
    evidence_text: Optional[str] = None
    page_number: int = 1
    confidence: Optional[float] = None
    section: Optional[str] = None
    evidence_line_ids: List[str] = Field(default_factory=list)
    original_category: Optional[str] = None
    original_field_path: Optional[str] = None
    adapter_transformation_notes: Optional[str] = None

class RawMedicationItem(BaseModel):
    raw_line_text: str
    raw_name: Optional[str] = None
    raw_dosage: Optional[str] = None
    raw_route: Optional[str] = None
    raw_frequency: Optional[str] = None
    raw_duration: Optional[str] = None
    raw_instruction: Optional[str] = None
    raw_timing: Optional[str] = None
    evidence_text: Optional[str] = None
    page_number: int = 1
    confidence: Optional[float] = None
    section: Optional[str] = None
    evidence_line_ids: List[str] = Field(default_factory=list)
    original_category: Optional[str] = None
    original_field_path: Optional[str] = None
    adapter_transformation_notes: Optional[str] = None

class RawFollowUp(BaseModel):
    raw_text: Optional[str] = None
    date: Optional[str] = None
    review_after: Optional[str] = None

class RawLabObservationItem(BaseModel):
    raw_line_text: Optional[str] = None
    test_name: Optional[str] = None
    result: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    evidence_text: Optional[str] = None
    page_number: int = 1
    confidence: Optional[float] = None
    section: Optional[str] = None
    evidence_line_ids: List[str] = Field(default_factory=list)
    original_category: Optional[str] = None
    original_field_path: Optional[str] = None
    adapter_transformation_notes: Optional[str] = None

class CanonicalRawDoc(BaseModel):
    schema_version: str = "raw_rx_v2"
    document_id: str
    patient_information: PatientInformation = Field(default_factory=PatientInformation)
    encounter_information: EncounterInformation = Field(default_factory=EncounterInformation)
    complaints_or_diagnosis: List[RawEntityItem] = Field(default_factory=list)
    observations: List[RawEntityItem] = Field(default_factory=list)
    medications: List[RawMedicationItem] = Field(default_factory=list)
    procedures: List[RawEntityItem] = Field(default_factory=list)
    advice: List[RawEntityItem] = Field(default_factory=list)
    follow_up: Optional[RawFollowUp] = None
    allergy_mentions: List[RawEntityItem] = Field(default_factory=list)
    other_notes: List[RawEntityItem] = Field(default_factory=list)
    lab_observations: List[RawLabObservationItem] = Field(default_factory=list)
    metadata: Optional[Metadata] = None
