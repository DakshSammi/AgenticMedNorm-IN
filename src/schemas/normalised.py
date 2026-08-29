from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.schemas.raw_extraction import CanonicalRawDoc

class NormalisationTrace(BaseModel):
    source_raw_text: str = Field(..., description="Raw text before normalization")
    source_entity_id: Optional[str] = Field(None, description="Optional index or key of source raw item")
    normalisation_method: str = Field(..., description="LLM, lookup dictionary, regex rule-based, etc.")
    confidence: float = Field(1.0, description="Confidence in the normalization step")

class NormalisedMedication(BaseModel):
    raw_medication_text: str
    normalized_medication_name: str
    generic_name: Optional[str] = None
    drug_class: Optional[str] = None
    dosage_normalized: Optional[str] = None
    frequency_normalized: Optional[str] = None
    route_normalized: Optional[str] = None
    duration_normalized: Optional[str] = None
    instructions_normalized: Optional[str] = None
    trace: NormalisationTrace

class NormalisedCondition(BaseModel):
    raw_text: str
    normalized_text: str
    clinical_category: Optional[str] = None
    trace: NormalisationTrace

class NormalisedObservation(BaseModel):
    raw_text: str
    normalized_text: str
    observation_type: Optional[str] = None
    trace: NormalisationTrace

class NormalisedProcedure(BaseModel):
    raw_text: str
    normalized_text: str
    procedure_category: Optional[str] = None
    trace: NormalisationTrace

class NormalisedAdvice(BaseModel):
    raw_text: str
    normalized_text: str
    advice_category: Optional[str] = None
    trace: NormalisationTrace

class AbbreviationExpansion(BaseModel):
    abbreviation: str
    expanded_form: str
    confidence: float = 1.0

class NormalisedClinicalDoc(BaseModel):
    document_id: str
    schema_version: str = "normalised_rx_v1"
    raw_doc: CanonicalRawDoc = Field(..., description="Direct link back to raw source extraction doc")
    normalized_conditions: List[NormalisedCondition] = Field(default_factory=list)
    normalized_medications: List[NormalisedMedication] = Field(default_factory=list)
    normalized_observations: List[NormalisedObservation] = Field(default_factory=list)
    normalized_procedures: List[NormalisedProcedure] = Field(default_factory=list)
    normalized_advice: List[NormalisedAdvice] = Field(default_factory=list)
    abbreviation_expansions: List[AbbreviationExpansion] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
