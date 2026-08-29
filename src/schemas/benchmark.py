from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ScalarMatchDetail(BaseModel):
    field_name: str
    gt_value: Optional[str] = None
    pred_value: Optional[str] = None
    exact_match: bool
    lenient_match: bool
    similarity_score: float

class EntityMatchDetail(BaseModel):
    category: str
    gt_raw_text: Optional[str] = None
    pred_raw_text: Optional[str] = None
    exact_match: bool
    lenient_match: bool
    similarity_score: float
    alignment_status: str = Field(..., description="TP_EXACT, TP_LENIENT, FP, FN")
    medication_subfields: Optional[Dict[str, Any]] = None

class UnmatchedPredictionRecord(BaseModel):
    category: str
    pred_text: str
    confidence: float
    evidence_text: Optional[str] = None
    classification: str = Field(..., description="likely_hallucination, annotation_gap_candidate, manual_review_required")
    rationale: str
    matched_snippet: Optional[str] = None

class CategoryMetrics(BaseModel):
    precision_exact: float = 0.0
    recall_exact: float = 0.0
    f1_exact: float = 0.0
    precision_lenient: float = 0.0
    recall_lenient: float = 0.0
    f1_lenient: float = 0.0

class DocumentBenchmarkResult(BaseModel):
    document_id: str
    document_type: Optional[str] = None
    schema_parse_success: int = 1
    scalar_accuracy_exact: float = 0.0
    scalar_accuracy_lenient: float = 0.0
    
    # Granular entity metrics
    metrics_by_category: Dict[str, CategoryMetrics] = Field(default_factory=dict)
    
    # Counts of categorizations
    likely_hallucination_count: int = 0
    annotation_gap_candidate_count: int = 0
    manual_review_required_count: int = 0
    
    # Rates
    hallucination_rate: float = 0.0
    missing_entity_rate: float = 0.0
    annotation_gap_rate: float = 0.0
    
    # Experimental headline score
    experimental_overall_score: float = 0.0
    
    # Detail lists
    scalars: List[ScalarMatchDetail] = Field(default_factory=list)
    entity_alignments: List[EntityMatchDetail] = Field(default_factory=list)
    unmatched_predictions: List[UnmatchedPredictionRecord] = Field(default_factory=list)
    
    # Runtime info
    model_name: Optional[str] = None
    backend_name: Optional[str] = None
    latency_ms: float = 0.0
