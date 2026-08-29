from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalBranch(StrEnum):
    R1_EXACT_FUZZY = "R1_EXACT_FUZZY"
    R2_BM25 = "R2_BM25"
    R3_BIOMEDICAL_DENSE = "R3_BIOMEDICAL_DENSE"
    R4_RXNORM = "R4_RXNORM"
    R5_INDIA_KB = "R5_INDIA_KB"


class MedicationCandidate(BaseModel):
    candidate_id: str
    candidate_name: str
    candidate_type: str
    brand_family_id: str | None = None
    brand_product_id: str | None = None
    formulation_id: str | None = None
    ingredient_ids: list[str] = Field(default_factory=list)
    retrieval_branch: RetrievalBranch
    branch_rank: int | None = None
    branch_score: float | None = None
    score_semantics: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rxnorm_rxcui: str | None = None
    local_product_id: str | None = None
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class CandidatePool(BaseModel):
    mention_id: str
    K: int
    branch_results: dict[RetrievalBranch, list[MedicationCandidate]] = Field(default_factory=dict)
    union_candidates: list[MedicationCandidate] = Field(default_factory=list)
    retrieval_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_config_version: str
