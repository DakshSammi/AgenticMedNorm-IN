from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class VerificationDecision(StrEnum):
    ACCEPT = "ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    NIL = "NIL"


class VerificationResult(BaseModel):
    mention_id: str
    decision: VerificationDecision
    selected_candidate_id: str | None = None
    decision_reason_codes: list[str] = Field(default_factory=list)
    hard_conflicts: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    verification_method: str
    verification_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def decision_invariants(self) -> "VerificationResult":
        if self.decision == VerificationDecision.ACCEPT and not self.selected_candidate_id:
            raise ValueError("ACCEPT requires selected_candidate_id")
        if self.decision == VerificationDecision.NIL and self.selected_candidate_id:
            raise ValueError("NIL must not select or fabricate a candidate")
        return self
