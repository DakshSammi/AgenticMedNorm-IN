from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class RankedCandidate(BaseModel):
    candidate_id: str
    ranking_position: int
    ranking_score: float | None = None
    ranking_components: dict[str, float] = Field(default_factory=dict)
    source_candidate_metadata: dict[str, Any] = Field(default_factory=dict)


class RankingResult(BaseModel):
    mention_id: str
    input_candidate_ids: list[str]
    ranked_candidates: list[RankedCandidate]
    ranking_method: str
    ranking_config_version: str

    @model_validator(mode="after")
    def output_is_subset_of_input(self) -> "RankingResult":
        input_ids = set(self.input_candidate_ids)
        output_ids = {candidate.candidate_id for candidate in self.ranked_candidates}
        if not output_ids <= input_ids:
            extra = sorted(output_ids - input_ids)
            raise ValueError(f"RankingResult introduced candidates not in input pool: {extra}")
        return self
