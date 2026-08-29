from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["PENDING", "RUNNING", "SUCCESS", "NEEDS_REVIEW", "FAILED", "SKIPPED_DUPLICATE", "BLOCKED"]


class StageRun(BaseModel):
    run_id: str
    stage_name: str
    status: TaskStatus
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    error_count: int = 0


class PipelineTask(BaseModel):
    task_id: str
    stage_name: str
    stable_input_id: str
    status: TaskStatus = "PENDING"
    priority: int = 100
    reason: str | None = None
    duplicate_group_id: str | None = None
    canonical_duplicate_representative: str | None = None
