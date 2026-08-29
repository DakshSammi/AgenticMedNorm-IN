from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactProvenance(BaseModel):
    artifact_id: str
    artifact_type: str
    parent_artifact_ids: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    source_sha256: list[str] = Field(default_factory=list)
    run_id: str
    pipeline_version: str
    git_commit: str | None = None
    model_provider: str | None = None
    model_id: str | None = None
    model_settings: dict[str, Any] | None = None
    prompt_version: str | None = None
    resource_versions: dict[str, str] = Field(default_factory=dict)
    retrieval_config_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    human_review_status: str | None = None


class SourceFieldValue(BaseModel):
    source_field_name: str
    source_field_value: Any = None


HumanReviewStatus = Literal["NOT_REQUIRED", "PENDING", "IN_REVIEW", "COMPLETE", "UNKNOWN"]
