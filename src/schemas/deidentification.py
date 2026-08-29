from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas.provenance import ArtifactProvenance


DeidentificationStatus = Literal["SUCCESS", "NEEDS_REVIEW", "FAILED"]
QCStatus = Literal["NOT_CHECKED", "PASS", "FAIL", "AMBIGUOUS"]


class DeidentificationRequest(BaseModel):
    page_uid: str
    raw_image_path: str
    raw_image_sha256: str
    run_id: str | None = None


class DeidentificationResult(BaseModel):
    page_uid: str
    output_path: str | None = None
    output_sha256: str | None = None
    status: DeidentificationStatus
    qc_status: QCStatus = "NOT_CHECKED"
    redaction_metadata: dict[str, Any] = Field(default_factory=dict)
    tool_version: str | None = None
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_code: str | None = None
    provenance: ArtifactProvenance | None = None
