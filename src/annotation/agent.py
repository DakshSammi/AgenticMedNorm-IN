from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Literal

import pandas as pd
from openai import OpenAI
from openai.types.responses import Response
from pydantic import BaseModel, Field, ValidationError, ConfigDict


# ============================================================
# Schema aligned with historical annotations
# ============================================================

class HistoricalMedication(BaseModel):
    model_config = ConfigDict(extra='forbid')
    raw_medication_text: str
    raw_dosage_text: Optional[str] = Field(default=None)
    raw_frequency_text: Optional[str] = Field(default=None)
    raw_timing_text: Optional[str] = Field(default=None)
    raw_duration_text: Optional[str] = Field(default=None)
    raw_route_text: Optional[str] = Field(default=None)
    page_number: int = Field(default=1)


class ClinicalNotes(BaseModel):
    model_config = ConfigDict(extra='forbid')
    medications: list[HistoricalMedication] = Field(default_factory=list)


class RawEntities(BaseModel):
    model_config = ConfigDict(extra='forbid')
    clinical_history: list[str] = Field(default_factory=list)
    medications: list[HistoricalMedication] = Field(default_factory=list)
    clinical_notes: Optional[ClinicalNotes] = Field(default=None)
    prescription: list[HistoricalMedication] = Field(default_factory=list)
    plan_of_care: Optional[ClinicalNotes] = Field(default=None)
    chief_complaint: Optional[str] = Field(default=None)
    complaints_or_diagnosis: Optional[str] = Field(default=None)
    diagnosis: Optional[str] = Field(default=None)
    clinical_impression: Optional[str] = Field(default=None)
    history: Optional[str] = Field(default=None)
    clinical_findings: Optional[str] = Field(default=None)
    clinical_examination: Optional[str] = Field(default=None)
    examination_findings: Optional[str] = Field(default=None)
    investigations: Optional[str] = Field(default=None)
    investigations_advised: Optional[str] = Field(default=None)
    lab_observations: Optional[str] = Field(default=None)
    laboratory_results: Optional[str] = Field(default=None)
    procedures: Optional[str] = Field(default=None)
    advice: Optional[str] = Field(default=None)
    recommendation: Optional[str] = Field(default=None)
    follow_up: Optional[str] = Field(default=None)


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: Optional[str] = Field(default=None)
    source_type: str = Field(default="prescription")
    language: list[str] = Field(default_factory=lambda: ["en"])
    total_pages: int = Field(default=1)


class AnnotationOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    raw_entities: RawEntities = Field(default_factory=RawEntities)


def get_annotation_output_schema() -> dict[str, Any]:
    """Generate JSON Schema from Pydantic model for Structured Outputs."""
    schema = AnnotationOutput.model_json_schema()
    # Ensure all object schemas have required = list of all properties
    def add_required(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["required"] = list(obj["properties"].keys())
            for v in obj.values():
                add_required(v)
        elif isinstance(obj, list):
            for item in obj:
                add_required(item)
    add_required(schema)
    # Also process $defs
    if "$defs" in schema:
        for def_name in list(schema["$defs"].keys()):
            add_required(schema["$defs"][def_name])
    return schema


# ============================================================
# Prompt
# ============================================================

PROMPT_V1 = """You are a medical document transcription specialist. Transcribe ONLY what is visibly present in this de-identified Indian prescription image.

CRITICAL RULES:
1. TRANSCRIBE VISIBLE CONTENT ONLY — never infer, normalize, or complete missing information
2. NO SEMANTIC DRUG NORMALIZATION — do NOT map to RxCUI, ATC, canonical generic, or brand names
3. PRESERVE UNCERTAINTY — if text is unclear, omit the field rather than guessing
4. NEVER INVENT medication attributes not visibly present
5. RETURN JSON matching the exact schema provided

Extract into the schema:
- document_metadata: document_id (if visible), source_type, language, total_pages
- raw_entities: medications[], clinical_history[], clinical_notes, prescription[], plan_of_care[]
- Each medication: raw_medication_text, raw_dosage_text, raw_frequency_text, raw_timing_text, raw_duration_text, raw_route_text, page_number
- Clinical context: clinical_history, diagnosis, investigations, advice, follow_up (only if visibly present)

Do not spend tokens on de-identified header details irrelevant to medication transcription.

Output MUST conform to the supplied JSON Schema."""


def load_prompt_sha() -> tuple[str, str]:
    return PROMPT_V1, hashlib.sha256(PROMPT_V1.encode("utf-8")).hexdigest()


# ============================================================
# Image handling
# ============================================================

def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================================
# OpenAI Responses API client
# ============================================================

@dataclass
class ModelCallResult:
    annotation: Optional[AnnotationOutput]
    status: Literal["SUCCESS", "NEEDS_REVIEW", "FAILED", "BLOCKED"]
    error_code: Optional[str]
    error_message: Optional[str]
    attempts: int
    model_used: str
    reasoning_effort_sent: bool
    structured_outputs_used: bool
    response_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


def call_annotation_model(
    image_b64: str,
    model: str = "gpt-5.5-2026-04-23",
    reasoning_effort: str = "high",
    max_retries: int = 2,
) -> ModelCallResult:
    """Call GPT-5.5 via Responses API with Structured Outputs and reasoning.effort=high."""
    
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANNOTATION_API_KEY")
    if not api_key:
        return ModelCallResult(
            annotation=None,
            status="BLOCKED",
            error_code="CREDENTIALS_MISSING",
            error_message="No OPENAI_API_KEY or ANNOTATION_API_KEY environment variable set",
            attempts=0,
            model_used="",
            reasoning_effort_sent=False,
            structured_outputs_used=False,
        )
    
    client = OpenAI(api_key=api_key)
    prompt, prompt_sha = load_prompt_sha()
    schema = get_annotation_output_schema()
    
    current_prompt = prompt
    
    for attempt in range(1, max_retries + 2):
        try:
            # Use Responses API with Structured Outputs
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": current_prompt},
                    {"role": "user", "content": [
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}
                    ]},
                ],
                reasoning={"effort": reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "annotation_output",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            
            model_used = response.model
            response_id = response.id
            
            # Parse output
            output_text = response.output_text
            if not output_text:
                raise ValueError("Empty response output_text")
            
            data = json.loads(output_text)
            annotation = AnnotationOutput(**data)
            
            # Determine confidence-based review status
            status = "SUCCESS"
            meds = annotation.raw_entities.medications
            if not meds:
                status = "NEEDS_REVIEW"
            
            return ModelCallResult(
                annotation=annotation,
                status=status,
                error_code=None,
                error_message=None,
                attempts=attempt,
                model_used=model_used,
                reasoning_effort_sent=True,
                structured_outputs_used=True,
                response_id=response_id,
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
            )
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error: {e}"
        except ValidationError as e:
            error_msg = f"Schema validation failed: {e}"
        except Exception as e:
            error_msg = f"API error: {type(e).__name__}: {e}"
        
        if attempt <= max_retries:
            current_prompt = f"{prompt}\n\nPREVIOUS ATTEMPT FAILED (attempt {attempt}):\n{error_msg}\n\nCorrect the JSON to match the schema exactly. Return only the corrected JSON."
            time.sleep(1)
        else:
            return ModelCallResult(
                annotation=None,
                status="FAILED",
                error_code="SCHEMA_VALIDATION_FAILED" if "validation" in error_msg.lower() else "API_ERROR",
                error_message=error_msg,
                attempts=attempt,
                model_used=model,
                reasoning_effort_sent=True,
                structured_outputs_used=True,
            )
    
    return ModelCallResult(
        annotation=None,
        status="FAILED",
        error_code="MAX_RETRIES_EXCEEDED",
        error_message="Max retries exceeded",
        attempts=max_retries + 1,
        model_used=model,
        reasoning_effort_sent=True,
        structured_outputs_used=True,
    )


# ============================================================
# Task processing
# ============================================================

def process_annotation_task(
    task: dict[str, Any],
    model: str = "gpt-5.5-2026-04-23",
    reasoning_effort: str = "high",
    project_root: Path | None = None,
    run_id: str = "RUN_ANNOTATION_PILOT_V1",
) -> dict[str, Any]:
    """Process a single annotation task."""
    project_root = project_root or Path(os.environ.get("AGENTICMEDNORM_PROJECT_ROOT", ".")).resolve()
    page_uid = task["page_uid"]
    inference_group_id = task["inference_group_id"]
    document_uid = task.get("document_uid")
    page_number = task.get("page_number")
    collection_date = task["collection_date"]
    source_type = task["source_type"]
    deidentified_image_path = task["deidentified_image_path"]
    deidentified_sha256 = task["deidentified_sha256"]
    duplicate_group_id = task.get("duplicate_group_id")
    canonical_inference_page_uid = task.get("canonical_inference_page_uid")
    derived_from_duplicate_representative = task.get("derived_from_duplicate_representative")
    
    image_path = project_root / deidentified_image_path
    if not image_path.exists():
        return {
            "task_id": task["task_id"],
            "page_uid": page_uid,
            "inference_group_id": inference_group_id,
            "annotation_status": "FAILED",
            "error_code": "IMAGE_NOT_FOUND",
            "error_message": f"Deidentified image not found: {deidentified_image_path}",
            "attempts": 0,
            "model_used": "",
            "reasoning_effort_sent": False,
            "structured_outputs_used": False,
        }
    
    image_b64 = image_to_base64(image_path)
    result = call_annotation_model(image_b64, model, reasoning_effort)
    
    if result.annotation:
        artifact = {
            "artifact_id": f"ART_ANN_{hashlib.sha256(f'{page_uid}{deidentified_sha256}'.encode()).hexdigest()[:20]}",
            "page_uid": page_uid,
            "inference_group_id": inference_group_id,
            "document_uid": document_uid,
            "page_number": page_number,
            "collection_date": collection_date,
            "source_type": source_type,
            "deidentified_image_path": deidentified_image_path,
            "deidentified_sha256": deidentified_sha256,
            "annotation": result.annotation.model_dump(),
            "annotation_status": result.status,
            "annotation_model": result.model_used,
            "reasoning_effort": reasoning_effort,
            "prompt_version": "v1",
            "prompt_sha256": load_prompt_sha()[1],
            "duplicate_group_id": duplicate_group_id if duplicate_group_id else None,
            "canonical_inference_page_uid": canonical_inference_page_uid if canonical_inference_page_uid else None,
            "derived_from_duplicate_representative": derived_from_duplicate_representative if derived_from_duplicate_representative else None,
            "tool_version": "annotation_agent_v1.0",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": result.error_code,
            "api_response_id": result.response_id,
            "api_input_tokens": result.input_tokens,
            "api_output_tokens": result.output_tokens,
        }
        
        return {
            "task_id": task["task_id"],
            "page_uid": page_uid,
            "inference_group_id": inference_group_id,
            "annotation_status": result.status,
            "artifact": artifact,
            "attempts": result.attempts,
            "model_used": result.model_used,
            "reasoning_effort_sent": result.reasoning_effort_sent,
            "structured_outputs_used": result.structured_outputs_used,
        }
    else:
        return {
            "task_id": task["task_id"],
            "page_uid": page_uid,
            "inference_group_id": inference_group_id,
            "annotation_status": result.status,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "attempts": result.attempts,
            "model_used": result.model_used,
            "reasoning_effort_sent": result.reasoning_effort_sent,
            "structured_outputs_used": result.structured_outputs_used,
        }


# ============================================================
# Pilot sampling
# ============================================================

def select_stratified_pilot(
    queue_path: Path,
    max_tasks: int = 20,
) -> pd.DataFrame:
    """Select 20 unique inference inputs stratified across dates, layouts, source types."""
    queue = pd.read_csv(queue_path)
    canonical = queue[queue["is_canonical_inference_input"] == True].copy()
    
    # Add layout profile info by reading bulk results
    # For now, stratify by collection_date and source_type
    canonical["_stratum"] = canonical["collection_date"] + "_" + canonical["source_type"]
    
    # Sample proportionally from each stratum
    strata = canonical.groupby("_stratum")
    pilot_rows = []
    
    for stratum_name, group in strata:
        # Take at least 1 from each, proportional to size
        n = max(1, int(len(group) * max_tasks / len(canonical)))
        n = min(n, len(group))
        sampled = group.sample(n=n, random_state=42)
        pilot_rows.append(sampled)
    
    pilot = pd.concat(pilot_rows).head(max_tasks).reset_index(drop=True)
    
    # Add selection rationale
    pilot["selection_rationale"] = pilot["_stratum"]
    
    return pilot


# ============================================================
# Bulk execution with resume
# ============================================================

def run_bulk_annotation(
    queue_path: Path,
    output_dir: Path,
    model: str = "gpt-5.5-2026-04-23",
    reasoning_effort: str = "high",
    project_root: Path | None = None,
    max_concurrent: int = 3,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Run bulk annotation with resume capability and controlled concurrency."""
    project_root = project_root or Path(os.environ.get("AGENTICMEDNORM_PROJECT_ROOT", ".")).resolve()
    queue = pd.read_csv(queue_path)
    canonical = queue[queue["is_canonical_inference_input"] == True].copy()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for idx, task in canonical.iterrows():
        out_file = output_dir / f"{task['page_uid']}_annotation.json"
        
        # Resume: skip if already SUCCESS
        if resume and out_file.exists():
            try:
                existing = json.loads(out_file.read_text())
                if existing.get("annotation_status") == "SUCCESS":
                    results.append({
                        "task_id": task["task_id"],
                        "page_uid": task["page_uid"],
                        "inference_group_id": task["inference_group_id"],
                        "annotation_status": "SUCCESS",
                        "resumed": True,
                    })
                    continue
            except:
                pass
        
        result = process_annotation_task(
            task.to_dict(),
            model=model,
            reasoning_effort=reasoning_effort,
            project_root=project_root,
            run_id="RUN_ANNOTATION_BULK_V1",
        )
        results.append(result)
        
        # Save result
        if "artifact" in result:
            out_file.write_text(json.dumps(result["artifact"], indent=2), encoding="utf-8")
        
        # Brief pause between requests for rate limiting
        time.sleep(0.5)
    
    return results


if __name__ == "__main__":
    print("Annotation agent v1.0 loaded")
    print(f"Model: gpt-5.5-2026-04-23")
    print(f"API: Responses API with Structured Outputs")
    print(f"Reasoning effort: high")
