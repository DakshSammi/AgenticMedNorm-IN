import re
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ENTITY_TYPES = {
    "medication", "dosage", "frequency", "duration", "diagnosis", "complaint",
    "observation", "vital", "lab_result", "procedure", "advice", "follow_up",
}

def normalize(text: Any) -> str:
    """Normalizes string to lowercase alphanumeric words."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

def validate_semantic_format(data: Any, doc_id: str, source_system: str = "") -> Tuple[bool, List[str]]:
    """Validates semantic extraction JSON structure and field names."""
    errors = []
    if not isinstance(data, dict):
        return False, ["root_not_object"]
    for key in ["document_id", "source_system", "semantic_entities", "semantic_relations", "unsupported_inferences", "warnings", "metadata"]:
        if key not in data:
            errors.append(f"missing_{key}")
    if data.get("document_id") != doc_id:
        errors.append("document_id_mismatch")
    if source_system and data.get("source_system") != source_system:
        errors.append("source_system_mismatch")
    for key in ["semantic_entities", "semantic_relations", "unsupported_inferences", "warnings"]:
        if key in data and not isinstance(data.get(key), list):
            errors.append(f"{key}_not_list")
            
    required = {"semantic_type", "normalized_name", "raw_evidence_text", "source_page_or_image", "confidence", "evidence_supported", "normalization_method"}
    for index, entity in enumerate(data.get("semantic_entities") or []):
        if not isinstance(entity, dict):
            errors.append(f"entity_{index}_not_object")
            continue
        missing = required - set(entity)
        if missing:
            errors.append(f"entity_{index}_missing_fields_{sorted(list(missing))}")
        if entity.get("semantic_type") not in ENTITY_TYPES:
            errors.append(f"entity_{index}_invalid_type_{entity.get('semantic_type')}")
            
    return not errors, errors

def extract_raw_entities(path: Path) -> List[str]:
    """Extracts raw text strings from Stage 1B structured extraction JSON files."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values: List[str] = []
    for key in ["complaints_or_diagnosis", "observations", "medications", "advice"]:
        for value in data.get(key) or []:
            values.append(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    if isinstance(data.get("follow_up"), str):
        values.append(data["follow_up"])
    elif isinstance(data.get("follow_up"), dict):
        values.append(json.dumps(data["follow_up"], ensure_ascii=False))
    return [value for value in values if normalize(value)]

def evidence_matches(evidence: str, candidates: List[str]) -> bool:
    """Checks if the evidence quote exists in candidate sentences using fuzzy overlap."""
    needle = normalize(evidence)
    if not needle:
        return False
    for candidate in candidates:
        haystack = normalize(candidate)
        if needle in haystack or haystack in needle:
            return True
        left, right = set(needle.split()), set(haystack.split())
        if left and right and len(left & right) / max(1, len(left)) >= 0.7:
            return True
    return False
