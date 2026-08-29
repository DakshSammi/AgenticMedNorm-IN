from __future__ import annotations

from pathlib import Path


LEGACY_REFERENCE_ROOT = Path("legacy/ClinDoc-Weave-In")
LEGACY_SCHEMA_FILES = {
    "annotation_record": "schemas/annotation_record.schema.json",
    "clinical_mention": "schemas/clinical_mention.schema.json",
    "normalization_candidate": "schemas/normalization_candidate.schema.json",
    "normalized_entity": "schemas/normalized_entity.schema.json",
    "evidence_graph": "schemas/evidence_graph.schema.json",
}


def legacy_schema_path(name: str, root: Path = LEGACY_REFERENCE_ROOT) -> Path:
    if name not in LEGACY_SCHEMA_FILES:
        raise KeyError(f"Unknown legacy schema reference: {name}")
    return root / LEGACY_SCHEMA_FILES[name]
