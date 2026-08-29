from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.schemas.layer_a import CanonicalDocument, CanonicalPage, DocumentContextBundle, LayerAMedicationMention
from src.schemas.provenance import SourceFieldValue
from src.utils.stable_ids import context_bundle_id, document_uid, lexical_surface, mention_id, page_uid


MEDICATION_LIST_PATHS = {
    "raw_entities.medications[]": "primary_raw_entities_medications",
    "raw_entities.clinical_notes.medications[]": "alternate_clinical_notes_medications",
    "raw_entities.prescription[]": "alternate_prescription",
    "raw_entities.plan_of_care.medications[]": "alternate_plan_of_care_medications",
    "reference_annotations[].raw_entities.medications[]": "reference_annotation_medications",
}

MEDICATION_NAME_FIELDS = ("raw_medication_text", "name", "medicine", "medicine_name")
STRENGTH_FIELDS = ("raw_strength_text", "strength")
DOSAGE_FIELDS = ("raw_dosage_text", "raw_dose_text", "dosage", "dose")
FREQUENCY_FIELDS = ("raw_frequency_text", "frequency")
DURATION_FIELDS = ("raw_duration_text", "duration")
ROUTE_FIELDS = ("raw_route_text", "route")
TIMING_FIELDS = ("raw_timing_text", "timing")
INSTRUCTION_FIELDS = ("raw_instruction_text", "instructions", "instruction")
NOTES_FIELDS = ("raw_notes", "note", "notes")


@dataclass(frozen=True)
class SourceObject:
    source_json_path: str
    source_object_index: int
    source_schema_variant: str
    value: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object root in {path}")
    return data


def normalize_language(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def discover_medication_objects(data: dict[str, Any]) -> list[SourceObject]:
    out: list[SourceObject] = []

    def add_from_list(path: str, value: Any) -> None:
        if not isinstance(value, list):
            return
        variant = MEDICATION_LIST_PATHS[path]
        for index, item in enumerate(value):
            if isinstance(item, dict):
                out.append(SourceObject(path, index, variant, item))

    raw = data.get("raw_entities")
    if isinstance(raw, dict):
        add_from_list("raw_entities.medications[]", raw.get("medications"))
        clinical_notes = raw.get("clinical_notes")
        if isinstance(clinical_notes, dict):
            add_from_list("raw_entities.clinical_notes.medications[]", clinical_notes.get("medications"))
        elif isinstance(clinical_notes, list):
            for note_index, note in enumerate(clinical_notes):
                if isinstance(note, dict):
                    for obj in _list_objects(note.get("medications")):
                        out.append(
                            SourceObject(
                                f"raw_entities.clinical_notes[{note_index}].medications[]",
                                len([x for x in out if x.source_json_path.startswith("raw_entities.clinical_notes")]),
                                MEDICATION_LIST_PATHS["raw_entities.clinical_notes.medications[]"],
                                obj,
                            )
                        )
        add_from_list("raw_entities.prescription[]", raw.get("prescription"))
        plan = raw.get("plan_of_care")
        if isinstance(plan, dict):
            add_from_list("raw_entities.plan_of_care.medications[]", plan.get("medications"))

    refs = data.get("reference_annotations")
    if isinstance(refs, list):
        for ref_index, ref in enumerate(refs):
            ref_raw = ref.get("raw_entities") if isinstance(ref, dict) else None
            if isinstance(ref_raw, dict):
                meds = ref_raw.get("medications")
                for med_index, obj in enumerate(_list_objects(meds)):
                    out.append(
                        SourceObject(
                            f"reference_annotations[{ref_index}].raw_entities.medications[]",
                            med_index,
                            MEDICATION_LIST_PATHS["reference_annotations[].raw_entities.medications[]"],
                            obj,
                        )
                    )
    return out


def deduplicate_source_objects(sources: list[SourceObject]) -> tuple[list[SourceObject], list[dict[str, Any]]]:
    """Conservatively remove explicit wrapper copies, not repeated medication lines."""
    primary_by_signature: dict[tuple[Any, ...], SourceObject] = {}
    kept: list[SourceObject] = []
    duplicates: list[dict[str, Any]] = []
    for source in sources:
        signature = medication_object_signature(source)
        is_reference_copy = source.source_json_path.startswith("reference_annotations")
        existing = primary_by_signature.get(signature)
        if is_reference_copy and existing is not None:
            duplicates.append(
                {
                    "kept_source_json_path": existing.source_json_path,
                    "kept_source_object_index": existing.source_object_index,
                    "duplicate_source_json_path": source.source_json_path,
                    "duplicate_source_object_index": source.source_object_index,
                    "deduplication_reason": "reference_annotation_copy_of_primary_object",
                }
            )
            continue
        kept.append(source)
        if not is_reference_copy:
            primary_by_signature.setdefault(signature, source)
    return kept, duplicates


def medication_object_signature(source: SourceObject) -> tuple[Any, ...]:
    obj = source.value
    _, raw_name = first_present(obj, MEDICATION_NAME_FIELDS)
    _, strength = first_present(obj, STRENGTH_FIELDS)
    _, dosage = first_present(obj, DOSAGE_FIELDS)
    _, frequency = first_present(obj, FREQUENCY_FIELDS)
    _, duration = first_present(obj, DURATION_FIELDS)
    _, route = first_present(obj, ROUTE_FIELDS)
    _, timing = first_present(obj, TIMING_FIELDS)
    page = page_number_for(obj)
    return (
        page,
        lexical_surface(str(raw_name)) if raw_name is not None else None,
        str(strength).strip().lower() if strength is not None else None,
        str(dosage).strip().lower() if dosage is not None else None,
        str(frequency).strip().lower() if frequency is not None else None,
        str(duration).strip().lower() if duration is not None else None,
        str(route).strip().lower() if route is not None else None,
        str(timing).strip().lower() if timing is not None else None,
    )


def source_path_family(source_json_path: str) -> str:
    if source_json_path.startswith("raw_entities.clinical_notes"):
        return "raw_entities.clinical_notes.medications[]"
    if source_json_path.startswith("reference_annotations"):
        return "reference_annotations[].raw_entities.medications[]"
    return source_json_path


def _list_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def first_present(obj: dict[str, Any], fields: tuple[str, ...]) -> tuple[str | None, Any]:
    for field in fields:
        if field in obj:
            return field, obj.get(field)
    return None, None


def source_fields(obj: dict[str, Any]) -> list[SourceFieldValue]:
    return [SourceFieldValue(source_field_name=str(k), source_field_value=v) for k, v in obj.items()]


def page_number_for(obj: dict[str, Any]) -> int | None:
    value = obj.get("page_number")
    if value is None:
        return 1
    try:
        return int(value)
    except Exception:
        return None


def build_document(
    *,
    data: dict[str, Any],
    collection_date: str,
    source_json_relpath: str,
    source_json_sha256: str,
    duplicate_group_id: str | None = None,
    canonical_duplicate_representative: str | None = None,
) -> CanonicalDocument:
    metadata = data.get("document_metadata") if isinstance(data.get("document_metadata"), dict) else {}
    source_document_id = str(metadata.get("document_id") or Path(source_json_relpath).stem)
    total_pages = metadata.get("total_pages")
    try:
        total_pages_value = int(total_pages) if total_pages is not None else None
    except Exception:
        total_pages_value = None
    return CanonicalDocument(
        document_uid=document_uid(collection_date, source_document_id, source_json_relpath, source_json_sha256),
        source_document_id=source_document_id,
        collection_date=collection_date,
        source_json_relpath=source_json_relpath,
        source_json_sha256=source_json_sha256,
        total_pages=total_pages_value,
        language=normalize_language(metadata.get("language")),
        source_type=metadata.get("source_type"),
        duplicate_group_id=duplicate_group_id,
        canonical_duplicate_representative=canonical_duplicate_representative,
    )


def build_page(
    *,
    document: CanonicalDocument,
    page_number: int | None,
    raw_image_relpath: str | None,
    raw_image_sha256: str | None,
    anonymized_image_relpath: str | None,
    anonymized_image_sha256: str | None,
    lineage_status: str,
    duplicate_group_id: str | None = None,
    canonical_duplicate_representative: str | None = None,
) -> CanonicalPage:
    image_identity = anonymized_image_sha256 or raw_image_sha256 or document.source_json_sha256
    return CanonicalPage(
        page_uid=page_uid(document.document_uid, page_number, image_identity),
        document_uid=document.document_uid,
        page_number=page_number,
        raw_image_relpath=raw_image_relpath or None,
        raw_image_sha256=raw_image_sha256 or None,
        anonymized_image_relpath=anonymized_image_relpath or None,
        anonymized_image_sha256=anonymized_image_sha256 or None,
        lineage_status=lineage_status,
        duplicate_group_id=duplicate_group_id,
        canonical_duplicate_representative=canonical_duplicate_representative,
        evaluation_lineage_eligible=lineage_status in {"VERIFIED_EXACT_METADATA", "VERIFIED_VISUAL_HIGH", "VERIFIED_VISUAL_MEDIUM"},
    )


def build_mention(
    *,
    source: SourceObject,
    document: CanonicalDocument,
    page: CanonicalPage,
    source_json_relpath: str,
    source_json_sha256: str,
) -> LayerAMedicationMention | None:
    obj = source.value
    _, raw_name = first_present(obj, MEDICATION_NAME_FIELDS)
    if raw_name is None:
        return None
    strength_field, strength = first_present(obj, STRENGTH_FIELDS)
    dosage_field, dosage = first_present(obj, DOSAGE_FIELDS)
    frequency_field, frequency = first_present(obj, FREQUENCY_FIELDS)
    duration_field, duration = first_present(obj, DURATION_FIELDS)
    route_field, route = first_present(obj, ROUTE_FIELDS)
    timing_field, timing = first_present(obj, TIMING_FIELDS)
    instruction_field, instruction = first_present(obj, INSTRUCTION_FIELDS)
    notes_field, notes = first_present(obj, NOTES_FIELDS)
    _ = (strength_field, dosage_field, frequency_field, duration_field, route_field, timing_field, instruction_field, notes_field)
    ctx_id = context_bundle_id(document.document_uid, page.page_number)
    return LayerAMedicationMention(
        mention_id=mention_id(document.document_uid, source.source_json_path, source.source_object_index),
        document_uid=document.document_uid,
        page_uid=page.page_uid,
        collection_date=document.collection_date,
        raw_medication_text=str(raw_name) if raw_name is not None else None,
        lexical_surface_normalized=lexical_surface(str(raw_name)) if raw_name is not None else None,
        raw_strength_text=str(strength) if strength is not None else None,
        raw_dosage_text=str(dosage) if dosage is not None else None,
        raw_frequency_text=str(frequency) if frequency is not None else None,
        raw_duration_text=str(duration) if duration is not None else None,
        raw_route_text=str(route) if route is not None else None,
        raw_timing_text=str(timing) if timing is not None else None,
        raw_instruction_text=str(instruction) if instruction is not None else None,
        raw_notes=str(notes) if notes is not None else None,
        source_json_path=source.source_json_path,
        source_object_index=source.source_object_index,
        source_json_relpath=source_json_relpath,
        source_json_sha256=source_json_sha256,
        source_schema_variant=source.source_schema_variant,
        context_bundle_id=ctx_id,
        source_fields=source_fields(obj),
    )


def build_context_bundle(document: CanonicalDocument, page: CanonicalPage, data: dict[str, Any]) -> DocumentContextBundle:
    raw = data.get("raw_entities") if isinstance(data.get("raw_entities"), dict) else {}
    refs = data.get("reference_annotations") if isinstance(data.get("reference_annotations"), list) else []
    ref_raw = refs[0].get("raw_entities") if refs and isinstance(refs[0], dict) and isinstance(refs[0].get("raw_entities"), dict) else {}
    merged = raw if raw else ref_raw
    return DocumentContextBundle(
        context_bundle_id=context_bundle_id(document.document_uid, page.page_number),
        document_uid=document.document_uid,
        page_uid=page.page_uid,
        chief_complaints=_as_list(merged.get("chief_complaint") or merged.get("complaints_or_diagnosis")),
        diagnoses=_as_list(merged.get("diagnosis") or merged.get("clinical_impression")),
        clinical_history=_as_list(merged.get("clinical_history") or merged.get("history")),
        clinical_findings=_as_list(merged.get("clinical_findings") or merged.get("clinical_examination") or merged.get("examination_findings")),
        investigations=_as_list(merged.get("investigations") or merged.get("investigations_advised")),
        lab_observations=_as_list(merged.get("lab_observations") or merged.get("laboratory_results")),
        procedures=_as_list(merged.get("procedures")),
        advice=_as_list(merged.get("advice") or merged.get("recommendation")),
        follow_up=_as_list(merged.get("follow_up")),
        other_context=_collect_other_context(merged),
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _collect_other_context(raw_entities: dict[str, Any]) -> list[Any]:
    known = {
        "chief_complaint",
        "complaints_or_diagnosis",
        "diagnosis",
        "clinical_impression",
        "clinical_history",
        "history",
        "clinical_findings",
        "clinical_examination",
        "examination_findings",
        "investigations",
        "investigations_advised",
        "lab_observations",
        "laboratory_results",
        "procedures",
        "advice",
        "recommendation",
        "follow_up",
        "medications",
        "prescription",
        "clinical_notes",
        "plan_of_care",
    }
    return [{key: value} for key, value in raw_entities.items() if key not in known]
