from __future__ import annotations

from pathlib import Path

from src.adapters.current_annotation_adapter import (
    build_document,
    build_mention,
    build_page,
    deduplicate_source_objects,
    discover_medication_objects,
    load_json,
)


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_annotation_adapter_extracts_mentions():
    path = ROOT / "data/examples/annotations/synth_p1.json"
    data = load_json(path)
    doc = build_document(data=data, collection_date="SYNTHETIC", source_json_relpath="data/examples/annotations/synth_p1.json", source_json_sha256="synthetic")
    page = build_page(document=doc, page_number=1, raw_image_relpath="", raw_image_sha256="", anonymized_image_relpath="", anonymized_image_sha256="", lineage_status="UNVERIFIED_HEURISTIC")
    kept, duplicates = deduplicate_source_objects(discover_medication_objects(data))
    mentions = [build_mention(source=s, document=doc, page=page, source_json_relpath="data/examples/annotations/synth_p1.json", source_json_sha256="synthetic") for s in kept]
    assert len(duplicates) == 0
    assert [m.raw_medication_text for m in mentions if m] == ["Tab Pantop 40", "Syp Mucaine Gel"]
