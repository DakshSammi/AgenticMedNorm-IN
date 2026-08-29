from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.current_annotation_adapter import deduplicate_source_objects, discover_medication_objects
from src.schemas.evidence import IngredientComponent, IngredientSet
from src.schemas.provenance import ArtifactProvenance
from src.schemas.ranking import RankedCandidate, RankingResult
from src.schemas.verification import VerificationDecision, VerificationResult
from src.utils.stable_ids import context_bundle_id, document_uid, mention_id, page_uid


ROOT = Path(__file__).resolve().parents[1]


def expected_preserved_json_paths() -> list[Path]:
    return sorted((ROOT / "prescription_pipeline_jbhi_ieee/ground_truths_json").rglob("p*.json"))


def expected_medication_source_count() -> int:
    total = 0
    for path in expected_preserved_json_paths():
        data = json.loads(path.read_text(encoding="utf-8"))
        total += len(discover_medication_objects(data))
    return total


def test_every_preserved_json_loads_and_adapter_classifies():
    paths = expected_preserved_json_paths()
    assert paths
    total = 0
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        sources = discover_medication_objects(data)
        assert isinstance(sources, list)
        total += len(sources)
    assert total > 0


def test_supported_medication_source_paths_with_synthetic_fixture():
    payload = {
        "raw_entities": {
            "medications": [{"raw_medication_text": "Synthetic A"}],
            "clinical_notes": {"medications": [{"medicine": "Synthetic B", "strength": "1 mg"}]},
            "prescription": [{"medicine": "Synthetic C", "dose": "1"}],
            "plan_of_care": {"medications": [{"medicine_name": "Synthetic D", "dosage": "1-0-0"}]},
        },
        "reference_annotations": [{"raw_entities": {"medications": [{"raw_medication_text": "Synthetic E"}]}}],
    }
    paths = {source.source_schema_variant for source in discover_medication_objects(payload)}
    assert {
        "primary_raw_entities_medications",
        "alternate_clinical_notes_medications",
        "alternate_prescription",
        "alternate_plan_of_care_medications",
        "reference_annotation_medications",
    } <= paths


def test_reference_annotation_copy_is_deduplicated_conservatively():
    payload = {
        "raw_entities": {"medications": [{"raw_medication_text": "Synthetic A", "raw_dosage_text": "1 mg", "page_number": 1}]},
        "reference_annotations": [{"raw_entities": {"medications": [{"raw_medication_text": "Synthetic A", "raw_dosage_text": "1 mg", "page_number": 1}]}}],
    }
    kept, duplicates = deduplicate_source_objects(discover_medication_objects(payload))
    assert len(kept) == 1
    assert len(duplicates) == 1


def test_repeated_same_name_lines_are_not_deduplicated():
    payload = {"raw_entities": {"medications": [{"raw_medication_text": "Synthetic A"}, {"raw_medication_text": "Synthetic A"}]}}
    kept, duplicates = deduplicate_source_objects(discover_medication_objects(payload))
    assert len(kept) == 2
    assert duplicates == []


def test_stable_ids_are_reproducible_and_do_not_depend_on_medication_text():
    doc1 = document_uid("01-01-2099", "p1", "x/p1.json", "abc")
    doc2 = document_uid("01-01-2099", "p1", "x/p1.json", "abc")
    assert doc1 == doc2
    page1 = page_uid(doc1, 1, "image-sha")
    assert page1 == page_uid(doc2, 1, "image-sha")
    assert mention_id(doc1, "raw_entities.medications[]", 0) == mention_id(doc1, "raw_entities.medications[]", 0)
    assert context_bundle_id(doc1, 1) == context_bundle_id(doc1, 1)


def test_generated_layer_a_outputs_exist_and_counts_match():
    mentions = pd.read_csv(ROOT / "derived/layer_a_medication_mentions.csv")
    docs = pd.read_csv(ROOT / "derived/layer_a_documents.csv")
    pages = pd.read_csv(ROOT / "derived/layer_a_pages.csv")
    assert len(docs) == len(expected_preserved_json_paths())
    assert len(pages) == len(expected_preserved_json_paths())
    assert len(mentions) == expected_medication_source_count()
    assert mentions["mention_id"].is_unique


def test_ranking_contract_cannot_introduce_candidate():
    RankingResult(
        mention_id="MENT_x",
        input_candidate_ids=["a"],
        ranked_candidates=[RankedCandidate(candidate_id="a", ranking_position=1)],
        ranking_method="synthetic",
        ranking_config_version="test",
    )
    with pytest.raises(ValueError):
        RankingResult(
            mention_id="MENT_x",
            input_candidate_ids=["a"],
            ranked_candidates=[RankedCandidate(candidate_id="b", ranking_position=1)],
            ranking_method="synthetic",
            ranking_config_version="test",
        )


def test_verification_contract_invariants():
    VerificationResult(
        mention_id="MENT_x",
        decision=VerificationDecision.ACCEPT,
        selected_candidate_id="cand_a",
        verification_method="synthetic",
        verification_version="test",
    )
    with pytest.raises(ValueError):
        VerificationResult(mention_id="MENT_x", decision=VerificationDecision.ACCEPT, verification_method="synthetic", verification_version="test")
    with pytest.raises(ValueError):
        VerificationResult(
            mention_id="MENT_x",
            decision=VerificationDecision.NIL,
            selected_candidate_id="cand_a",
            verification_method="synthetic",
            verification_version="test",
        )


def test_provenance_and_ingredient_set_serializable():
    prov = ArtifactProvenance(
        artifact_id="art_1",
        artifact_type="synthetic",
        run_id="run_1",
        pipeline_version="test",
        source_paths=["synthetic.json"],
        source_sha256=["0" * 64],
    )
    assert json.loads(prov.model_dump_json())["artifact_id"] == "art_1"
    ing = IngredientSet(components=[IngredientComponent(raw_name="synthetic ingredient", strength_value="1", strength_unit="mg", order=1)])
    assert len(json.loads(ing.model_dump_json())["components"]) == 1


def test_source_hashes_unchanged():
    before = json.loads((ROOT / "rebuild/manifests/source_hashes_stage1_before.json").read_text(encoding="utf-8"))
    after = json.loads((ROOT / "rebuild/manifests/source_hashes_stage1_after.json").read_text(encoding="utf-8"))
    assert before == after


def test_state_database_restart_and_queue_determinism():
    db = ROOT / "state/pipeline_state.sqlite"
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == len(expected_preserved_json_paths())
        assert conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0] == expected_medication_source_count()
        task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    finally:
        conn.close()
    queue_rows = 0
    for name in ["deidentification_queue.csv", "annotation_queue_existing.csv", "layer_a_ready_for_normalization.csv"]:
        with (ROOT / "rebuild/queues" / name).open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len({row["task_id"] for row in rows}) == len(rows)
        queue_rows += len(rows)
    assert task_count == queue_rows


def test_duplicate_group_behavior_reported():
    with (ROOT / "rebuild/manifests/duplicate_groups.csv").open(newline="", encoding="utf-8") as fh:
        duplicate_groups = list(csv.DictReader(fh))
    assert len(duplicate_groups) == 47
    report = (ROOT / "rebuild/reports/DUPLICATE_INFERENCE_SAVINGS_ESTIMATE.md").read_text(encoding="utf-8")
    assert "potential_duplicate_savings" in report
