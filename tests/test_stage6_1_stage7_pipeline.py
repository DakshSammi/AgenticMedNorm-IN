from __future__ import annotations

import json
import hashlib
import uuid
import inspect
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import scripts.run_layer_a_normalization_batch as batch
import scripts.process_new_prescriptions as incoming
from scripts.process_new_prescriptions import process_incoming
from src.pipeline.orchestrator import PipelineOrchestrator, PipelineStatus


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, label: str = "TAB AZEE") -> None:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.putText(image, "Patient Name", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, label[:18], (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(str(path), image)


def test_stage6_1_brand_family_audit_outputs() -> None:
    summary = json.loads((ROOT / "derived/verification/verification_summary_v1_1.json").read_text(encoding="utf-8"))
    assert summary["nppa_brand_family_evidence_linkage_operational"] is True
    assert summary["verifier_bug_found"] is True
    assert summary["layer_b_v1_1_created"] is True
    assert summary["context_evidence_implemented"] is False
    assert summary["brand_family_accept_v1"] == 0
    assert summary["brand_family_accept_v1_1"] > 0


def test_layer_b_v1_1_null_unsupported_ids() -> None:
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1_1.csv", dtype=str).fillna("")
    accepted = layer_b[layer_b["verification_decision"] == "ACCEPT"]
    unsupported_exact = accepted[
        (accepted["resolution_level"] == "EXACT_LOCAL_PRODUCT")
        & ((accepted["local_brand_product_id"] == "") | (accepted["supporting_evidence_ids_json"] == "[]"))
    ]
    assert unsupported_exact.empty
    assert layer_b[(layer_b["verification_decision"] == "NIL") & (layer_b["primary_candidate_id"] != "")].empty
    assert not layer_b["drugbank_id"].astype(str).str.len().gt(0).any()
    assert not layer_b["pubchem_cid"].astype(str).str.len().gt(0).any()


def test_online_layer_a_to_layer_b_transitions_and_provenance() -> None:
    result = PipelineOrchestrator(ROOT).process_mention_text("TAB AZEE")
    assert result.status == PipelineStatus.SUCCESS
    expected = {
        "layer_a.json",
        "retrieval_trace.csv",
        "ranked_candidates.csv",
        "evidence_trace.csv",
        "verification_results.csv",
        "layer_b_normalized.json",
        "pipeline_provenance.json",
    }
    assert expected <= {path.name for path in result.output_dir.iterdir()}
    provenance = json.loads((result.output_dir / "pipeline_provenance.json").read_text(encoding="utf-8"))
    assert provenance["states"] == ["LAYER_A", "RETRIEVED", "RANKED", "EVIDENCE_ASSESSED", "VERIFIED", "LAYER_B"]
    assert provenance["status"] == "SUCCESS"


def test_online_no_new_candidate_invariant() -> None:
    result = PipelineOrchestrator(ROOT).process_mention_text("TAB AZEE")
    trace = pd.read_csv(result.output_dir / "retrieval_trace.csv", dtype=str).fillna("")
    ranked = pd.read_csv(result.output_dir / "ranked_candidates.csv", dtype=str).fillna("")
    assert set(ranked["candidate_id"]) <= set(trace[trace["candidate_id"] != ""]["candidate_id"])


def test_online_evidence_to_verification_preserves_selected_hard_conflict() -> None:
    row = pd.read_csv(ROOT / "derived/evidence/evidence_assessments.csv", dtype=str).fillna("")
    hard_row = row[row["hard_conflicts_json"] != "[]"].sort_values("ranking_position").iloc[0].copy()
    mention = {"mention_id": hard_row["mention_id"], "document_uid": "DOC_TEST", "page_uid": "PAGE_TEST", "raw_medication_text": hard_row["raw_medication_text"]}
    verification, layer_b = PipelineOrchestrator(ROOT).verify(pd.DataFrame([hard_row]), mention)
    assert verification.iloc[0]["verification_decision"] == "HUMAN_REVIEW"
    assert verification.iloc[0]["hard_conflicts_json"] != "[]"
    assert layer_b[0]["hard_conflicts_json"] != "[]"


def test_online_index_reuse() -> None:
    orch = PipelineOrchestrator(ROOT)
    orch._load_retrieval()
    first = (id(orch.r2), id(orch.r3), id(orch.r3.index))
    orch._load_retrieval()
    second = (id(orch.r2), id(orch.r3), id(orch.r3.index))
    assert first == second


def test_deidentified_image_blocks_cleanly_without_annotation_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANNOTATION_API_KEY", raising=False)
    img = tmp_path / "deidentified.png"
    _write_image(img)
    result = PipelineOrchestrator(ROOT).process_deidentified_image(img)
    assert result.status == PipelineStatus.BLOCKED
    assert result.error_code == "CREDENTIALS_MISSING"
    provenance = json.loads((result.output_dir / "pipeline_provenance.json").read_text(encoding="utf-8"))
    assert provenance["blocked_at"] == "ANNOTATED"


def test_raw_image_deidentifies_then_blocks_without_annotation_credentials(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANNOTATION_API_KEY", raising=False)
    img = tmp_path / "raw.png"
    _write_image(img)
    result = PipelineOrchestrator(ROOT).process_raw_image(img)
    assert result.status == PipelineStatus.BLOCKED
    deid = json.loads((result.output_dir / "deidentification_result.json").read_text(encoding="utf-8"))
    assert deid["status"] == "SUCCESS"
    assert deid["raw_image_copied"] is False
    assert Path(deid["deidentified_image_path"]).exists()


def test_incoming_workflow_duplicate_input_reuses_previous_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANNOTATION_API_KEY", raising=False)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    img = incoming / "rx.png"
    _write_image(img, label=f"TAB AZEE {uuid.uuid4().hex[:6]}")
    first = process_incoming(ROOT, incoming, limit=1)
    second = process_incoming(ROOT, incoming, limit=1)
    assert first["processed_new"] == 1
    assert second["reused_previous"] == 1
    assert second["records"][0]["idempotency"] == "REUSED_PREVIOUS_RESULT"


def test_layer_a_incremental_batch_resume_skips_unchanged(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path
    (root / "rebuild/queues").mkdir(parents=True)
    (root / "derived/layer_a").mkdir(parents=True)
    queue = root / "rebuild/queues/layer_a_ready_for_normalization.csv"
    queue.write_text("task_id,mention_id,document_uid,page_uid,status\nTASK_1,MENT_1,DOC_1,PAGE_1,PENDING\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "mention_id": "MENT_1",
                "document_uid": "DOC_1",
                "page_uid": "PAGE_1",
                "raw_medication_text": "TAB AZEE",
                "raw_strength_text": "",
                "raw_dosage_text": "",
                "raw_frequency_text": "",
                "raw_duration_text": "",
                "raw_route_text": "",
                "raw_timing_text": "",
                "context_bundle_id": "",
            }
        ]
    ).to_csv(root / "derived/layer_a_medication_mentions.csv", index=False)

    class FakeOrchestrator:
        def __init__(self, root_path: Path) -> None:
            self.state_db = root_path / "state/pipeline_orchestrator.sqlite"
            self.state_db.parent.mkdir(parents=True, exist_ok=True)

        def process_mentions(self, mentions, document_uid=None, mode="layer_a_batch"):
            class Result:
                status = PipelineStatus.SUCCESS
                document_uid = "DOC_1"
                layer_b_records = [{"mention_id": "MENT_1", "document_uid": "DOC_1", "raw_medication_text": mentions[0]["raw_medication_text"]}]

            return Result()

    monkeypatch.setattr(batch, "PipelineOrchestrator", FakeOrchestrator)
    first = batch.run_incremental_batch(root, queue, "online_incremental", limit=1)
    second = batch.run_incremental_batch(root, queue, "online_incremental", limit=1)
    assert first["processed_new_or_changed"] == 1
    assert second["processed_new_or_changed"] == 0
    assert second["skipped_unchanged"] == 1


def test_stage7_smoke_and_full_consistency_reports_clean() -> None:
    smoke = json.loads((ROOT / "derived/pipeline/stage7_smoke_summary.json").read_text(encoding="utf-8"))
    consistency = json.loads((ROOT / "derived/pipeline/full_dataset_consistency_audit.json").read_text(encoding="utf-8"))
    assert smoke["ready_for_operational_use"] is True
    assert smoke["gold_metrics_reported"] is False
    assert consistency["blockers"] == []
    assert consistency["gold_metrics_reported"] is False


def test_stage7_1_accepted_brand_family_provenance_chain() -> None:
    audit = pd.read_csv(ROOT / "derived/audit/accepted_brand_family_provenance.csv", dtype=str).fillna("")
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1_1.csv", dtype=str).fillna("")
    expected = int(((layer_b["verification_decision"] == "ACCEPT") & (layer_b["resolution_level"] == "LOCAL_BRAND_FAMILY")).sum())
    assert len(audit) == expected
    assert (audit["provenance_valid"] == "TRUE").all()
    assert (audit["nppa_evidence_id"] == "NPPA_BRAND_INDEX_L1_MATCH").all()
    assert set(audit["relation_type"]) <= {"PRIMARY_ID_DIRECTLY_ASSESSED", "PRIMARY_ID_DERIVED_VIA_SUPPORTED_RELATION"}


def test_stage7_1_direct_vs_derived_selected_ids_reported() -> None:
    consistency = json.loads((ROOT / "derived/pipeline/full_dataset_consistency_audit.json").read_text(encoding="utf-8"))
    provenance = consistency["selected_id_provenance"]
    assert provenance["without_direct_or_supported_relation_count"] == 0
    assert provenance["derived_via_supported_relation_count"] > 0
    assert provenance["directly_assessed_count"] > 0


def test_stage7_1_decision_specific_resolution_reporting() -> None:
    consistency = json.loads((ROOT / "derived/pipeline/full_dataset_consistency_audit.json").read_text(encoding="utf-8"))
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1_1.csv", dtype=str).fillna("")
    for key in [
        "resolution_distribution_all",
        "resolution_distribution_ACCEPT",
        "resolution_distribution_HUMAN_REVIEW",
        "resolution_distribution_NIL",
    ]:
        assert key in consistency
    assert consistency["accepted_exact_local_product"] == 0
    expected_review_exact = int(((layer_b["verification_decision"] == "HUMAN_REVIEW") & (layer_b["resolution_level"] == "EXACT_LOCAL_PRODUCT")).sum())
    assert consistency["proposed_review_exact_local_product"] == expected_review_exact
    assert "EXACT_LOCAL_PRODUCT" not in consistency["resolution_distribution_ACCEPT"]
    assert consistency["resolution_distribution_HUMAN_REVIEW"]["EXACT_LOCAL_PRODUCT"] == expected_review_exact


def test_stage7_1_operational_manifest_and_hashes() -> None:
    manifest = json.loads((ROOT / "configs/frozen/operational_v1_manifest.json").read_text(encoding="utf-8"))
    assert manifest["operational_version_identifier"] == "OPERATIONAL_V1"
    assert manifest["architecture_frozen"] is True
    assert manifest["secrets_included"] is False
    assert manifest["context_evidence_implemented"] is False
    for key in ["bm25_index", "faiss_index", "retrieval_documents"]:
        path = ROOT / manifest[key]["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert manifest[key]["sha256"] == digest
    assert manifest["components"]["ranking"]["config_sha256"] == hashlib.sha256((ROOT / "configs/ranking/stage4_ranking_config.json").read_bytes()).hexdigest()
    assert manifest["components"]["evidence"]["config_sha256"] == hashlib.sha256((ROOT / "configs/evidence/stage5_evidence_config.json").read_bytes()).hexdigest()
    assert manifest["components"]["verification"]["config_sha256"] == hashlib.sha256((ROOT / "configs/verification/stage6_verification_config.json").read_bytes()).hexdigest()
    assert len(manifest["manifest_hash"]) == 64


def test_stage7_1_resource_manifest_truthfulness() -> None:
    resources = pd.read_csv(ROOT / "knowledge/reports/OPERATIONAL_RESOURCE_MANIFEST.csv", dtype=str).fillna("")
    by_resource = resources.set_index("resource").to_dict("index")
    for resource in ["NPPA", "CDSCO", "NLEM", "open Indian medicine dataset", "RxNorm/RxNav", "ATC/RxClass"]:
        assert by_resource[resource]["implemented"] == "TRUE"
        assert by_resource[resource]["used_in_operational_v1"] == "TRUE"
    for resource in ["BODHI-M", "PMBI", "DrugBank", "PubChem"]:
        assert by_resource[resource]["used_in_operational_v1"] == "FALSE"
    assert by_resource["BODHI-M"]["implemented"] == "FALSE"


def test_stage7_1_online_incoming_and_batch_use_same_orchestrator() -> None:
    assert "PipelineOrchestrator" in inspect.getsource(incoming.process_incoming)
    assert "PipelineOrchestrator" in inspect.getsource(batch.run_incremental_batch)
    assert "process_mentions" in inspect.getsource(batch.run_incremental_batch)
    assert "process_raw_image" in inspect.getsource(incoming.process_incoming)
