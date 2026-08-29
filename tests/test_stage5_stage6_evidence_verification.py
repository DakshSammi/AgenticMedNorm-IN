from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evidence.stage5 import compare_component_count, compare_dosage_form, compare_strength
from src.verification.stage6 import _adequate_for_accept


ROOT = Path(__file__).resolve().parents[1]


def _j(value: str):
    return json.loads(value)


def test_evidence_agent_does_not_retrieve_or_introduce_candidates():
    ranked = pd.read_csv(ROOT / "derived/ranking/ranked_candidates.csv", dtype=str).fillna("")
    evidence = pd.read_csv(ROOT / "derived/evidence/evidence_assessments.csv", dtype=str).fillna("")
    summary = json.loads((ROOT / "derived/evidence/evidence_summary.json").read_text(encoding="utf-8"))
    source = (ROOT / "src/evidence/stage5.py").read_text(encoding="utf-8")
    assert summary["retrieved_new_candidates"] is False
    assert set(evidence["candidate_id"]) <= set(ranked["candidate_id"])
    forbidden = ["Stage2CRetrievalAgent", "R1ExactFuzzy(", "R2BM25(", "R3BiomedicalDense(", "R4RxNorm(", "StructuredIndiaRetriever("]
    assert not any(token in source for token in forbidden)


def test_evidence_assesses_all_ranked_top20_candidates():
    evidence = pd.read_csv(ROOT / "derived/evidence/evidence_assessments.csv", dtype=str).fillna("")
    ranked = pd.read_csv(ROOT / "derived/ranking/ranked_candidates.csv", dtype=str).fillna("")
    layer_a = pd.read_csv(ROOT / "derived/layer_a_medication_mentions.csv", dtype=str).fillna("")
    assert evidence["mention_id"].nunique() == layer_a["mention_id"].nunique()
    assert len(evidence) == len(ranked)
    assert pd.to_numeric(evidence["ranking_position"], errors="coerce").max() <= 20
    assert {"lexical", "semantic", "formulation", "provenance", "context"} <= set(
        json.loads((ROOT / "derived/evidence/evidence_summary.json").read_text(encoding="utf-8"))["evidence_dimensions_operational"]
    )


def test_strength_missing_is_not_conflict_and_conflict_is_detected():
    assert compare_strength("", ["10mg"])[0] == "NOT_COMPARABLE"
    assert compare_strength("10mg", [])[0] == "NOT_COMPARABLE"
    assert compare_strength("10 mg", ["10mg"])[0] == "MATCH"
    status, reason = compare_strength("10mg", ["20mg"])
    assert status == "CONFLICT"
    assert reason == "STRENGTH_CONFLICT"


def test_dosage_form_contradiction_behaves_correctly():
    assert compare_dosage_form("tab", "tablet")[0] == "MATCH"
    assert compare_dosage_form("", "tablet")[0] == "NOT_COMPARABLE"
    status, reason = compare_dosage_form("tablet", "syrup")
    assert status == "CONFLICT"
    assert reason == "DOSAGE_FORM_CONFLICT"


def test_fdc_component_count_and_component_strengths_are_preserved():
    assert compare_component_count(2, 3) == ("CONFLICT", "COMPONENT_COUNT_CONFLICT")
    evidence = pd.read_csv(ROOT / "derived/evidence/evidence_assessments.csv", dtype=str).fillna("")
    facts = evidence["candidate_facts_json"].map(_j)
    multi = [fact for fact in facts if len(fact.get("components", [])) > 1]
    assert multi
    assert any(len([c.get("normalized_strength", "") for c in fact["components"] if c.get("normalized_strength", "")]) > 1 for fact in multi)


def test_hard_conflict_cannot_be_overridden_by_semantic_or_context_support():
    row = pd.Series(
        {
            "resolution_level": "INGREDIENT_ONLY",
            "entity_type": "INGREDIENT",
            "lexical_status": "MATCH",
            "semantic_status": "MATCH",
            "hard_conflicts_json": json.dumps(["STRENGTH_CONFLICT"]),
            "candidate_facts_json": json.dumps({"source_state": "AUTHORITATIVE_NLEM_CONTEXT"}),
            "provenance_evidence_json": json.dumps({"source_states": ["AUTHORITATIVE_NLEM_CONTEXT"]}),
        }
    )
    ok, reason = _adequate_for_accept(row)
    assert ok is False
    assert reason == "HARD_FORMULATION_CONFLICT"


def test_official_source_record_cannot_become_final_primary_identity():
    verification = pd.read_csv(ROOT / "derived/verification/verification_results.csv", dtype=str).fillna("")
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1.csv", dtype=str).fillna("")
    assert not verification["selected_candidate_id"].str.contains("OFFICIAL_SOURCE_RECORD", regex=False).any()
    assert not layer_b["primary_candidate_id"].str.contains("OFFICIAL_SOURCE_RECORD", regex=False).any()


def test_quarantine_only_brand_product_cannot_automatically_be_exact_product_accept():
    verification = pd.read_csv(ROOT / "derived/verification/verification_results.csv", dtype=str).fillna("")
    exact_accepts = verification[(verification["verification_decision"] == "ACCEPT") & (verification["resolution_level"] == "EXACT_LOCAL_PRODUCT")]
    assert exact_accepts.empty
    summary = json.loads((ROOT / "derived/verification/verification_summary.json").read_text(encoding="utf-8"))
    assert summary["exact_local_product_count"] == 0
    assert summary["quarantine_only_human_review_count"] > 0


def test_rxnorm_is_not_mandatory_for_accept_and_is_retained_when_supported():
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1.csv", dtype=str).fillna("")
    accepted = layer_b[layer_b["verification_decision"] == "ACCEPT"]
    assert not accepted.empty
    assert (accepted["rxnorm_rxcui"].astype(str).str.len() == 0).any()
    assert (layer_b["rxnorm_rxcui"].astype(str).str.len() > 0).any()


def test_context_cannot_establish_brand_identity_or_override_conflict():
    evidence = pd.read_csv(ROOT / "derived/evidence/evidence_assessments.csv", dtype=str).fillna("")
    assert set(evidence["context_status"]) == {"NOT_COMPARABLE"}
    assert evidence["context_implemented"].astype(str).isin(["False", "false", "0"]).all()


def test_nil_has_no_fabricated_candidate_and_unsupported_fields_are_empty():
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1.csv", dtype=str).fillna("")
    nil = layer_b[layer_b["verification_decision"] == "NIL"]
    assert not nil.empty
    assert nil["primary_candidate_id"].eq("").all()
    assert nil["local_brand_product_id"].eq("").all()
    assert nil["ingredient_components_json"].eq("[]").all()
    human_review = layer_b[layer_b["verification_decision"] == "HUMAN_REVIEW"]
    assert human_review["local_brand_product_id"].eq("").all()


def test_every_asserted_final_identifier_has_candidate_provenance():
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1.csv", dtype=str).fillna("")
    asserted = layer_b[layer_b["primary_candidate_id"].astype(str).str.len() > 0]
    assert not asserted.empty
    for row in asserted.itertuples(index=False):
        supporting = set(_j(row.supporting_candidate_ids_json))
        assert row.primary_candidate_id in supporting


def test_verification_is_deterministic_and_reports_no_gold_metrics():
    verification = pd.read_csv(ROOT / "derived/verification/verification_results.csv", dtype=str).fillna("")
    layer_b = pd.read_csv(ROOT / "derived/layer_b/layer_b_v1.csv", dtype=str).fillna("")
    assert verification["mention_id"].is_unique
    assert layer_b["mention_id"].is_unique
    summary = json.loads((ROOT / "derived/verification/verification_summary.json").read_text(encoding="utf-8"))
    report = (ROOT / "rebuild/reports/STAGE6_VERIFICATION_REPORT.md").read_text(encoding="utf-8")
    assert summary["gold_metrics_reported"] is False
    assert not any(token in report for token in ["Accuracy:", "MRR:", "NDCG:"])


def test_stage5_stage6_outputs_and_sanity_review_exist():
    for path in [
        "derived/evidence/evidence_assessments.parquet",
        "derived/evidence/evidence_assessments.csv",
        "derived/verification/verification_results.parquet",
        "derived/verification/verification_results.csv",
        "derived/layer_b/layer_b_v1.parquet",
        "derived/layer_b/layer_b_v1.csv",
        "rebuild/reports/STAGE5_EVIDENCE_ASSESSMENT_REPORT.md",
        "rebuild/reports/STAGE6_VERIFICATION_REPORT.md",
    ]:
        assert (ROOT / path).exists()
    sample = pd.read_csv(ROOT / "review/layer_b_sanity_100.csv", dtype=str).fillna("")
    assert len(sample) == 100
    for column in ["human_decision_correct", "human_resolution_correct", "human_best_candidate_id", "human_notes"]:
        assert column in sample.columns
        assert set(sample[column]) == {""}
