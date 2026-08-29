from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ranking.stage4 import reciprocal_rank_fusion


ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_stage2c1_candidate_identities_are_canonical_entity_keys():
    trace = pd.read_csv(ROOT / "derived/retrieval/stage2c1_branch_traces.csv", dtype=str).fillna("")
    candidates = trace[trace["candidate_id"] != ""]
    assert not candidates.empty
    assert candidates["candidate_id"].str.startswith("ENTITY:").all()
    assert candidates["entity_type"].isin(
        ["BRAND_PRODUCT", "BRAND_FAMILY", "INGREDIENT", "CLINICAL_FORMULATION", "RXNORM_CONCEPT", "OFFICIAL_SOURCE_RECORD"]
    ).all()
    assert candidates["raw_candidate_id"].str.len().gt(0).all()


def test_stage2c1_collapses_aliases_and_source_records_without_false_rxnorm_merge():
    summary = _json("derived/retrieval/stage2c1_summary.json")
    collapse = pd.read_csv(ROOT / "derived/retrieval/stage2c1_identity_collapse_log.csv", dtype=str).fillna("")
    trace = pd.read_csv(ROOT / "derived/retrieval/stage2c1_branch_traces.csv", dtype=str).fillna("")
    assert summary["aliases_surfaces_collapsed_rows"] > 0
    assert pd.to_numeric(collapse["collapsed_surface_hit_count"], errors="coerce").max() > 1
    r4 = trace[(trace["branch"] == "R4_RXNORM") & (trace["candidate_id"] != "")]
    assert r4["entity_type"].eq("RXNORM_CONCEPT").all()
    assert r4["candidate_id"].str.startswith("ENTITY:RXNORM_CONCEPT:").all()


def test_r3_namespace_issue_audited_and_explicit_mapping_used():
    summary = _json("derived/retrieval/stage2c1_summary.json")
    trace = pd.read_csv(ROOT / "derived/retrieval/stage2c1_branch_traces.csv", dtype=str).fillna("")
    r3 = trace[(trace["branch"] == "R3_BIOMEDICAL_DENSE") & (trace["candidate_id"] != "")]
    assert summary["r3_namespace_issue_found"] is True
    assert "cdsco_drug_or_single_component_exact_local_ingredient" in set(r3["identity_mapping_basis"]) or "nlem_normalized_ingredient_exact_local_ingredient" in set(r3["identity_mapping_basis"])
    assert "no_supported_entity_crosswalk_preserved_as_source_record" in set(r3["identity_mapping_basis"])


def test_r2_r5_near_duplication_is_fixed_in_stage2c1():
    summary = _json("derived/retrieval/stage2c1_summary.json")
    before = pd.read_csv(ROOT / "derived/retrieval/stage2c_pairwise_overlap.csv", dtype=str).fillna("")
    after = pd.read_csv(ROOT / "derived/retrieval/stage2c1_pairwise_overlap.csv", dtype=str).fillna("")
    before_j = float(before[(before["branch_a"] == "R2_BM25") & (before["branch_b"] == "R5_INDIA_KB")]["jaccard"].iloc[0])
    after_j = float(after[(after["branch_a"] == "R2_BM25") & (after["branch_b"] == "R5_INDIA_KB")]["jaccard"].iloc[0])
    assert summary["r2_r5_near_duplication_found"] is True
    assert summary["r2_r5_independence_status"] == "R5_REFACTORED_STAGE2C1_STRUCTURED_INDIA_KB_NOT_R2_BM25"
    assert before_j > 0.9
    assert after_j < 0.5
    r5 = pd.read_csv(ROOT / "derived/retrieval/stage2c1_branch_traces.csv", dtype=str).fillna("")
    r5 = r5[(r5["branch"] == "R5_INDIA_KB") & (r5["candidate_id"] != "")]
    assert r5["matched_field"].str.startswith("india_structured.").all()


def test_stage2c1_union_is_set_union_of_canonicalized_trace():
    trace = pd.read_csv(ROOT / "derived/retrieval/stage2c1_branch_traces.csv", dtype=str).fillna("")
    union = pd.read_csv(ROOT / "derived/retrieval/stage2c1_candidate_union.csv", dtype=str).fillna("")
    sample_mentions = union["mention_id"].drop_duplicates().head(25)
    for mention_id in sample_mentions:
        expected = set(trace[(trace["mention_id"] == mention_id) & (trace["candidate_id"] != "")]["candidate_id"])
        actual = set(union[union["mention_id"] == mention_id]["candidate_id"])
        assert actual == expected


def test_rrf_formula_is_correct_and_deterministic():
    score1, components1 = reciprocal_rank_fusion({"R1_EXACT_FUZZY": 1, "R3_BIOMEDICAL_DENSE": 4}, rrf_k=60)
    score2, components2 = reciprocal_rank_fusion({"R3_BIOMEDICAL_DENSE": 4, "R1_EXACT_FUZZY": 1}, rrf_k=60)
    expected = (1 / 61) + (1 / 64)
    assert score1 == score2
    assert components1 == components2
    assert abs(score1 - expected) < 1e-12


def test_ranking_outputs_are_subset_of_stage2c1_union_and_top20():
    union = pd.read_csv(ROOT / "derived/retrieval/stage2c1_candidate_union.csv", dtype=str).fillna("")
    ranked = pd.read_csv(ROOT / "derived/ranking/ranked_candidates.csv", dtype=str).fillna("")
    assert set(ranked["candidate_id"]) <= set(union["candidate_id"])
    assert pd.to_numeric(ranked["final_rank"], errors="coerce").max() <= 20
    assert ranked.groupby("mention_id")["final_rank"].nunique().max() <= 20


def test_ranking_preserves_branch_scores_and_provenance():
    ranked = pd.read_csv(ROOT / "derived/ranking/ranked_candidates.csv", dtype=str).fillna("")
    assert not ranked.empty
    for column in ["participating_branches_json", "per_branch_rank_json", "per_branch_score_json", "ranking_components_json"]:
        parsed = ranked[column].head(50).map(json.loads)
        assert parsed.map(lambda value: isinstance(value, dict | list)).all()
    assert ranked["source_state"].str.len().gt(0).all()
    assert ranked["raw_candidate_ids"].str.len().gt(0).all()


def test_ranking_does_not_retrieve_or_report_gold_metrics():
    summary = _json("derived/ranking/ranking_summary.json")
    report = (ROOT / "rebuild/reports/STAGE4_CANDIDATE_RANKING_REPORT.md").read_text(encoding="utf-8")
    assert summary["retrieved_new_candidates"] is False
    assert summary["source_authority_acceptance_performed"] is False
    assert summary["gold_metrics_reported"] is False
    assert summary["READY_FOR_EVIDENCE_ASSESSMENT"] is True
    assert not any(token in report for token in ["Accuracy:", "MRR:", "NDCG:"])


def test_parquet_ranking_output_is_readable():
    parquet = pd.read_parquet(ROOT / "derived/ranking/ranking_results.parquet")
    csv = pd.read_csv(ROOT / "derived/ranking/ranked_candidates.csv", dtype=str).fillna("")
    layer_a = pd.read_csv(ROOT / "derived/layer_a_medication_mentions.csv", dtype=str).fillna("")
    assert len(parquet) == len(csv)
    assert len(parquet) == layer_a["mention_id"].nunique() * 20
