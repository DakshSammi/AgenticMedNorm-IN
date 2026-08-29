from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "derived/retrieval"
CACHE = ROOT / "knowledge/cache/stage2c"


def _summary() -> dict:
    return json.loads((DERIVED / "stage2c_summary.json").read_text(encoding="utf-8"))


def _trace() -> pd.DataFrame:
    return pd.read_csv(DERIVED / "stage2c_branch_traces.csv", dtype=str).fillna("")


def test_stage2c_outputs_cover_all_mentions_and_branches():
    summary = _summary()
    trace = _trace()
    layer_a = pd.read_csv(ROOT / "derived/layer_a_medication_mentions.csv", dtype=str).fillna("")
    assert summary["mentions_processed"] == layer_a["mention_id"].nunique()
    assert summary["unique_surfaces"] > 100
    assert set(trace["branch"]) == {"R1_EXACT_FUZZY", "R2_BM25", "R3_BIOMEDICAL_DENSE", "R4_RXNORM", "R5_INDIA_KB"}
    assert trace.groupby(["mention_id", "branch"]).size().groupby("mention_id").size().eq(5).all()


def test_r1_exact_fuzzy_is_deterministic_and_records_match_fields():
    trace = _trace()
    r1 = trace[(trace["branch"] == "R1_EXACT_FUZZY") & (trace["candidate_id"] != "")]
    assert not r1.empty
    assert r1["score_semantics"].str.contains("rapidfuzz|exact", case=False).all()
    assert r1["matched_field"].str.len().gt(0).all()
    assert r1["matched_alias"].str.len().gt(0).all()


def test_r2_bm25_is_deterministic_and_index_is_persisted():
    summary = _summary()
    trace = _trace()
    r2 = trace[(trace["branch"] == "R2_BM25") & (trace["candidate_id"] != "")]
    assert not r2.empty
    assert r2["score_semantics"].eq("bm25_okapi_unbounded").all()
    assert Path(summary["paths"]["bm25_index"]).exists()


def test_r3_dense_metadata_uses_true_encoder_or_explicit_unavailable_state():
    meta = json.loads((CACHE / "dense_metadata.json").read_text(encoding="utf-8"))
    assert meta["char_ngram_used"] is False
    if meta["available"]:
        assert "SapBERT" in meta["model_name"]
        assert meta["model_type"] == "true_biomedical_transformer_encoder"
        assert int(meta["embedding_dimension"]) > 0
        assert "char" not in meta["pooling_method"].lower()
    else:
        assert meta["unavailable_reason"]
        trace = _trace()
        r3 = trace[trace["branch"] == "R3_BIOMEDICAL_DENSE"]
        assert set(r3["status"]) <= {"UNAVAILABLE"}


def test_char_ngram_cannot_be_labelled_biomedical_dense():
    source = (ROOT / "src/retrieval/stage2c.py").read_text(encoding="utf-8")
    forbidden = ["char_ngram_dense", "TfidfVectorizer", "analyzer=\"char\"", "analyzer='char'"]
    assert not any(token in source for token in forbidden)


def test_r4_rxnorm_cache_exists_and_candidates_are_rxnorm_concepts():
    trace = _trace()
    r4 = trace[(trace["branch"] == "R4_RXNORM") & (trace["candidate_id"] != "")]
    assert (CACHE / "rxnav").exists()
    if not r4.empty:
        assert r4["candidate_id"].str.startswith("RXCUI_").all()
        assert r4["source_state"].eq("RXNORM_CONCEPT").all()


def test_r5_preserves_authority_and_quarantine_labels():
    trace = _trace()
    r5 = trace[(trace["branch"] == "R5_INDIA_KB") & (trace["candidate_id"] != "")]
    assert not r5.empty
    assert r5["source_state"].str.len().gt(0).all()
    assert "CANDIDATE_QUARANTINE" in set(r5["source_state"])


def test_branch_failure_isolation_statuses_are_per_branch():
    trace = _trace()
    statuses = set(trace["status"])
    assert statuses <= {"SUCCESS", "EMPTY", "FAILED", "UNAVAILABLE"}
    per_mention = trace.groupby("mention_id")["branch"].nunique()
    assert per_mention.eq(5).all()


def test_candidate_ids_are_valid_known_prefixes():
    trace = _trace()
    candidates = trace[trace["candidate_id"] != ""]["candidate_id"]
    allowed_prefixes = ("BPROD_", "BFAM_", "ING_", "FORM_", "CDSCO_", "NLEM_", "RXCUI_")
    assert not candidates.empty
    assert candidates.map(lambda value: value.startswith(allowed_prefixes)).all()


def test_true_union_is_set_union_and_not_ranked():
    trace = _trace()
    union = pd.read_csv(DERIVED / "stage2c_candidate_union.csv", dtype=str).fillna("")
    for mention_id, group in trace[trace["candidate_id"] != ""].groupby("mention_id"):
        expected = set(group["candidate_id"])
        actual = set(union[union["mention_id"] == mention_id]["candidate_id"])
        assert actual == expected
    assert set(union["is_ranked"]) <= {"false"}
    assert "rank" not in union.columns


def test_stage2c_does_not_perform_ranking_or_gold_metric_reporting():
    summary = _summary()
    assert summary["ranking_performed"] is False
    assert summary["gold_metrics_reported"] is False
    report = (ROOT / "rebuild/reports/STAGE2C_CANDIDATE_RETRIEVAL_REPORT.md").read_text(encoding="utf-8")
    forbidden = ["Recall@K:", "MRR:", "accuracy:"]
    assert not any(token in report for token in forbidden)


def test_no_kb_mutation_and_no_prescription_driven_source_acquisition():
    summary = _summary()
    hashes = json.loads((DERIVED / "stage2c_kb_input_hashes.json").read_text(encoding="utf-8"))
    assert summary["kb_inputs_unchanged"] is True
    assert hashes["unchanged"] is True
    assert summary["prescription_driven_source_acquisition"] is False
    source = (ROOT / "src/retrieval/stage2c.py").read_text(encoding="utf-8")
    assert "knowledge/raw/nppa" not in source
    assert "knowledge/raw/cdsco" not in source


def test_diagnostic_sample_has_blank_reviewer_fields_and_branch_top10():
    sample = pd.read_csv(ROOT / "review/retrieval_candidate_inspection_100.csv", dtype=str).fillna("")
    assert 80 <= len(sample) <= 100
    for column in ["correct_candidate_present", "best_candidate_id", "branch_helpful", "notes"]:
        assert column in sample.columns
        assert set(sample[column]) == {""}
    for branch in ["R1_EXACT_FUZZY", "R2_BM25", "R3_BIOMEDICAL_DENSE", "R4_RXNORM", "R5_INDIA_KB"]:
        assert f"{branch}_top10" in sample.columns
