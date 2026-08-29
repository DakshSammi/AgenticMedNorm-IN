from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"


def test_open_dataset_freeze_preserves_commit_hash_and_row_count():
    freeze = json.loads((KNOWLEDGE / "provenance/open_indian_dataset_freeze.json").read_text())
    assert freeze["row_count"] == 253973
    assert len(freeze["commit"]) == 40
    assert len(freeze["primary_csv_sha256"]) == 64
    assert freeze["stage2b_policy"] == "retrieval_recall_inventory_only_not_validated_canonical_database"


def test_nppa_detail_pilot_is_bounded_and_does_not_promote_products():
    pilot = pd.read_csv(KNOWLEDGE / "staging/nppa_product_detail_pilot_requests.csv", dtype=str).fillna("")
    details = pd.read_csv(KNOWLEDGE / "canonical/nppa_product_details.csv", dtype=str).fillna("")
    summary = json.loads((KNOWLEDGE / "reports/stage2b_summary.json").read_text())
    assert len(pilot) == 120
    assert details.empty
    assert summary["nppa_endpoint_status"] == "DETAIL_ENDPOINT_NOT_OPERATIONAL_DIRECT_PUBLIC_GET"
    assert summary["supported_products"] == 0
    assert summary["authoritative_products"] == 0


def test_cdsco_structured_extraction_outputs_records_and_fdc_components():
    records = pd.read_csv(KNOWLEDGE / "canonical/cdsco_structured_records.csv", dtype=str).fillna("")
    components = pd.read_csv(KNOWLEDGE / "canonical/cdsco_formulation_components.csv", dtype=str).fillna("")
    failures = pd.read_csv(KNOWLEDGE / "reports/cdsco_parse_failures.csv", dtype=str).fillna("")
    assert len(records) >= 1000
    assert (records["is_fdc"] == "true").sum() >= 100
    assert len(components) >= (records["is_fdc"] == "true").sum()
    assert len(failures) >= 0
    assert records["source_pdf_sha256"].str.len().eq(64).all()


def test_nlem_entries_are_structured_and_not_brand_identity():
    nlem = pd.read_csv(KNOWLEDGE / "canonical/nlem_entries.csv", dtype=str).fillna("")
    assert len(nlem) >= 500
    assert {"ingredient", "strength", "dosage_form", "section_category", "evidence_id"} <= set(nlem.columns)
    assert "brand" not in {column.lower() for column in nlem.columns}


def test_rxnorm_ingredient_enrichment_completed_over_canonical_ingredients():
    ingredients = pd.read_csv(KNOWLEDGE / "canonical/ingredients.csv", dtype=str).fillna("")
    rxnorm = pd.read_csv(KNOWLEDGE / "crosswalks/rxnorm_ingredient_mappings.csv", dtype=str).fillna("")
    assert len(rxnorm) == len(ingredients) == 1716
    assert set(rxnorm["mapping_status"]) <= {"EXACT", "NORMALIZED_SUPPORTED", "APPROXIMATE_REVIEW", "AMBIGUOUS", "NO_MATCH"}
    accepted = rxnorm[rxnorm["mapping_status"].isin(["EXACT", "NORMALIZED_SUPPORTED"])]
    assert accepted["tty"].isin(["IN", "PIN", "MIN"]).all()


def test_approximate_rxnorm_matches_are_not_auto_accepted():
    rxnorm = pd.read_csv(KNOWLEDGE / "crosswalks/rxnorm_ingredient_mappings.csv", dtype=str).fillna("")
    approx = rxnorm[rxnorm["mapping_status"] == "APPROXIMATE_REVIEW"]
    assert not approx.empty
    assert not approx["mapping_status"].isin(["EXACT", "NORMALIZED_SUPPORTED"]).any()


def test_atc_mapping_is_partial_and_provenanced():
    atc = pd.read_csv(KNOWLEDGE / "crosswalks/rxclass_atc_mappings.csv", dtype=str).fillna("")
    assert len(atc) > 0
    assert atc["evidence_id"].str.startswith("EVID_").all()


def test_open_nppa_validation_l3_l5_stay_not_comparable_without_product_detail():
    validation = pd.read_csv(KNOWLEDGE / "reports/open_vs_nppa_validation_stage2b.csv", dtype=str).fillna("")
    assert len(validation) == 253973
    assert set(validation["l3_composition"]) == {"NOT_COMPARABLE"}
    assert set(validation["l4_pack_or_sku"]) == {"NOT_COMPARABLE"}
    assert set(validation["l5_price"]) == {"NOT_COMPARABLE"}
    assert set(validation["promotion_decision"]) == {"NO_PROMOTION_NPPA_DETAIL_UNAVAILABLE"}


def test_kb_qc_sheet_has_blank_reviewer_columns():
    qc = pd.read_csv(ROOT / "review/kb_qc_150.csv", dtype=str).fillna("")
    assert len(qc) == 150
    reviewer_columns = [
        "brand_correct",
        "ingredient_correct",
        "strength_correct",
        "form_correct",
        "fdc_correct",
        "company_correct",
        "source_evidence_correct",
        "overall_status",
        "notes",
    ]
    for column in reviewer_columns:
        assert column in qc.columns
        assert set(qc[column]) == {""}


def test_paper_source_matrix_uses_allowed_statuses_and_downgrades_incomplete_sources():
    matrix = pd.read_csv(KNOWLEDGE / "reports/PAPER_SOURCE_IMPLEMENTATION_MATRIX.csv", dtype=str).fillna("")
    allowed = {"FULLY_IMPLEMENTED", "PARTIALLY_IMPLEMENTED", "REGISTERED_ONLY", "NOT_IMPLEMENTED", "REMOVE_FROM_PAPER"}
    assert set(matrix["final_status"]) <= allowed
    nppa = matrix[matrix["paper_source"] == "NPPA Pharma Sahi Daam"].iloc[0]
    assert nppa["final_status"] == "PARTIALLY_IMPLEMENTED"
    assert "do not claim product-detail integration" in nppa["paper_text_action"].lower()
    mims = matrix[matrix["paper_source"] == "MIMS/CIMS India"].iloc[0]
    assert mims["final_status"] == "REMOVE_FROM_PAPER"


def test_stage2b_does_not_read_prescription_mention_outputs():
    source = (ROOT / "src/knowledge/stage2b.py").read_text(encoding="utf-8")
    forbidden = ["layer_a_medication_mentions", "ground_truths_json", "prescription strings"]
    assert not any(token in source for token in forbidden)


def test_sqlite_stage2b_views_are_queryable():
    conn = sqlite3.connect(KNOWLEDGE / "medication_knowledge.sqlite")
    try:
        for view in ["v_rxnorm_ingredient_support", "v_nlem_entries", "v_cdsco_evidence"]:
            assert conn.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0] > 0
    finally:
        conn.close()

