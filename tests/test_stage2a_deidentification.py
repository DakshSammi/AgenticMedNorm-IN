from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.deidentification.stage2a import (
    ROOT,
    RUN_ID,
    STATUS_NEEDS_REVIEW,
    derive_historical_mask,
    layout_signature,
    pilot_candidates,
    predict_mask,
    sha256_file,
)


def test_deterministic_mask_generation_on_synthetic_pair():
    raw = np.full((300, 220, 3), 245, dtype=np.uint8)
    cv2.putText(raw, "Name: Test", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    anon = raw.copy()
    cv2.rectangle(anon, (15, 20), (170, 62), (0, 0, 0), thickness=-1)
    mask1, boxes1, status1 = derive_historical_mask(raw, anon)
    mask2, boxes2, status2 = derive_historical_mask(raw, anon)
    assert np.array_equal(mask1, mask2)
    assert boxes1 == boxes2
    assert status1 == status2 == "ATTRIBUTABLE_TO_DEID"


def test_layout_assignment_is_deterministic():
    image = np.full((1600, 1200, 3), 255, dtype=np.uint8)
    assert layout_signature(image) == layout_signature(image.copy())
    assert layout_signature(image)[3] == "portrait_1200x1600"


def test_source_images_were_not_mutated_by_pilot():
    results = pd.read_csv(ROOT / "derived/deid_stage2a/pilot_results.csv", dtype=str).fillna("")
    assert len(results) == 25
    for _, row in results.iterrows():
        metadata = json.loads(row["redaction_metadata_json"])
        assert metadata["raw_sha256_before"] == metadata["raw_sha256_after"]


def test_no_cloud_network_dependency_in_stage2a_agent():
    source = (ROOT / "src/deidentification/stage2a.py").read_text(encoding="utf-8").lower()
    forbidden = ["import requests", "openai.", "gemini.", "claude.", "http://", "https://"]
    assert not any(token in source for token in forbidden)


def test_duplicate_reuse_provenance_is_explicit():
    dup_path = ROOT / "derived/deid_stage2a/duplicate_reuse_provenance.csv"
    rows = pd.read_csv(dup_path, dtype=str).fillna("")
    assert len(rows) >= 1
    assert rows["reuse_allowed"].isin(["True", "true"]).all()
    assert rows["derived_from_duplicate_representative"].ne("").all()
    assert rows["raw_sha256"].str.len().eq(64).all()


def test_no_stage2a_outputs_written_into_raw_or_anonymized_roots():
    results = pd.read_csv(ROOT / "derived/deid_stage2a/pilot_results.csv", dtype=str).fillna("")
    for output_path in results["output_path"]:
        assert output_path.startswith("generated/anonymized_stage2a_pilot/")
        assert not output_path.startswith("prescription_pipeline_jbhi_ieee/raw/")
        assert not output_path.startswith("prescription_pipeline_jbhi_ieee/anonymized/")


def test_output_hash_generation_matches_filesystem():
    results = pd.read_csv(ROOT / "derived/deid_stage2a/pilot_results.csv", dtype=str).fillna("")
    for _, row in results.iterrows():
        output = ROOT / row["output_path"]
        assert output.exists()
        assert sha256_file(output) == row["output_sha256"]


def test_resumability_state_records_are_insert_replace_safe():
    conn = sqlite3.connect(ROOT / "state/pipeline_state.sqlite")
    try:
        run = conn.execute("SELECT status FROM stage_runs WHERE run_id = ?", (RUN_ID,)).fetchone()
        assert run and run[0] == "SUCCESS"
        artifact_count = conn.execute("SELECT COUNT(*) FROM artifacts WHERE run_id = ?", (RUN_ID,)).fetchone()[0]
        task_count = conn.execute("SELECT COUNT(*) FROM tasks WHERE stage_name = 'stage2a_deidentification_pilot'").fetchone()[0]
        assert artifact_count == 25
        assert task_count == 0
    finally:
        conn.close()


def test_unusual_layout_routes_to_review():
    image = np.full((333, 777, 3), 255, dtype=np.uint8)
    mask, profile, reasons = predict_mask(image, profiles={})
    assert profile == "LAYOUT_UNASSIGNED"
    assert "UNUSUAL_LAYOUT" in reasons
    assert np.count_nonzero(mask) == 0


def test_pilot_selection_is_unique_and_capped():
    selected = pilot_candidates(profiles={}, count=25)
    assert len(selected) == 25
    assert selected["raw_sha256"].is_unique


def test_qc_sheet_has_blank_reviewer_fields():
    qc = pd.read_csv(ROOT / "review/deid_qc_sample.csv", dtype=str).fillna("")
    assert 50 <= len(qc) <= 75
    reviewer_fields = [
        "all_identifiers_masked",
        "medication_content_preserved",
        "over_redaction_problem",
        "under_redaction_problem",
        "overall_pass",
        "notes",
    ]
    for field in reviewer_fields:
        assert field in qc.columns
        assert set(qc[field]) == {""}
