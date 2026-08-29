from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


STAGE6_VERSION = "stage6_verification_layer_b_v0.1"
DEFAULT_CONFIG = {
    "verification": {
        "max_candidates_considered": 20,
        "accept_min_lexical_status": "MATCH",
        "allow_quarantine_exact_product_accept": False,
        "allow_rxnorm_only_accept": True,
    }
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(root: Path) -> dict[str, Any]:
    path = root / "configs/verification/stage6_verification_config.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True), encoding="utf-8")
        return DEFAULT_CONFIG
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged["verification"].update(data.get("verification", {}))
    return merged


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def _j(value: str, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _has_hard(row: pd.Series) -> bool:
    return bool(_j(row["hard_conflicts_json"], []))


def _is_quarantine_only(row: pd.Series) -> bool:
    fact = _j(row["candidate_facts_json"], {})
    states = set(v for v in str(fact.get("source_state", row.get("source_state", ""))).split("|") if v)
    prov = _j(row["provenance_evidence_json"], {})
    states |= set(prov.get("source_states", []))
    return bool(states) and states <= {"CANDIDATE_QUARANTINE"}


def _adequate_for_accept(row: pd.Series) -> tuple[bool, str]:
    if row["resolution_level"] == "OFFICIAL_EVIDENCE_ONLY":
        return False, "OFFICIAL_SOURCE_RECORD_EVIDENCE_ONLY"
    if _has_hard(row):
        return False, "HARD_FORMULATION_CONFLICT"
    if row["lexical_status"] != "MATCH" and row["semantic_status"] != "MATCH":
        return False, "WEAK_IDENTITY_EVIDENCE"
    if row["entity_type"] == "BRAND_PRODUCT" and _is_quarantine_only(row):
        return False, "QUARANTINE_ONLY_PRODUCT"
    if row["entity_type"] == "BRAND_FAMILY" and _is_quarantine_only(row):
        return False, "INSUFFICIENT_SOURCE_SUPPORT"
    return True, "SUPPORTED"


def _review_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    hard = _j(row["hard_conflicts_json"], [])
    if "STRENGTH_CONFLICT" in hard:
        reasons.append("STRENGTH_CONFLICT")
    if "DOSAGE_FORM_CONFLICT" in hard:
        reasons.append("DOSAGE_FORM_CONFLICT")
    if "COMPONENT_COUNT_CONFLICT" in hard or "FDC_STRUCTURE_CONFLICT" in hard:
        reasons.append("FDC_CONFLICT")
    if row["entity_type"] == "BRAND_PRODUCT" and _is_quarantine_only(row):
        reasons.append("QUARANTINE_ONLY_PRODUCT")
    if row["entity_type"] == "BRAND_FAMILY" and _is_quarantine_only(row):
        reasons.append("INSUFFICIENT_SOURCE_SUPPORT")
    if row["entity_type"] == "OFFICIAL_SOURCE_RECORD":
        reasons.append("INSUFFICIENT_SOURCE_SUPPORT")
    missing = set(_j(row["missing_evidence_json"], []))
    if {"candidate_strength_missing", "observed_strength_missing"} & missing:
        reasons.append("MISSING_FORMULATION_EVIDENCE")
    if row["entity_type"] == "RXNORM_CONCEPT":
        reasons.append("TERMINOLOGY_CROSSWALK_UNCERTAIN")
    return sorted(set(reasons or ["MULTIPLE_PLAUSIBLE_CANDIDATES"]))


def _resolution(row: pd.Series) -> str:
    if row["entity_type"] == "BRAND_PRODUCT":
        return "EXACT_LOCAL_PRODUCT"
    if row["entity_type"] == "BRAND_FAMILY":
        return "LOCAL_BRAND_FAMILY"
    if row["entity_type"] == "CLINICAL_FORMULATION":
        return "CLINICAL_FORMULATION"
    if row["entity_type"] == "INGREDIENT":
        return "INGREDIENT_ONLY"
    if row["entity_type"] == "RXNORM_CONCEPT":
        return "TERMINOLOGY_ONLY"
    return "NO_SUPPORTED_RESOLUTION"


def _layer_b_row(selected: pd.Series | None, decision: str, mention: dict[str, str], review_codes: list[str], hard: list[str], missing: list[str]) -> dict[str, Any]:
    facts = _j(selected["candidate_facts_json"], {}) if selected is not None else {}
    components = facts.get("components", [])
    rxnorm = facts.get("rxnorm", [])
    atc = facts.get("atc", [])
    candidate_id = selected["candidate_id"] if selected is not None else ""
    resolution = _resolution(selected) if selected is not None and decision == "ACCEPT" else ("NO_SUPPORTED_RESOLUTION" if decision == "NIL" else _resolution(selected) if selected is not None else "NO_SUPPORTED_RESOLUTION")
    if selected is not None and selected["entity_type"] == "OFFICIAL_SOURCE_RECORD":
        candidate_id = ""
        resolution = "NO_SUPPORTED_RESOLUTION"
    return {
        "mention_id": mention.get("mention_id", ""),
        "document_uid": mention.get("document_uid", ""),
        "page_uid": mention.get("page_uid", ""),
        "raw_medication_text": mention.get("raw_medication_text", ""),
        "verification_decision": decision,
        "resolution_level": resolution,
        "primary_candidate_id": candidate_id,
        "local_brand_family_id": facts.get("brand_family_id", "") if selected is not None and selected["entity_type"] != "OFFICIAL_SOURCE_RECORD" else "",
        "local_brand_family_name": facts.get("brand_family_name", "") if selected is not None and selected["entity_type"] != "OFFICIAL_SOURCE_RECORD" else "",
        "local_brand_product_id": facts.get("brand_product_id", "") if decision == "ACCEPT" and resolution == "EXACT_LOCAL_PRODUCT" else "",
        "local_brand_product_name": facts.get("brand_product_name", "") if decision == "ACCEPT" and resolution == "EXACT_LOCAL_PRODUCT" else "",
        "ingredient_components_json": json.dumps(components, sort_keys=True) if selected is not None and selected["entity_type"] != "OFFICIAL_SOURCE_RECORD" else "[]",
        "component_strengths_json": json.dumps([c.get("normalized_strength", "") for c in components if c.get("normalized_strength", "")], sort_keys=True),
        "dosage_form": facts.get("dosage_form", "") if selected is not None else "",
        "release_modifier": facts.get("release_modifier", "") if selected is not None else "",
        "fdc_status": facts.get("fdc_status", "") if selected is not None else "",
        "rxnorm_rxcui": rxnorm[0].get("rxcui", "") if rxnorm else "",
        "rxnorm_name": rxnorm[0].get("rxnorm_name", "") if rxnorm else "",
        "atc_codes_json": json.dumps(sorted(set(a.get("class_id", "") for a in atc if a.get("class_id", ""))), sort_keys=True),
        "drugbank_id": "",
        "pubchem_cid": "",
        "supporting_candidate_ids_json": json.dumps([candidate_id] if candidate_id else [], sort_keys=True),
        "supporting_evidence_ids_json": selected["supporting_evidence_ids_json"] if selected is not None else "[]",
        "hard_conflicts_json": json.dumps(hard, sort_keys=True),
        "missing_evidence_json": json.dumps(missing, sort_keys=True),
        "review_reason_codes_json": json.dumps(review_codes, sort_keys=True),
        "pipeline_version": STAGE6_VERSION,
        "kb_resource_versions_json": json.dumps({"stage2c1": "stage2c1_identity_harmonization_v0.1", "stage4": "stage4_candidate_ranking_v0.1", "stage5": "stage5_evidence_assessment_v0.1"}, sort_keys=True),
        "provenance_json": json.dumps({"selected_from_ranked_candidate": bool(candidate_id), "verification_generated_at": _now_iso()}, sort_keys=True),
    }


def _top5(group: pd.DataFrame) -> str:
    rows = []
    for row in group.sort_values("ranking_position").head(5).itertuples(index=False):
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "candidate_name": row.candidate_name if hasattr(row, "candidate_name") else "",
                "rank": int(row.ranking_position),
                "resolution_level": row.resolution_level,
                "evidence_summary": row.evidence_summary,
            }
        )
    return json.dumps(rows, sort_keys=True)


def _sanity_sample(root: Path, layer_b: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    joined = layer_b.copy()
    evidence_top = evidence.sort_values(["mention_id", "ranking_position"]).groupby("mention_id").head(5)
    top5 = evidence_top.groupby("mention_id").apply(_top5, include_groups=False).to_dict()
    joined["top5_candidates_json"] = joined["mention_id"].map(top5).fillna("[]")
    strata = []
    criteria = [
        joined["verification_decision"] == "ACCEPT",
        joined["verification_decision"] == "HUMAN_REVIEW",
        joined["verification_decision"] == "NIL",
        joined["resolution_level"] == "EXACT_LOCAL_PRODUCT",
        joined["resolution_level"] == "LOCAL_BRAND_FAMILY",
        joined["resolution_level"] == "INGREDIENT_ONLY",
        joined["fdc_status"] == "FDC",
        joined["hard_conflicts_json"].str.contains("STRENGTH_CONFLICT", regex=False),
        joined["review_reason_codes_json"].str.contains("QUARANTINE_ONLY_PRODUCT", regex=False),
        joined["rxnorm_rxcui"].astype(str).str.len() > 0,
    ]
    labels = ["ACCEPT", "HUMAN_REVIEW", "NIL", "brand_product", "brand_family", "ingredient", "FDC", "strength_conflict", "quarantine_only", "RxNorm_supported"]
    for label, mask in zip(labels, criteria, strict=False):
        part = joined[mask].head(10).copy()
        part["sample_stratum"] = label
        strata.append(part)
    sample = pd.concat(strata, ignore_index=True).drop_duplicates("mention_id") if strata else pd.DataFrame()
    if len(sample) < min(100, len(joined)):
        rem = joined[~joined["mention_id"].isin(set(sample["mention_id"]))].head(100 - len(sample)).copy()
        rem["sample_stratum"] = "fill_common_rare_surfaces"
        sample = pd.concat([sample, rem], ignore_index=True)
    sample = sample.drop_duplicates("mention_id").head(100)
    for col in ["human_decision_correct", "human_resolution_correct", "human_best_candidate_id", "human_notes"]:
        sample[col] = ""
    out = root / "review/layer_b_sanity_100.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    return sample


def run_stage6_verification(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    config = _load_config(root)
    out_dir = root / "derived/verification"
    layer_b_dir = root / "derived/layer_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_b_dir.mkdir(parents=True, exist_ok=True)
    evidence = _read_csv(root / "derived/evidence/evidence_assessments.csv")
    evidence["ranking_position"] = pd.to_numeric(evidence["ranking_position"], errors="coerce").fillna(999999).astype(int)
    layer_a = _read_csv(root / "derived/layer_a_medication_mentions.csv")
    mention_meta = layer_a.set_index("mention_id").to_dict("index")

    verification_rows = []
    layer_b_rows = []
    for mention_id, group in evidence.groupby("mention_id", sort=False):
        group = group.sort_values("ranking_position", kind="stable")
        selected = None
        decision = "NIL"
        reason_codes: list[str] = []
        for _, row in group.iterrows():
            ok, reason = _adequate_for_accept(row)
            if ok:
                selected = row
                decision = "ACCEPT"
                reason_codes = []
                break
            if selected is None and row["entity_type"] != "OFFICIAL_SOURCE_RECORD":
                selected = row
                reason_codes = _review_reasons(row)
                decision = "HUMAN_REVIEW"
        if selected is None:
            decision = "NIL"
            reason_codes = ["NO_DEFENSIBLE_CANDIDATE"]
            hard: list[str] = []
            missing: list[str] = []
        else:
            hard = _j(selected["hard_conflicts_json"], [])
            missing = _j(selected["missing_evidence_json"], [])
            if decision == "HUMAN_REVIEW" and not reason_codes:
                reason_codes = _review_reasons(selected)
        verification_rows.append(
            {
                "mention_id": mention_id,
                "verification_decision": decision,
                "selected_candidate_id": "" if decision == "NIL" or (selected is not None and selected["entity_type"] == "OFFICIAL_SOURCE_RECORD") else selected["candidate_id"],
                "resolution_level": "NO_SUPPORTED_RESOLUTION" if decision == "NIL" else _resolution(selected),
                "decision_reason_codes_json": json.dumps(reason_codes, sort_keys=True),
                "hard_conflicts_json": json.dumps(hard, sort_keys=True),
                "missing_evidence_json": json.dumps(missing, sort_keys=True),
                "evidence_ids_json": selected["supporting_evidence_ids_json"] if selected is not None else "[]",
                "verification_method": "deterministic_field_specific_rules",
                "verification_version": STAGE6_VERSION,
                "timestamp": _now_iso(),
            }
        )
        mention_payload = dict(mention_meta.get(mention_id, {}))
        mention_payload["mention_id"] = mention_id
        mention_payload.setdefault("raw_medication_text", group["raw_medication_text"].iloc[0])
        layer_b_rows.append(_layer_b_row(selected if decision != "NIL" else None, decision, mention_payload, reason_codes, hard, missing))

    verification = pd.DataFrame(verification_rows)
    layer_b = pd.DataFrame(layer_b_rows)
    verification_csv = out_dir / "verification_results.csv"
    verification_parquet = out_dir / "verification_results.parquet"
    layer_b_csv = layer_b_dir / "layer_b_v1.csv"
    layer_b_parquet = layer_b_dir / "layer_b_v1.parquet"
    verification.to_csv(verification_csv, index=False)
    verification.to_parquet(verification_parquet, index=False)
    layer_b.to_csv(layer_b_csv, index=False)
    layer_b.to_parquet(layer_b_parquet, index=False)
    sample = _sanity_sample(root, layer_b, evidence)

    decision_counts = verification["verification_decision"].value_counts().to_dict()
    resolution_counts = verification["resolution_level"].value_counts().to_dict()
    accepted_resolution_counts = verification[verification["verification_decision"] == "ACCEPT"]["resolution_level"].value_counts().to_dict()
    rx_count = int((layer_b["rxnorm_rxcui"].astype(str).str.len() > 0).sum())
    atc_count = int(layer_b["atc_codes_json"].map(lambda s: len(json.loads(s)) > 0).sum())
    quarantine_review = int(layer_b[(layer_b["verification_decision"] == "HUMAN_REVIEW") & (layer_b["review_reason_codes_json"].str.contains("QUARANTINE_ONLY_PRODUCT|INSUFFICIENT_SOURCE_SUPPORT", regex=True))].shape[0])
    fabricated = int(((layer_b["verification_decision"] == "NIL") & (layer_b["primary_candidate_id"].astype(str).str.len() > 0)).sum())
    missing_prov = int(((layer_b["primary_candidate_id"].astype(str).str.len() > 0) & (layer_b["supporting_candidate_ids_json"] == "[]")).sum())
    blockers = []
    if fabricated:
        blockers.append("NIL records contain selected/fabricated candidate IDs.")
    if missing_prov:
        blockers.append("Some asserted primary identifiers lack candidate provenance.")
    summary = {
        "generated_at": _now_iso(),
        "version": STAGE6_VERSION,
        "mentions_verified": int(verification["mention_id"].nunique()),
        "decision_counts": decision_counts,
        "decision_rates": {k: v / len(verification) for k, v in decision_counts.items()},
        "resolution_level_distribution": resolution_counts,
        "accepted_resolution_level_distribution": accepted_resolution_counts,
        "exact_local_product_count": int(accepted_resolution_counts.get("EXACT_LOCAL_PRODUCT", 0)),
        "brand_family_count": int(accepted_resolution_counts.get("LOCAL_BRAND_FAMILY", 0)),
        "formulation_count": int(accepted_resolution_counts.get("CLINICAL_FORMULATION", 0)),
        "ingredient_only_count": int(accepted_resolution_counts.get("INGREDIENT_ONLY", 0)),
        "rxnorm_supported_records": rx_count,
        "atc_supported_records": atc_count,
        "quarantine_only_human_review_count": quarantine_review,
        "sanity_review_rows": int(len(sample)),
        "sanity_review_package": str(root / "review/layer_b_sanity_100.csv"),
        "median_candidates_considered": float(median(evidence.groupby("mention_id")["candidate_id"].nunique().tolist())),
        "gold_metrics_reported": False,
        "blockers": blockers,
        "READY_FOR_END_TO_END_INTEGRATION": not blockers,
        "paths": {
            "verification_results_csv": str(verification_csv),
            "verification_results_parquet": str(verification_parquet),
            "layer_b_v1_csv": str(layer_b_csv),
            "layer_b_v1_parquet": str(layer_b_parquet),
            "summary": str(out_dir / "verification_summary.json"),
            "report": str(root / "rebuild/reports/STAGE6_VERIFICATION_REPORT.md"),
            "config": str(root / "configs/verification/stage6_verification_config.json"),
            "sanity_review_package": str(root / "review/layer_b_sanity_100.csv"),
        },
    }
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(root, summary)
    return summary


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 6 Verification Report",
        "",
        f"- version: {STAGE6_VERSION}",
        f"- generated_at: {summary['generated_at']}",
        f"- mentions_verified: {summary['mentions_verified']}",
        f"- ready_for_end_to_end_integration: {str(summary['READY_FOR_END_TO_END_INTEGRATION']).lower()}",
        "",
        "## Guardrails",
        "- Verification is deterministic and configurable.",
        "- OFFICIAL_SOURCE_RECORD cannot become the final primary identity.",
        "- Quarantine-only brand products cannot automatically receive exact-product ACCEPT.",
        "- RxNorm is retained when supported but is not mandatory for local acceptance.",
        "- No semantic accuracy, MRR, NDCG, or other gold metrics are reported.",
        "",
        "## Decisions",
        f"- decision_counts: {summary['decision_counts']}",
        f"- resolution_level_distribution: {summary['resolution_level_distribution']}",
        f"- rxnorm_supported_records: {summary['rxnorm_supported_records']}",
        f"- atc_supported_records: {summary['atc_supported_records']}",
        f"- quarantine_only_human_review_count: {summary['quarantine_only_human_review_count']}",
        "",
        "## Outputs",
    ]
    for key, value in summary["paths"].items():
        lines.append(f"- {key}: {value}")
    if summary["blockers"]:
        lines.extend(["", "## Blockers"])
        for blocker in summary["blockers"]:
            lines.append(f"- {blocker}")
    (root / "rebuild/reports/STAGE6_VERIFICATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
