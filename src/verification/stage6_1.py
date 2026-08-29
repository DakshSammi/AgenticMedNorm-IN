from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.verification.stage6 import _j, _layer_b_row, _read_csv, _resolution


STAGE6_1_VERSION = "stage6_1_resolution_audit_v0.1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nppa_supported_families(root: Path) -> set[str]:
    products = _read_csv(root / "knowledge/canonical/brand_products.csv")
    families = _read_csv(root / "knowledge/canonical/brand_families.csv")
    validation = _read_csv(root / "knowledge/reports/open_vs_nppa_validation_stage2b.csv")
    nppa = _read_csv(root / "knowledge/staging/nppa_brand_index.csv")
    matched_source_ids = set(validation[validation["l1_brand_family_overlap"] == "MATCH"]["source_product_id"])
    by_product = set(products[products["source_product_id"].isin(matched_source_ids)]["brand_family_id"])
    nppa_names = set(nppa["normalized_brand_family"]) | set(nppa["normalized_brand_name"])
    by_family = set(families[families["normalized_name"].isin(nppa_names)]["brand_family_id"])
    return set(v for v in (by_product | by_family) if v)


def _candidate_brand_family_id(row: pd.Series) -> str:
    facts = _j(row["candidate_facts_json"], {})
    if row["entity_type"] == "BRAND_FAMILY":
        return row["entity_id"]
    return facts.get("brand_family_id", "")


def _has_hard(row: pd.Series) -> bool:
    return bool(_j(row["hard_conflicts_json"], []))


def _nppa_family_candidate(row: pd.Series, supported_families: set[str]) -> bool:
    family_id = _candidate_brand_family_id(row)
    return bool(family_id and family_id in supported_families)


def _accept_family_from(row: pd.Series, family_id: str, family_name: str, mention: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    supporting = sorted(set(_j(row["supporting_evidence_ids_json"], []) + ["NPPA_BRAND_INDEX_L1_MATCH"]))
    verification = {
        "mention_id": row["mention_id"],
        "verification_decision": "ACCEPT",
        "selected_candidate_id": f"ENTITY:BRAND_FAMILY:{family_id}",
        "resolution_level": "LOCAL_BRAND_FAMILY",
        "decision_reason_codes_json": "[]",
        "hard_conflicts_json": row["hard_conflicts_json"],
        "missing_evidence_json": row["missing_evidence_json"],
        "evidence_ids_json": json.dumps(supporting, sort_keys=True),
        "verification_method": "deterministic_stage6_1_nppa_brand_family_fallback",
        "verification_version": STAGE6_1_VERSION,
        "timestamp": _now_iso(),
    }
    facts = {
        "components": [],
        "rxnorm": [],
        "atc": [],
        "brand_family_id": family_id,
        "brand_family_name": family_name,
        "brand_product_id": "",
        "brand_product_name": "",
        "dosage_form": "",
        "release_modifier": "",
        "fdc_status": "UNKNOWN",
    }
    layer_b = _layer_b_row(row, "ACCEPT", mention, [], _j(row["hard_conflicts_json"], []), _j(row["missing_evidence_json"], []))
    layer_b["resolution_level"] = "LOCAL_BRAND_FAMILY"
    layer_b["primary_candidate_id"] = f"ENTITY:BRAND_FAMILY:{family_id}"
    layer_b["local_brand_family_id"] = family_id
    layer_b["local_brand_family_name"] = family_name
    layer_b["local_brand_product_id"] = ""
    layer_b["local_brand_product_name"] = ""
    layer_b["ingredient_components_json"] = "[]"
    layer_b["component_strengths_json"] = "[]"
    layer_b["dosage_form"] = ""
    layer_b["release_modifier"] = ""
    layer_b["fdc_status"] = facts["fdc_status"]
    layer_b["rxnorm_rxcui"] = ""
    layer_b["rxnorm_name"] = ""
    layer_b["atc_codes_json"] = "[]"
    layer_b["supporting_candidate_ids_json"] = json.dumps([f"ENTITY:BRAND_FAMILY:{family_id}"], sort_keys=True)
    layer_b["supporting_evidence_ids_json"] = json.dumps(supporting, sort_keys=True)
    layer_b["review_reason_codes_json"] = "[]"
    layer_b["pipeline_version"] = STAGE6_1_VERSION
    layer_b["provenance_json"] = json.dumps({"stage6_1_fallback_from_candidate": row["candidate_id"], "nppa_brand_family_supported": True, "verification_generated_at": _now_iso()}, sort_keys=True)
    return verification, layer_b


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 6.1 Resolution Audit Report",
        "",
        f"- version: {STAGE6_1_VERSION}",
        f"- generated_at: {summary['generated_at']}",
        f"- verifier_bug_found: {str(summary['verifier_bug_found']).lower()}",
        f"- layer_b_v1_1_created: {str(summary['layer_b_v1_1_created']).lower()}",
        f"- context_evidence_implemented: {str(summary['context_evidence_implemented']).lower()}",
        "",
        "## Brand-Family Audit",
        f"- brand_like_mentions: {summary['brand_like_mentions']}",
        f"- top_ranked_brand_product_candidates: {summary['top_ranked_brand_product_candidates']}",
        f"- top_ranked_brand_family_candidates: {summary['top_ranked_brand_family_candidates']}",
        f"- candidates_with_nppa_brand_index_corroboration: {summary['candidates_with_nppa_brand_index_corroboration']}",
        f"- candidates_with_strong_lexical_brand_agreement: {summary['candidates_with_strong_lexical_brand_agreement']}",
        f"- candidates_blocked_only_because_product_level_evidence_is_quarantine: {summary['blocked_only_product_quarantine']}",
        f"- brand_family_accept_under_stage6_v1: {summary['brand_family_accept_v1']}",
        f"- brand_family_accept_under_stage6_1_v1_1: {summary['brand_family_accept_v1_1']}",
        "",
        "Stage 6 had zero accepted brand-family results because NPPA L1 brand-family corroboration existed in Stage 1B/2B validation outputs but was not attached as field-level provenance to BRAND_FAMILY candidates or the brand family behind BRAND_PRODUCT candidates. Stage 6.1 fixes that wiring while preserving the rule that quarantine-only product rows cannot become exact-product ACCEPT.",
        "",
        "## Decision Comparison",
        f"- v1_decision_counts: {summary['v1_decision_counts']}",
        f"- v1_1_decision_counts: {summary['v1_1_decision_counts']}",
        f"- v1_resolution_counts: {summary['v1_resolution_counts']}",
        f"- v1_1_resolution_counts: {summary['v1_1_resolution_counts']}",
        "",
        "## Outputs",
    ]
    for key, value in summary["paths"].items():
        lines.append(f"- {key}: {value}")
    (root / "rebuild/reports/STAGE6_1_RESOLUTION_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "rebuild/reports/BRAND_FAMILY_RESOLUTION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage6_1_resolution_audit(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    out_ver = root / "derived/verification"
    out_layer = root / "derived/layer_b"
    out_ver.mkdir(parents=True, exist_ok=True)
    out_layer.mkdir(parents=True, exist_ok=True)
    evidence = _read_csv(root / "derived/evidence/evidence_assessments.csv")
    verification_v1 = _read_csv(root / "derived/verification/verification_results.csv")
    layer_b_v1 = _read_csv(root / "derived/layer_b/layer_b_v1.csv")
    layer_a = _read_csv(root / "derived/layer_a_medication_mentions.csv")
    mention_meta = layer_a.set_index("mention_id").to_dict("index")
    supported_families = _nppa_supported_families(root)

    evidence["ranking_position_int"] = pd.to_numeric(evidence["ranking_position"], errors="coerce").fillna(999999).astype(int)
    candidates = evidence[evidence["ranking_position_int"] <= 20].copy()
    candidates["nppa_brand_family_supported"] = candidates.apply(lambda row: _nppa_family_candidate(row, supported_families), axis=1)
    candidates["strong_lexical_brand_agreement"] = candidates["lexical_status"].eq("MATCH")
    candidates["has_hard_conflict"] = candidates.apply(_has_hard, axis=1)
    brand_like = candidates[candidates["entity_type"].isin(["BRAND_PRODUCT", "BRAND_FAMILY"])]
    top = candidates.sort_values(["mention_id", "ranking_position_int"]).groupby("mention_id").head(1)
    blocked_only = brand_like[
        brand_like["nppa_brand_family_supported"]
        & brand_like["strong_lexical_brand_agreement"]
        & ~brand_like["has_hard_conflict"]
        & brand_like["entity_type"].eq("BRAND_PRODUCT")
        & brand_like["provenance_evidence_json"].str.contains("CANDIDATE_QUARANTINE", regex=False)
    ]

    verification_rows = []
    layer_rows = []
    changed_mentions: set[str] = set()
    for _, base in verification_v1.iterrows():
        mention_id = base["mention_id"]
        accepted_existing = base["verification_decision"] == "ACCEPT"
        replacement_ver = None
        replacement_layer = None
        if not accepted_existing or base["resolution_level"] in {"INGREDIENT_ONLY", "TERMINOLOGY_ONLY"}:
            group = candidates[candidates["mention_id"] == mention_id].sort_values("ranking_position_int")
            eligible = group[
                group["entity_type"].isin(["BRAND_PRODUCT", "BRAND_FAMILY"])
                & group["nppa_brand_family_supported"]
                & group["strong_lexical_brand_agreement"]
                & ~group["has_hard_conflict"]
            ]
            if not eligible.empty:
                chosen = eligible.iloc[0]
                facts = _j(chosen["candidate_facts_json"], {})
                family_id = chosen["entity_id"] if chosen["entity_type"] == "BRAND_FAMILY" else facts.get("brand_family_id", "")
                family_name = facts.get("brand_family_name", "") or facts.get("candidate_name", "")
                if family_id:
                    mention = dict(mention_meta.get(mention_id, {}))
                    mention["mention_id"] = mention_id
                    mention.setdefault("raw_medication_text", chosen["raw_medication_text"])
                    replacement_ver, replacement_layer = _accept_family_from(chosen, family_id, family_name, mention)
        if replacement_ver and replacement_layer:
            verification_rows.append(replacement_ver)
            layer_rows.append(replacement_layer)
            changed_mentions.add(mention_id)
        else:
            verification_rows.append(base.to_dict())
            current_layer = layer_b_v1[layer_b_v1["mention_id"] == mention_id]
            if current_layer.empty:
                layer_rows.append({"mention_id": mention_id})
            else:
                layer_rows.append(current_layer.iloc[0].to_dict())

    verification_v1_1 = pd.DataFrame(verification_rows)
    layer_b_v1_1 = pd.DataFrame(layer_rows)
    verification_csv = out_ver / "verification_results_v1_1.csv"
    verification_parquet = out_ver / "verification_results_v1_1.parquet"
    layer_csv = out_layer / "layer_b_v1_1.csv"
    layer_parquet = out_layer / "layer_b_v1_1.parquet"
    audit_csv = out_ver / "stage6_1_brand_family_resolution_audit.csv"
    candidates[
        [
            "mention_id",
            "raw_medication_text",
            "candidate_id",
            "entity_type",
            "ranking_position",
            "lexical_status",
            "nppa_brand_family_supported",
            "strong_lexical_brand_agreement",
            "has_hard_conflict",
            "resolution_level",
        ]
    ].to_csv(audit_csv, index=False)
    verification_v1_1.to_csv(verification_csv, index=False)
    verification_v1_1.to_parquet(verification_parquet, index=False)
    layer_b_v1_1.to_csv(layer_csv, index=False)
    layer_b_v1_1.to_parquet(layer_parquet, index=False)

    context_flag = root / "configs/evidence/context_runtime_flags.json"
    context_flag.parent.mkdir(parents=True, exist_ok=True)
    context_flag.write_text(json.dumps({"context_evidence_implemented": False, "reason": "No deterministic BODHI-M drug-context mapping is operational in current production configuration."}, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "generated_at": _now_iso(),
        "version": STAGE6_1_VERSION,
        "reason_brand_family_accept_zero": "NPPA L1 brand-family support existed in validation artifacts but was not consumed as field-level candidate evidence by Stage 6.",
        "nppa_brand_family_evidence_linkage_operational": True,
        "verifier_bug_found": bool(changed_mentions),
        "layer_b_v1_1_created": bool(changed_mentions),
        "changed_mentions": len(changed_mentions),
        "brand_like_mentions": int(brand_like["mention_id"].nunique()),
        "top_ranked_brand_product_candidates": int(top["entity_type"].eq("BRAND_PRODUCT").sum()),
        "top_ranked_brand_family_candidates": int(top["entity_type"].eq("BRAND_FAMILY").sum()),
        "candidates_with_nppa_brand_index_corroboration": int(candidates["nppa_brand_family_supported"].sum()),
        "candidates_with_strong_lexical_brand_agreement": int(candidates["strong_lexical_brand_agreement"].sum()),
        "blocked_only_product_quarantine": int(blocked_only["mention_id"].nunique()),
        "brand_family_accept_v1": int(((verification_v1["verification_decision"] == "ACCEPT") & (verification_v1["resolution_level"] == "LOCAL_BRAND_FAMILY")).sum()),
        "brand_family_accept_v1_1": int(((verification_v1_1["verification_decision"] == "ACCEPT") & (verification_v1_1["resolution_level"] == "LOCAL_BRAND_FAMILY")).sum()),
        "v1_decision_counts": verification_v1["verification_decision"].value_counts().to_dict(),
        "v1_1_decision_counts": verification_v1_1["verification_decision"].value_counts().to_dict(),
        "v1_resolution_counts": verification_v1["resolution_level"].value_counts().to_dict(),
        "v1_1_resolution_counts": verification_v1_1["resolution_level"].value_counts().to_dict(),
        "context_evidence_implemented": False,
        "paths": {
            "brand_family_resolution_audit": str(audit_csv),
            "verification_results_v1_1_csv": str(verification_csv),
            "verification_results_v1_1_parquet": str(verification_parquet),
            "layer_b_v1_1_csv": str(layer_csv),
            "layer_b_v1_1_parquet": str(layer_parquet),
            "stage6_1_report": str(root / "rebuild/reports/STAGE6_1_RESOLUTION_AUDIT_REPORT.md"),
            "brand_family_report": str(root / "rebuild/reports/BRAND_FAMILY_RESOLUTION_AUDIT.md"),
            "context_runtime_flags": str(context_flag),
            "summary": str(out_ver / "verification_summary_v1_1.json"),
        },
    }
    (out_ver / "verification_summary_v1_1.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(root, summary)
    return summary
