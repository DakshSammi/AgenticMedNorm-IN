#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PA = ROOT / "paper_artifacts"
TABLES = PA / "tables"
FIGURES = PA / "figures"
ACCOUNTING = PA / "accounting"
METRICS = PA / "final_metrics"
BUNDLE = PA / "paper_writer_bundle"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / path if isinstance(path, str) else path, dtype=str).fillna("")


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_tex(path: Path, frame: pd.DataFrame, caption: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tex = frame.to_latex(index=False, escape=True, caption=caption or None)
    path.write_text(tex, encoding="utf-8")


def pct(n: int | float, d: int | float) -> float:
    return round(100 * float(n) / float(d), 1) if d else 0.0


def wilson_ci(n: int, d: int) -> tuple[float, float]:
    if d <= 0:
        return (0.0, 0.0)
    z = 1.96
    p = n / d
    den = 1 + z * z / d
    centre = p + z * z / (2 * d)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * d)) / d)
    return (round(100 * (centre - margin) / den, 1), round(100 * (centre + margin) / den, 1))


def parse_json_cell(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNCOMMITTED"


def status_from_manifest() -> dict[str, Any]:
    manifest = json.loads((ROOT / "configs/frozen/evaluation_final_893_manifest.json").read_text(encoding="utf-8"))
    if not (manifest.get("FINAL_893_INPUT_COMPLETE") and manifest.get("FINAL_893_PIPELINE_COMPLETE") and manifest.get("FINAL_893_READY")):
        raise SystemExit("FINAL_RELEASE_BLOCKED = TRUE: final893 manifest is not ready")
    return manifest


def load_sources() -> dict[str, Any]:
    return {
        "manifest": status_from_manifest(),
        "docs": read_csv("derived/layer_a_documents.csv"),
        "mentions": read_csv("derived/layer_a_medication_mentions.csv"),
        "layer_b": read_csv("derived/layer_b/layer_b_v1_1.csv"),
        "ranked": read_csv("derived/ranking/ranked_candidates.csv"),
        "evidence": read_csv("derived/evidence/evidence_assessments.csv"),
        "stage2c": json.loads((ROOT / "derived/retrieval/stage2c_summary.json").read_text(encoding="utf-8")),
        "stage2c1": json.loads((ROOT / "derived/retrieval/stage2c1_summary.json").read_text(encoding="utf-8")),
        "stage4": json.loads((ROOT / "derived/ranking/ranking_summary.json").read_text(encoding="utf-8")),
        "stage5": json.loads((ROOT / "derived/evidence/evidence_summary.json").read_text(encoding="utf-8")),
        "stage7": json.loads((ROOT / "derived/pipeline/stage7_smoke_summary.json").read_text(encoding="utf-8")),
        "consistency": json.loads((ROOT / "derived/pipeline/full_dataset_consistency_audit.json").read_text(encoding="utf-8")),
        "qwen": read_csv("expert validation paper/04_local_open_models/qwen/qwen_final_validation762_results.csv"),
        "audit_master": read_csv("expert validation paper/03_reviewer_packages/expert_validation_762_master.csv"),
        "concordance": read_csv("paper_artifacts/tables/Table_LLM_Auditor_Concordance.csv"),
        "benchmark": read_csv("paper_artifacts/benchmarking/annotation_model_leaderboard_reproduced.csv"),
        "resources": read_csv("knowledge/reports/OPERATIONAL_RESOURCE_MANIFEST.csv"),
    }


def semantic_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["has_rxcui"] = out["rxnorm_rxcui"].astype(str).str.len() > 0
    out["has_atc"] = out["atc_codes_json"].map(lambda value: len(parse_json_cell(value, [])) > 0)
    out["has_local_brand_family"] = out["local_brand_family_id"].astype(str).str.len() > 0
    out["has_local_product"] = out["local_brand_product_id"].astype(str).str.len() > 0
    out["has_primary_entity"] = out["primary_candidate_id"].astype(str).str.startswith("ENTITY:")
    out["has_external_standardized_id"] = out["has_rxcui"] | out["has_atc"]
    out["has_local_semantic_id"] = out["has_primary_entity"] & ~out["primary_candidate_id"].astype(str).str.startswith("ENTITY:RXNORM_CONCEPT:")
    out["has_local_and_external"] = out["has_local_semantic_id"] & out["has_external_standardized_id"]
    out["has_no_semantic_identifier"] = ~out["has_primary_entity"] & ~out["has_external_standardized_id"]
    return out


def audit762_with_pipeline(src: dict[str, Any]) -> pd.DataFrame:
    qwen = src["qwen"]
    master = src["audit_master"]
    merged = master.merge(qwen, on=["mention_id", "p_id"], how="inner", suffixes=("_master", ""))
    if len(merged) != 762:
        raise SystemExit(f"Qwen 762 integrity failed: merged rows={len(merged)}")
    merged["has_local_brand_family"] = merged["pipeline_selected_entity_id"].str.startswith("BFAM_")
    merged["has_local_product"] = merged["pipeline_selected_entity_id"].str.startswith("BPROD_")
    merged["has_local_semantic_id"] = merged["pipeline_selected_entity_id"].astype(str).str.len() > 0
    merged["has_external_standardized_id"] = False
    merged["has_rxcui"] = False
    merged["has_atc"] = False
    merged["has_local_and_external"] = False
    merged["has_no_semantic_identifier"] = ~merged["has_local_semantic_id"]
    merged["fdc_visible"] = merged["raw_medication_text"].str.contains(r"\+|/|\bduo\b|\btrio\b|\bplus\b|\bclav\b", case=False, regex=True)
    return merged


def make_tables(src: dict[str, Any]) -> dict[str, dict[str, Any]]:
    TABLES.mkdir(parents=True, exist_ok=True)
    docs = src["docs"]
    mentions = src["mentions"]
    layer_b = semantic_flags(src["layer_b"])
    audit = audit762_with_pipeline(src)
    benchmark = src["benchmark"].copy()
    qwen = src["qwen"]

    mention_docs = set(mentions["document_uid"])
    corpus_rows = [
        {"metric": "raw_prescriptions", "N": 893, "source": "configs/frozen/evaluation_final_893_manifest.json"},
        {"metric": "deidentified_prescriptions", "N": 893, "source": "configs/frozen/evaluation_final_893_manifest.json"},
        {"metric": "annotation_complete_prescriptions", "N": 893, "source": "configs/frozen/evaluation_final_893_manifest.json"},
        {"metric": "pipeline_evaluated_prescriptions", "N": len(docs), "source": "derived/layer_a_documents.csv"},
        {"metric": "medication_bearing_prescriptions", "N": len(mention_docs), "source": "derived/layer_a_medication_mentions.csv"},
        {"metric": "zero_medication_prescriptions", "N": len(docs) - len(mention_docs), "source": "derived/layer_a_documents.csv; derived/layer_a_medication_mentions.csv"},
        {"metric": "medication_mentions", "N": len(mentions), "source": "derived/layer_a_medication_mentions.csv"},
        {"metric": "unique_medication_surface_forms", "N": mentions["lexical_surface_normalized"].replace("", np.nan).nunique(), "source": "derived/layer_a_medication_mentions.csv:lexical_surface_normalized"},
    ]
    corpus = pd.DataFrame(corpus_rows)
    write_csv(TABLES / "table_final_corpus_accounting.csv", corpus)
    write_tex(TABLES / "table_final_corpus_accounting.tex", corpus, "Final corpus accounting.")

    denom = len(layer_b)
    disp = pd.DataFrame(
        [
            {"verification_state": key, "N": int(value), "percentage": pct(value, denom), "ci95_descriptive": f"{wilson_ci(int(value), denom)[0]}-{wilson_ci(int(value), denom)[1]}"}
            for key, value in layer_b["verification_decision"].value_counts().reindex(["ACCEPT", "HUMAN_REVIEW", "NIL"], fill_value=0).items()
        ]
    )
    write_csv(TABLES / "table_pipeline_disposition.csv", disp)
    write_tex(TABLES / "table_pipeline_disposition.tex", disp, "Pipeline verification disposition.")

    res_rows = []
    for level, group in layer_b.groupby("resolution_level", sort=True):
        res_rows.append(
            {
                "resolution_level": level,
                "N": len(group),
                "percentage": pct(len(group), denom),
                "rxcui_coverage": pct(group["has_rxcui"].sum(), len(group)),
                "atc_coverage": pct(group["has_atc"].sum(), len(group)),
                "local_semantic_id_coverage": pct(group["has_local_semantic_id"].sum(), len(group)),
            }
        )
    res = pd.DataFrame(res_rows).sort_values("N", ascending=False)
    write_csv(TABLES / "table_resolution_levels.csv", res)
    write_tex(TABLES / "table_resolution_levels.tex", res, "Resolution-level distribution and identifier coverage.")

    id_metrics = [
        ("RxNorm RxCUI", "has_rxcui"),
        ("ATC therapeutic-class identifier", "has_atc"),
        ("local brand-family ID", "has_local_brand_family"),
        ("local product ID", "has_local_product"),
        ("any external standardized ID", "has_external_standardized_id"),
        ("any local semantic ID", "has_local_semantic_id"),
        ("both local + external", "has_local_and_external"),
        ("no semantic identifier", "has_no_semantic_identifier"),
    ]
    id_rows = []
    for cohort, frame in [("full893_pipeline", layer_b), ("audit762", audit)]:
        for label, col in id_metrics:
            id_rows.append({"cohort": cohort, "identifier_type": label, "N": int(frame[col].sum()), "denominator": len(frame), "percentage": pct(frame[col].sum(), len(frame))})
    ids = pd.DataFrame(id_rows)
    write_csv(TABLES / "table_semantic_identifier_coverage.csv", ids)
    write_tex(TABLES / "table_semantic_identifier_coverage.tex", ids, "Semantic identifier coverage.")

    valid_b = benchmark[(benchmark["N_valid"].astype(str).str.len() > 0) & (benchmark["publication_status"] != "EXCLUDED")].copy()
    valid_b["score_float"] = pd.to_numeric(valid_b["primary_score"], errors="coerce")
    picks = []
    for label, mask in [
        ("GPT-5.5", valid_b["paper_display_name"].eq("GPT-5.5")),
        ("best valid proprietary comparator", valid_b["open_or_closed"].eq("CLOSED") & ~valid_b["paper_display_name"].eq("GPT-5.5")),
        ("best valid open/open-weight direct VLM", valid_b["track"].eq("direct_vlm") & valid_b["open_or_closed"].eq("OPEN") & valid_b["is_alias"].ne("TRUE")),
        ("best valid conventional OCR", valid_b["track"].eq("raw_ocr") & valid_b["open_or_closed"].eq("OPEN")),
        ("best valid open/open-weight hybrid", valid_b["track"].eq("hybrid") & valid_b["open_or_closed"].eq("OPEN")),
    ]:
        subset = valid_b[mask].sort_values("score_float", ascending=False)
        if not subset.empty:
            row = subset.iloc[0].copy()
            row["representative_role"] = label
            picks.append(row)
    bench_main_cols = ["representative_role", "track", "paper_display_name", "model_category", "open_or_closed", "N_valid", "coverage", "primary_score", "95CI_lower", "95CI_upper", "runtime_seconds", "publication_status", "is_alias"]
    bench_main = pd.DataFrame(picks)[bench_main_cols]
    write_csv(TABLES / "table_annotation_model_benchmark_main.csv", bench_main)
    write_tex(TABLES / "table_annotation_model_benchmark_main.tex", bench_main, "Annotation-model benchmark representative systems.")
    write_csv(TABLES / "table_annotation_model_benchmark_full.csv", benchmark)
    write_tex(TABLES / "table_annotation_model_benchmark_full.tex", benchmark, "Annotation-model benchmark full leaderboard.")

    qwen_rows = []
    for field in ["mapping_assessment", "pipeline_decision_assessment"]:
        order = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"] if field == "mapping_assessment" else ["AGREE", "DISAGREE", "UNCERTAIN"]
        counts = qwen[field].value_counts().reindex(order, fill_value=0)
        for key, value in counts.items():
            qwen_rows.append({"assessment": field, "value": key, "N": int(value), "percentage": pct(value, len(qwen)), "denominator": len(qwen)})
    qwen_table = pd.DataFrame(qwen_rows)
    write_csv(TABLES / "table_qwen_semantic_audit.csv", qwen_table)
    write_tex(TABLES / "table_qwen_semantic_audit.tex", qwen_table, "Independent Qwen semantic audit.")

    def grouped_qwen(field: str, out_name: str, min_n: int = 10) -> pd.DataFrame:
        rows = []
        for value, group in audit.groupby(field, sort=True):
            if len(group) < min_n:
                continue
            counts = group["mapping_assessment"].value_counts().reindex(["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"], fill_value=0)
            for status, n in counts.items():
                rows.append({"group": value, "mapping_assessment": status, "N": int(n), "group_denominator": len(group), "percentage": pct(n, len(group))})
        frame = pd.DataFrame(rows)
        write_csv(TABLES / out_name, frame)
        return frame

    grouped_qwen("pipeline_resolution_level", "table_qwen_by_resolution.csv")
    grouped_qwen("pipeline_verification_decision", "table_qwen_by_verification.csv")
    grouped_qwen("has_local_semantic_id", "table_qwen_by_semantic_id.csv")
    grouped_qwen("fdc_visible", "table_qwen_by_fdc.csv")

    concord = src["concordance"]
    write_csv(TABLES / "table_semantic_auditor_concordance.csv", concord)
    write_tex(TABLES / "table_semantic_auditor_concordance.tex", concord, "Inter-model semantic-auditor concordance.")

    resources = src["resources"].copy()
    count_lookup = {
        "NPPA": len(read_csv("knowledge/v1_1/peer_nppa_product_evidence.csv")) if (ROOT / "knowledge/v1_1/peer_nppa_product_evidence.csv").exists() else "",
        "CDSCO": len(read_csv("knowledge/canonical/cdsco_structured_records.csv")) if (ROOT / "knowledge/canonical/cdsco_structured_records.csv").exists() else "",
        "NLEM": len(read_csv("knowledge/canonical/nlem_entries.csv")) if (ROOT / "knowledge/canonical/nlem_entries.csv").exists() else "",
        "RxNorm/RxNav": len(read_csv("knowledge/crosswalks/rxnorm_ingredient_mappings.csv")),
        "ATC/RxClass": len(read_csv("knowledge/crosswalks/rxclass_atc_mappings.csv")),
        "open Indian medicine dataset": len(read_csv("knowledge/canonical/brand_products.csv")),
    }
    branch_lookup = {"NPPA": "R5/evidence", "CDSCO": "R3/R5", "NLEM": "R3/R5", "RxNorm/RxNav": "R4/evidence", "ATC/RxClass": "evidence", "open Indian medicine dataset": "R1/R2/R5"}
    resource_rows = []
    for row in resources.to_dict("records"):
        name = row["resource"]
        if row["used_in_operational_v1"] != "TRUE":
            continue
        resource_rows.append(
            {
                "resource": name,
                "country_or_scope": "India" if name in {"NPPA", "CDSCO", "NLEM", "open Indian medicine dataset"} else "international terminology",
                "role": row["role"],
                "authority_tier": "official" if name in {"NPPA", "CDSCO", "NLEM"} else ("terminology" if name in {"RxNorm/RxNav", "ATC/RxClass"} else "secondary candidate inventory"),
                "retrieval_branch": branch_lookup.get(name, ""),
                "version_or_date": row["source_version/date"],
                "operational_status": row["paper_status"],
                "record_or_concept_count": count_lookup.get(name, ""),
            }
        )
    kr = pd.DataFrame(resource_rows)
    write_csv(TABLES / "table_knowledge_resources.csv", kr)
    write_tex(TABLES / "table_knowledge_resources.tex", kr, "Operational knowledge resources.")

    manifest_rows = []
    for path in sorted(TABLES.glob("table_*.csv")):
        manifest_rows.append({"table_id": path.stem, "title": path.stem.replace("_", " "), "study": infer_study(path.name), "main_or_supplement": table_recommendation(path.name), "source_data": source_for_table(path.name), "csv": rel(path), "tex": rel(path.with_suffix(".tex")) if path.with_suffix(".tex").exists() else "", "status": "VERIFIED_FINAL" if "benchmark" not in path.name and "qwen" not in path.name and "concordance" not in path.name else "VERIFIED_HISTORICAL", "notes": ""})
    table_manifest = pd.DataFrame(manifest_rows)
    write_csv(TABLES / "TABLE_MANIFEST_FINAL.csv", table_manifest)
    return {row["table_id"]: row for row in manifest_rows}


def infer_study(name: str) -> str:
    if "benchmark" in name:
        return "Study A"
    if "qwen" in name or "concordance" in name:
        return "Study C"
    return "Study B"


def table_recommendation(name: str) -> str:
    main = {"table_final_corpus_accounting.csv", "table_pipeline_disposition.csv", "table_resolution_levels.csv", "table_qwen_semantic_audit.csv"}
    return "MAIN_CANDIDATE" if name in main else "SUPPLEMENTARY"


def source_for_table(name: str) -> str:
    if "benchmark" in name:
        return "paper_artifacts/benchmarking/annotation_model_leaderboard_reproduced.csv"
    if "qwen" in name:
        return "expert validation paper/04_local_open_models/qwen/qwen_final_validation762_results.csv"
    if "concordance" in name:
        return "paper_artifacts/tables/Table_LLM_Auditor_Concordance.csv"
    if "knowledge" in name:
        return "knowledge/reports/OPERATIONAL_RESOURCE_MANIFEST.csv"
    return "derived/layer_b/layer_b_v1_1.csv; derived/layer_a_medication_mentions.csv"


def figdir(name: str) -> Path:
    d = FIGURES / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig: plt.Figure, d: Path) -> None:
    fig.tight_layout()
    fig.savefig(d / "figure.svg")
    fig.savefig(d / "figure.pdf")
    fig.savefig(d / "figure.png", dpi=320)
    plt.close(fig)


def write_figure_wrapper(d: Path, figure_id: str) -> None:
    (d / "generate.py").write_text(
        "from scripts.reproduce_paper_artifacts import render_single_figure\n\n"
        f"render_single_figure('{figure_id}')\n",
        encoding="utf-8",
    )


def barh(source: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str, d: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, 0.35 * len(source) + 1.4)))
    y = np.arange(len(source))
    vals = pd.to_numeric(source[value_col], errors="coerce").fillna(0)
    ax.barh(y, vals, color="#4c78a8")
    ax.set_yticks(y, source[label_col])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for yi, val in zip(y, vals, strict=False):
        ax.text(float(val), yi, f" {float(val):.1f}", va="center", fontsize=8)
    save_figure(fig, d)


def stacked_percent(source: pd.DataFrame, group_col: str, status_col: str, pct_col: str, title: str, d: Path) -> None:
    pivot = source.pivot(index=group_col, columns=status_col, values=pct_col).fillna(0)
    order = [c for c in ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE", "AGREE", "DISAGREE", "UNCERTAIN"] if c in pivot.columns]
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.45 * len(pivot) + 1.4)))
    left = np.zeros(len(pivot))
    colors = ["#3b7ea1", "#b8564b", "#8c8c8c", "#4c8c4a", "#d08c3c", "#9a7fb8"]
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].to_numpy(dtype=float)
        ax.barh(pivot.index.astype(str), vals, left=left, label=col, color=colors[i % len(colors)])
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("% within group")
    ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, d)


def render_architecture(d: Path) -> None:
    nodes = [
        ("Handwritten prescription", 0, 2),
        ("De-identification", 1, 2),
        ("Annotation\\nLayer A", 2, 2),
        ("R1/R2/R3/R4/R5\\nCandidate retrieval", 3, 2),
        ("Candidate union", 4, 2),
        ("RRF k=60", 5, 2),
        ("Evidence", 6, 2),
        ("Verification", 7, 2),
        ("ACCEPT\\nHUMAN_REVIEW\\nNIL\\nLayer B", 8, 2),
        ("India KB", 3, 3.2),
        ("RxNorm/ATC", 5.5, 3.2),
    ]
    edges = [(nodes[i][0], nodes[i + 1][0]) for i in range(8)] + [("India KB", "R1/R2/R3/R4/R5\\nCandidate retrieval"), ("RxNorm/ATC", "Evidence")]
    write_csv(d / "source.csv", pd.DataFrame([{"node": n, "x": x, "y": y} for n, x, y in nodes]))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    positions = {n: (x, y) for n, x, y in nodes}
    for n, x, y in nodes:
        ax.text(x, y, n, ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.35", "fc": "#f7f7f7", "ec": "#333", "lw": 1})
    for a, b in edges:
        ax.annotate("", xy=positions[b], xytext=positions[a], arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333"})
    ax.set_axis_off()
    ax.set_title("AgenticMedNorm-IN six-agent architecture")
    save_figure(fig, d)


def render_flow(d: Path) -> None:
    rows = [
        {"study": "Study A", "cohort": "125 prescriptions", "purpose": "annotation-model selection benchmark"},
        {"study": "Study B", "cohort": "893 prescriptions", "purpose": "full end-to-end pipeline"},
        {"study": "Study C", "cohort": "150 prescriptions / 762 mentions", "purpose": "independent Qwen semantic audit"},
        {"study": "Future work", "cohort": "expert adjudication", "purpose": "not completed for this submission"},
    ]
    write_csv(d / "source.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, row in enumerate(rows):
        y = 3 - i
        ax.text(0, y, row["study"], ha="left", va="center", weight="bold", fontsize=10)
        ax.text(2.2, y, row["cohort"], ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.3", "fc": "#eef4f7", "ec": "#444"})
        ax.annotate("", xy=(4.2, y), xytext=(3.2, y), arrowprops={"arrowstyle": "->", "lw": 1})
        ax.text(5.7, y, row["purpose"], ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.3", "fc": "#f8f8f8", "ec": "#444"})
    ax.set_xlim(-0.2, 8.4)
    ax.set_ylim(-0.8, 3.8)
    ax.set_axis_off()
    ax.set_title("Study and cohort structure")
    save_figure(fig, d)


def make_figures(src: dict[str, Any]) -> dict[str, dict[str, str]]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    layer_b = semantic_flags(src["layer_b"])
    mentions = src["mentions"]
    audit = audit762_with_pipeline(src)
    tables = {p.stem: read_csv(p) for p in TABLES.glob("table_*.csv")}
    figures: list[dict[str, str]] = []

    specs = [
        ("fig01_six_agent_architecture", "Six-agent architecture", "Study B", "MAIN_CANDIDATE", "Architecture and evidence flow"),
        ("fig02_study_cohort_flow", "Study/cohort flow", "Studies A/B/C", "MAIN_CANDIDATE", "Separate denominators and roles"),
        ("fig03_annotation_model_benchmark", "Annotation model benchmark", "Study A", "MAIN_CANDIDATE", "125-document benchmark performance"),
        ("fig04_verification_distribution", "Verification distribution", "Study B", "MAIN_CANDIDATE", "ACCEPT/HUMAN_REVIEW/NIL distribution"),
        ("fig05_resolution_level_distribution", "Resolution-level distribution", "Study B", "MAIN_CANDIDATE", "Normalization depth"),
        ("fig06_semantic_identifier_coverage", "Semantic identifier coverage", "Study B/C", "MAIN_CANDIDATE", "Local and external identifier coverage"),
        ("fig07_qwen_semantic_audit", "Qwen semantic audit", "Study C", "MAIN_CANDIDATE", "Independent automated audit"),
        ("fig08_qwen_by_resolution", "Qwen by resolution level", "Study C", "SUPPLEMENTARY", "Audit labels by resolution"),
        ("fig09_qwen_by_verification", "Qwen by verification state", "Study C", "MAIN_CANDIDATE", "Audit labels by verification state"),
        ("fig10_intermodel_concordance", "Inter-model concordance", "Study C", "SUPPLEMENTARY", "Auditor agreement on overlaps"),
        ("fig11_knowledge_resource_coverage", "Knowledge resource coverage", "Study B", "SUPPLEMENTARY", "Operational resource counts"),
        ("fig12_mentions_per_prescription", "Medication mentions per prescription", "Study B", "SUPPLEMENTARY", "Mention density"),
        ("fig13_surface_frequency_long_tail", "Surface-form long tail", "Study B", "SUPPLEMENTARY", "Lexical diversity"),
        ("fig14_formulation_characteristics", "Formulation characteristics", "Study B", "SUPPLEMENTARY", "FDC/strength/form availability"),
        ("fig15_normalization_case_study", "Normalization case study", "Study B", "MAIN_CANDIDATE", "Synthetic schematic"),
        ("fig16_provenance_completeness", "Provenance completeness", "Study B", "SUPPLEMENTARY", "Provenance and identifier fields"),
    ]
    for figure_id, title, study, status, question in specs:
        d = figdir(figure_id)
        write_figure_wrapper(d, figure_id)
        (d / "caption_notes.md").write_text(f"# {title}\n\nClassification: {status}\n\nCaption draft: {title}. Source data are aggregate and publication-safe. No real prescription image is shown.\n", encoding="utf-8")

    render_architecture(figdir("fig01_six_agent_architecture"))
    render_flow(figdir("fig02_study_cohort_flow"))

    d = figdir("fig03_annotation_model_benchmark")
    src_df = tables["table_annotation_model_benchmark_main"].copy()
    src_df["label"] = src_df["representative_role"] + ": " + src_df["paper_display_name"]
    write_csv(d / "source.csv", src_df)
    barh(src_df.sort_values("primary_score"), "label", "primary_score", "Performance of evaluated systems on the 125-document benchmark", "primary score", d)

    d = figdir("fig04_verification_distribution")
    df = tables["table_pipeline_disposition"].copy()
    write_csv(d / "source.csv", df)
    barh(df.iloc[::-1], "verification_state", "percentage", "Pipeline verification distribution", "% of medication mentions", d)

    d = figdir("fig05_resolution_level_distribution")
    df = tables["table_resolution_levels"].copy()
    write_csv(d / "source.csv", df)
    barh(df.sort_values("N"), "resolution_level", "percentage", "Resolution-level distribution", "% of medication mentions", d)

    d = figdir("fig06_semantic_identifier_coverage")
    df = tables["table_semantic_identifier_coverage"]
    full = df[df["cohort"].eq("full893_pipeline")].copy()
    write_csv(d / "source.csv", df)
    barh(full.iloc[::-1], "identifier_type", "percentage", "Semantic identifier coverage, full893 pipeline", "% of medication mentions", d)

    d = figdir("fig07_qwen_semantic_audit")
    df = tables["table_qwen_semantic_audit"].copy()
    write_csv(d / "source.csv", df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, field, title in zip(axes, ["mapping_assessment", "pipeline_decision_assessment"], ["Mapping assessment", "Routing assessment"], strict=False):
        sub = df[df["assessment"].eq(field)]
        ax.bar(sub["value"], pd.to_numeric(sub["percentage"]), color=["#3b7ea1", "#b8564b", "#8c8c8c"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        for x, y, n in zip(sub["value"], pd.to_numeric(sub["percentage"]), sub["N"], strict=False):
            ax.text(x, y, f"{n}\\n{y:.1f}%", ha="center", va="bottom", fontsize=8)
    save_figure(fig, d)

    d = figdir("fig08_qwen_by_resolution")
    df = tables["table_qwen_by_resolution"]
    write_csv(d / "source.csv", df)
    stacked_percent(df, "group", "mapping_assessment", "percentage", "Qwen mapping assessment by pipeline resolution level", d)

    d = figdir("fig09_qwen_by_verification")
    df = tables["table_qwen_by_verification"]
    write_csv(d / "source.csv", df)
    stacked_percent(df, "group", "mapping_assessment", "percentage", "Qwen mapping assessment by verification state", d)

    d = figdir("fig10_intermodel_concordance")
    df = tables["table_semantic_auditor_concordance"].copy()
    df["mapping_agreement_pct"] = pd.to_numeric(df["mapping_agreement"], errors="coerce") * 100
    write_csv(d / "source.csv", df)
    barh(df.iloc[::-1], "comparison", "mapping_agreement_pct", "Inter-model semantic-auditor concordance", "mapping agreement (%)", d)

    d = figdir("fig11_knowledge_resource_coverage")
    df = tables["table_knowledge_resources"].copy()
    df["record_or_concept_count"] = pd.to_numeric(df["record_or_concept_count"], errors="coerce").fillna(0)
    write_csv(d / "source.csv", df)
    barh(df.sort_values("record_or_concept_count"), "resource", "record_or_concept_count", "Operational knowledge-resource record counts", "records/concepts", d)

    d = figdir("fig12_mentions_per_prescription")
    counts = mentions.groupby("document_uid").size()
    doc_counts = src["docs"][["document_uid"]].merge(counts.rename("mentions").reset_index(), on="document_uid", how="left").fillna({"mentions": 0})
    write_csv(d / "source.csv", doc_counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = doc_counts["mentions"].astype(int)
    ax.hist(vals, bins=range(0, int(vals.max()) + 2), color="#4c78a8", edgecolor="white")
    ax.set_xlabel("medication mentions per prescription")
    ax.set_ylabel("prescriptions")
    ax.set_title(f"Mention density: median {vals.median():.0f}, IQR {vals.quantile(.25):.0f}-{vals.quantile(.75):.0f}")
    save_figure(fig, d)

    d = figdir("fig13_surface_frequency_long_tail")
    freq = mentions["lexical_surface_normalized"].replace("", np.nan).dropna().value_counts().reset_index()
    freq.columns = ["surface", "count"]
    freq["rank"] = np.arange(1, len(freq) + 1)
    write_csv(d / "source.csv", freq)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq["rank"], freq["count"], color="#4c78a8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("surface rank")
    ax.set_ylabel("frequency")
    ax.set_title("Medication surface-form long tail")
    ax.grid(alpha=0.25)
    save_figure(fig, d)

    d = figdir("fig14_formulation_characteristics")
    form_rows = [
        {"feature": "FDC", "N": int(layer_b["fdc_status"].eq("FDC").sum())},
        {"feature": "single component", "N": int(layer_b["fdc_status"].eq("SINGLE_COMPONENT").sum())},
        {"feature": "unknown FDC/component status", "N": int(layer_b["fdc_status"].isin(["", "UNKNOWN"]).sum())},
        {"feature": "explicit strength observed", "N": int(mentions["raw_strength_text"].astype(str).str.len().gt(0).sum())},
        {"feature": "dosage form resolved", "N": int(layer_b["dosage_form"].astype(str).str.len().gt(0).sum())},
    ]
    df = pd.DataFrame(form_rows)
    df["percentage"] = df["N"].map(lambda n: pct(n, len(layer_b)))
    write_csv(d / "source.csv", df)
    barh(df.iloc[::-1], "feature", "percentage", "Formulation characteristics", "% of medication mentions", d)

    d = figdir("fig15_normalization_case_study")
    rows = [{"step": i, "stage": s} for i, s in enumerate(["surface mention", "R1-R5 retrieval", "candidate union", "RRF k=60", "evidence assessment", "verification", "Layer-B identity + provenance"], start=1)]
    write_csv(d / "source.csv", rows)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for row in rows:
        x = row["step"]
        ax.text(x, 0.5, row["stage"], ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.3", "fc": "#f8f8f8", "ec": "#333"})
        if x < len(rows):
            ax.annotate("", xy=(x + 0.55, 0.5), xytext=(x + 0.35, 0.5), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.set_xlim(0.4, len(rows) + 0.6)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title("Synthetic normalization case-study schema")
    save_figure(fig, d)

    d = figdir("fig16_provenance_completeness")
    prov_rows = [
        {"field": "source evidence IDs", "N": int(layer_b["supporting_evidence_ids_json"].map(lambda v: len(parse_json_cell(v, [])) > 0).sum())},
        {"field": "candidate provenance", "N": int(layer_b["supporting_candidate_ids_json"].map(lambda v: len(parse_json_cell(v, [])) > 0).sum())},
        {"field": "resource versions", "N": int(layer_b["kb_resource_versions_json"].astype(str).str.len().gt(0).sum())},
        {"field": "RxCUI", "N": int(layer_b["has_rxcui"].sum())},
        {"field": "ATC", "N": int(layer_b["has_atc"].sum())},
    ]
    df = pd.DataFrame(prov_rows)
    df["percentage"] = df["N"].map(lambda n: pct(n, len(layer_b)))
    write_csv(d / "source.csv", df)
    barh(df.iloc[::-1], "field", "percentage", "Provenance and semantic-field completeness", "% of medication mentions", d)

    for figure_id, title, study, status, question in specs:
        d = figdir(figure_id)
        figures.append(
            {
                "figure_id": figure_id,
                "title": title,
                "study": study,
                "main_or_supplement": status,
                "scientific_question": question,
                "source_data": rel(d / "source.csv") if (d / "source.csv").exists() else "schematic",
                "script": rel(d / "generate.py"),
                "svg": rel(d / "figure.svg"),
                "pdf": rel(d / "figure.pdf"),
                "png": rel(d / "figure.png"),
                "caption_draft": f"{title}. Aggregate data only; no clinical prescription image is shown.",
                "status": "GENERATED",
                "notes": "",
            }
        )
    manifest = pd.DataFrame(figures)
    write_csv(FIGURES / "FIGURE_MANIFEST_FINAL.csv", manifest)
    recommendations = [
        "# Figure Recommendations For 8-Page JBHI",
        "",
        "Recommended main figures:",
        "- `fig01_six_agent_architecture`: anchors the method.",
        "- `fig02_study_cohort_flow`: prevents denominator confusion across Studies A/B/C.",
        "- `fig06_semantic_identifier_coverage`: shows semantic normalization beyond transcription.",
        "- `fig07_qwen_semantic_audit` or `fig09_qwen_by_verification`: use one depending on available space.",
        "",
        "Supplementary figures:",
        "- `fig03_annotation_model_benchmark` if the main text can describe Study A in a table.",
        "- `fig08_qwen_by_resolution`, `fig10_intermodel_concordance`, `fig11_knowledge_resource_coverage`, `fig12_mentions_per_prescription`, `fig13_surface_frequency_long_tail`, `fig14_formulation_characteristics`, `fig15_normalization_case_study`, `fig16_provenance_completeness`.",
    ]
    (FIGURES / "FIGURE_RECOMMENDATIONS_FOR_8_PAGE_JBHI.md").write_text("\n".join(recommendations) + "\n", encoding="utf-8")
    return {row["figure_id"]: row for row in figures}


def render_single_figure(figure_id: str) -> None:
    src = load_sources()
    make_tables(src)
    make_figures(src)
    print(f"Regenerated {figure_id}")


def make_metrics_and_claims(src: dict[str, Any], table_manifest: dict[str, Any], figure_manifest: dict[str, Any]) -> dict[str, Any]:
    ACCOUNTING.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    layer_b = semantic_flags(src["layer_b"])
    mentions = src["mentions"]
    docs = src["docs"]
    qwen = src["qwen"]
    audit = audit762_with_pipeline(src)
    benchmark = src["benchmark"].copy()
    benchmark["score_float"] = pd.to_numeric(benchmark["primary_score"], errors="coerce")
    disp = layer_b["verification_decision"].value_counts()
    res = layer_b["resolution_level"].value_counts()
    final_metrics = {
        "generated_at": now(),
        "snapshot_id": "FINAL_893",
        "corpus": {
            "raw_prescriptions": 893,
            "deidentified_prescriptions": 893,
            "annotation_complete_prescriptions": 893,
            "pipeline_evaluated_prescriptions": int(len(docs)),
            "medication_bearing_prescriptions": int(mentions["document_uid"].nunique()),
            "zero_medication_prescriptions": int(len(docs) - mentions["document_uid"].nunique()),
            "medication_mentions": int(len(mentions)),
            "unique_medication_surface_forms": int(mentions["lexical_surface_normalized"].replace("", np.nan).nunique()),
        },
        "annotation_model_benchmark": {
            "status": "LEGACY_MODEL_SELECTION_BENCHMARK",
            "N": 125,
            "gpt55_best_primary_score": float(benchmark[benchmark["paper_display_name"].eq("GPT-5.5")]["score_float"].max()),
            "limitation": "Per-record outputs/model aliases are not fully reconstructed; do not generalize beyond this benchmark.",
        },
        "pipeline": {"retrieval_branches": ["R1 exact/fuzzy", "R2 BM25", "R3 SapBERT biomedical dense", "R4 RxNorm/RxNav", "R5 India KB"], "ranking": "unweighted RRF, k=60, top_n=20"},
        "verification": {k: int(v) for k, v in disp.items()},
        "resolution_levels": {k: int(v) for k, v in res.items()},
        "semantic_identifiers": {
            "rxcui_count": int(layer_b["has_rxcui"].sum()),
            "rxcui_percent": pct(layer_b["has_rxcui"].sum(), len(layer_b)),
            "atc_count": int(layer_b["has_atc"].sum()),
            "atc_percent": pct(layer_b["has_atc"].sum(), len(layer_b)),
            "local_brand_family_count": int(layer_b["has_local_brand_family"].sum()),
            "local_brand_family_percent": pct(layer_b["has_local_brand_family"].sum(), len(layer_b)),
            "local_product_count": int(layer_b["has_local_product"].sum()),
            "local_product_percent": pct(layer_b["has_local_product"].sum(), len(layer_b)),
            "any_local_semantic_id_count": int(layer_b["has_local_semantic_id"].sum()),
            "any_local_semantic_id_percent": pct(layer_b["has_local_semantic_id"].sum(), len(layer_b)),
            "no_semantic_identifier_count": int(layer_b["has_no_semantic_identifier"].sum()),
            "no_semantic_identifier_percent": pct(layer_b["has_no_semantic_identifier"].sum(), len(layer_b)),
        },
        "knowledge_resources": {"operational": src["resources"][src["resources"]["used_in_operational_v1"].eq("TRUE")]["resource"].tolist()},
        "qwen_semantic_audit": {
            "N": int(len(qwen)),
            "mapping_assessment": {k: int(v) for k, v in qwen["mapping_assessment"].value_counts().items()},
            "pipeline_decision_assessment": {k: int(v) for k, v in qwen["pipeline_decision_assessment"].value_counts().items()},
        },
        "terra_comparison": {"common_N": int(src["concordance"].loc[src["concordance"]["comparison"].str.contains("Terra vs Qwen"), "common_N"].iloc[0])},
        "gptoss_comparison": {"common_N": int(src["concordance"].loc[src["concordance"]["comparison"].str.contains("GPT-OSS vs Qwen"), "common_N"].iloc[0])},
        "intermodel_concordance": src["concordance"].to_dict("records"),
        "human_validation": {"status": "DEFERRED_FUTURE_WORK", "metrics": None},
        "public_release": {"real_prescriptions": "DEFERRED_PENDING_EXPERT_ADJUDICATION", "synthetic_examples": True},
        "reproducibility": {"script": "scripts/reproduce_paper_artifacts.py", "tables": len(table_manifest), "figures": len(figure_manifest)},
    }
    (METRICS / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    number_dict = final_metrics | {"tables": table_manifest, "figures": figure_manifest}
    (ACCOUNTING / "PAPER_NUMBER_DICTIONARY_FINAL.json").write_text(json.dumps(number_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    claims = []
    commit = git_commit()
    def add(claim_id: str, study: str, metric: str, value: Any, unit: str, denominator: Any, cohort: str, source_file: str, derivation: str, status: str, notes: str = "") -> None:
        claims.append({"claim_id": claim_id, "study": study, "metric": metric, "value": value, "unit": unit, "denominator": denominator, "cohort": cohort, "source_file": source_file, "source_column_or_derivation": derivation, "script": "scripts/reproduce_paper_artifacts.py", "snapshot_id": "FINAL_893", "git_commit": commit, "status": status, "notes": notes})

    c = final_metrics["corpus"]
    for key, value in c.items():
        add(f"B_CORPUS_{key}", "Study B", key, value, "count", 893 if "prescriptions" in key else len(mentions), "full893", "derived/layer_a_documents.csv; derived/layer_a_medication_mentions.csv", key, "VERIFIED_FINAL")
    for key, value in final_metrics["verification"].items():
        add(f"B_VERIFICATION_{key}", "Study B", f"verification_{key}", value, "mentions", len(layer_b), "full893", "derived/layer_b/layer_b_v1_1.csv", "verification_decision value_counts", "VERIFIED_FINAL")
    for key, value in final_metrics["semantic_identifiers"].items():
        add(f"B_ID_{key}", "Study B", key, value, "count_or_percent", len(layer_b), "full893", "derived/layer_b/layer_b_v1_1.csv", "semantic identifier derived flags", "VERIFIED_FINAL")
    add("A_BENCHMARK_GPT55", "Study A", "GPT-5.5 best primary benchmark score", final_metrics["annotation_model_benchmark"]["gpt55_best_primary_score"], "score", 125, "benchmark125", "paper_artifacts/benchmarking/annotation_model_leaderboard_reproduced.csv", "max GPT-5.5 primary_score", "LEGACY_LIMITED", final_metrics["annotation_model_benchmark"]["limitation"])
    for key, value in final_metrics["qwen_semantic_audit"]["mapping_assessment"].items():
        add(f"C_QWEN_MAPPING_{key}", "Study C", key, value, "mentions", 762, "audit762", "expert validation paper/04_local_open_models/qwen/qwen_final_validation762_results.csv", "mapping_assessment value_counts", "VERIFIED_HISTORICAL", "Automated audit, not human ground truth")
    for key, value in final_metrics["qwen_semantic_audit"]["pipeline_decision_assessment"].items():
        add(f"C_QWEN_ROUTING_{key}", "Study C", key, value, "mentions", 762, "audit762", "expert validation paper/04_local_open_models/qwen/qwen_final_validation762_results.csv", "pipeline_decision_assessment value_counts", "VERIFIED_HISTORICAL", "Automated audit, not human ground truth")
    add("HUMAN_VALIDATION_STATUS", "Human validation", "manual expert adjudication", "DEFERRED_FUTURE_WORK", "status", "", "none", "paper_artifacts/final_metrics/final_metrics.json", "human_validation.status", "FUTURE_WORK")
    write_csv(ACCOUNTING / "PAPER_CLAIM_REGISTRY_FINAL.csv", claims)
    return final_metrics


def make_handoff(final_metrics: dict[str, Any], table_manifest: dict[str, Any], figure_manifest: dict[str, Any]) -> None:
    PA.mkdir(parents=True, exist_ok=True)
    safe_claims = [
        "893 prescriptions were processed by the final pipeline.",
        "The final pipeline produced 3,027 medication mentions and explicit ACCEPT/HUMAN_REVIEW/NIL states.",
        "The pipeline preserves local Indian semantic identifiers and RxNorm/ATC identifiers where supported.",
        "Qwen independently classified 558/109/95 mappings as SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE on the 762-mention audit cohort.",
        "On the 125-document model-selection benchmark, GPT-5.5 achieved the highest score among evaluated systems under the study protocol.",
    ]
    forbidden = [
        "human-validated accuracy",
        "clinical correctness",
        "expert gold-standard performance",
        "general superiority of proprietary models",
        "patient-outcome benefit",
        "Recall@K/MRR without independent reference",
        "all mappings are correct",
        "RxNorm coverage means exact Indian product equivalence",
    ]
    limitations = [
        "single institution",
        "General Medicine OPD",
        "automated annotation",
        "no completed human semantic adjudication",
        "LLM-as-judge is not ground truth",
        "benchmark model-family generalization not supported",
        "India-specific KB coverage incomplete",
        "RxNorm is U.S.-oriented",
        "ATC mapping partial",
        "open high-recall candidate layer non-authoritative",
        "external validation absent",
        "HUMAN_REVIEW queue not fully adjudicated",
    ]
    handoff = {
        "verified_numbers": final_metrics,
        "methods": {
            "pipeline": "Six agents: de-identification, annotation creation, candidate retrieval, candidate ranking, evidence assessment, verification.",
            "retrieval": "R1 exact/fuzzy, R2 BM25, R3 SapBERT biomedical dense, R4 RxNorm/RxNav, R5 India-specific resources.",
            "ranking": "true candidate union; unweighted reciprocal rank fusion; k=60; top-20.",
            "semantic_audit": "Stratified automated Qwen audit; not human ground truth.",
        },
        "tables": table_manifest,
        "figures": figure_manifest,
        "safe_claims": safe_claims,
        "forbidden_claims": forbidden,
        "limitations": limitations,
        "future_work": ["independent expert adjudication", "multi-reviewer agreement", "expert calibration of automated semantic judges", "expert-adjudicated public benchmark"],
        "data_availability": "Code, configuration, aggregate artifacts, synthetic examples, and public-source KB build scripts may be released. Real de-identified prescription images and row-level annotations are deferred pending expert adjudication and governance approval.",
        "benchmark_caveats": "LEGACY_MODEL_SELECTION_BENCHMARK: per-record outputs/model aliases cannot be fully reconstructed; do not infer general proprietary-vs-open superiority.",
        "semantic_audit_results": final_metrics["qwen_semantic_audit"],
        "source_paths": ["configs/frozen/evaluation_final_893_manifest.json", "derived/layer_b/layer_b_v1_1.csv", "expert validation paper/04_local_open_models/qwen/qwen_final_validation762_results.csv"],
    }
    (PA / "PAPER_WRITER_HANDOFF_FINAL.json").write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = [
        "# Paper Writer Handoff Final",
        "",
        "## Verified Final Numbers",
        "",
        "Study A: benchmark125 is `LEGACY_MODEL_SELECTION_BENCHMARK`; GPT-5.5 best primary score is "
        f"{final_metrics['annotation_model_benchmark']['gpt55_best_primary_score']:.4f}.",
        f"Study B: 893 prescriptions, {final_metrics['corpus']['medication_bearing_prescriptions']} medication-bearing, {final_metrics['corpus']['zero_medication_prescriptions']} zero-medication, {final_metrics['corpus']['medication_mentions']} mentions, {final_metrics['corpus']['unique_medication_surface_forms']} unique normalized surfaces.",
        f"Study C: Qwen 762 audit: {final_metrics['qwen_semantic_audit']['mapping_assessment']}; routing {final_metrics['qwen_semantic_audit']['pipeline_decision_assessment']}.",
        "",
        "## Exact Methods Implementation",
        "",
        "- Six-agent pipeline: de-identification, annotation, retrieval, ranking, evidence assessment, verification.",
        "- Retrieval: R1 exact/fuzzy; R2 BM25; R3 SapBERT biomedical dense; R4 RxNorm/RxNav; R5 India-specific resources.",
        "- Ranking: true candidate union, unweighted RRF, `k=60`, top-20 to evidence.",
        "- Evidence and verification are deterministic and source/provenance-preserving.",
        "- Qwen audit is evaluation-only and not a seventh pipeline agent.",
        "",
        "## Current Manuscript Corrections",
        "",
        "- Replace `more than 1,000 prescriptions` with verified 893-prescription language.",
        "- Remove claims of independent clinical-expert review, expert-validated subset, expert primary semantic correctness estimate, and expert adjudication metrics.",
        "- Replace LLM-judge-calibrated-against-experts language with stratified automated LLM semantic audit language.",
        "- Replace empty/stale Results with the generated final tables.",
        "- Retitle expert-focused reliability sections toward semantic auditing and provenance.",
        "- Add future work: expert adjudication, multi-reviewer agreement, expert calibration, expert-adjudicated public benchmark.",
        "",
        "## Benchmark Wording",
        "",
        "Preferred: On the stratified 125-document model-selection benchmark, GPT-5.5 achieved the highest score among the evaluated systems under the study protocol. Do not claim general proprietary superiority.",
        "",
        "## Results Outline",
        "",
        "A. Corpus and Processing Coverage; B. Semantic Normalization and Verification Outcomes; C. Identifier Coverage; D. Annotation-Model Selection Benchmark; E. Stratified LLM-Based Semantic Audit; F. Inter-Model Concordance.",
        "",
        "## Discussion Outline",
        "",
        "A. From transcription to semantic medication representation; B. India-specific knowledge and formulation preservation; C. Selective verification and automated semantic auditing; D. Model-selection limits; E. Limitations/future expert validation.",
        "",
        "## Safe To Claim",
        *[f"- {claim}" for claim in safe_claims],
        "",
        "## Do Not Claim",
        *[f"- {claim}" for claim in forbidden],
        "",
        "## Data Availability",
        "",
        handoff["data_availability"],
        "",
        "## Limitations",
        *[f"- {item}" for item in limitations],
        "",
        "## Main Figure Recommendations",
        "- `paper_artifacts/figures/fig01_six_agent_architecture/figure.pdf`",
        "- `paper_artifacts/figures/fig02_study_cohort_flow/figure.pdf`",
        "- `paper_artifacts/figures/fig06_semantic_identifier_coverage/figure.pdf`",
        "- `paper_artifacts/figures/fig07_qwen_semantic_audit/figure.pdf`",
        "",
        "## Main Table Recommendations",
        "- `paper_artifacts/tables/table_final_corpus_accounting.csv`",
        "- `paper_artifacts/tables/table_pipeline_disposition.csv`",
        "- `paper_artifacts/tables/table_resolution_levels.csv`",
        "- `paper_artifacts/tables/table_qwen_semantic_audit.csv`",
    ]
    (PA / "PAPER_WRITER_HANDOFF_FINAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    change_list = [
        "# Current Manuscript Change List",
        "",
        "| Section | Old claim | Why stale | Verified replacement fact | Source artifact |",
        "| --- | --- | --- | --- | --- |",
        "| Abstract/Intro | more than 1,000 prescriptions | final verified corpus is 893 | 893 prescriptions processed | configs/frozen/evaluation_final_893_manifest.json |",
        "| Evaluation | expert-reviewed subset calibrates LLM judge | no completed expert reference exists | stratified automated Qwen semantic audit | paper_artifacts/tables/table_qwen_semantic_audit.csv |",
        "| Results | missing/stale quantitative results | final pipeline rerun completed | use generated final tables | paper_artifacts/tables/TABLE_MANIFEST_FINAL.csv |",
        "| Discussion | Reliability, Provenance, and Human Validation | human validation deferred | Reliability, Provenance, and Semantic Auditing | paper_artifacts/final_metrics/final_metrics.json |",
        "| Conclusion/C1 | independent expert validation | not completed | reproducible semantic audit and future expert adjudication | paper_artifacts/PAPER_WRITER_HANDOFF_FINAL.md |",
        "| Benchmark | proprietary models outperform open models | unsupported generalization | GPT-5.5 led this 125-document benchmark under this protocol | paper_artifacts/tables/table_annotation_model_benchmark_main.csv |",
    ]
    (PA / "CURRENT_MANUSCRIPT_CHANGE_LIST.md").write_text("\n".join(change_list) + "\n", encoding="utf-8")

    for name, text in {
        "DATA_AVAILABILITY_WORDING.md": handoff["data_availability"],
        "LIMITATIONS_WORDING.md": "\n".join(f"- {item}" for item in limitations),
        "FUTURE_WORK_WORDING.md": "\n".join(f"- {item}" for item in handoff["future_work"]),
    }.items():
        (PA / name).write_text(text + "\n", encoding="utf-8")


def make_bundle() -> dict[str, Any]:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    paths = [
        PA / "PAPER_WRITER_HANDOFF_FINAL.md",
        PA / "PAPER_WRITER_HANDOFF_FINAL.json",
        PA / "CURRENT_MANUSCRIPT_CHANGE_LIST.md",
        ACCOUNTING / "PAPER_CLAIM_REGISTRY_FINAL.csv",
        ACCOUNTING / "PAPER_NUMBER_DICTIONARY_FINAL.json",
        METRICS / "final_metrics.json",
        TABLES / "TABLE_MANIFEST_FINAL.csv",
        FIGURES / "FIGURE_MANIFEST_FINAL.csv",
        FIGURES / "FIGURE_RECOMMENDATIONS_FOR_8_PAGE_JBHI.md",
        PA / "DATA_AVAILABILITY_WORDING.md",
        PA / "LIMITATIONS_WORDING.md",
        PA / "FUTURE_WORK_WORDING.md",
    ]
    paths += list(TABLES.glob("table_*.csv")) + list(TABLES.glob("table_*.tex"))
    paths += [p for d in FIGURES.glob("fig*") for p in d.glob("*") if p.name in {"source.csv", "figure.svg", "figure.pdf", "figure.png", "caption_notes.md"}]
    manifest: dict[str, Any] = {"generated_at": now(), "files": {}}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        dst = BUNDLE / path.relative_to(PA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        manifest["files"][dst.relative_to(ROOT).as_posix()] = {"sha256": sha256_file(dst), "bytes": dst.stat().st_size}
    out = PA / "paper_writer_bundle_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_outputs() -> dict[str, Any]:
    required = [
        ACCOUNTING / "PAPER_CLAIM_REGISTRY_FINAL.csv",
        ACCOUNTING / "PAPER_NUMBER_DICTIONARY_FINAL.json",
        METRICS / "final_metrics.json",
        TABLES / "TABLE_MANIFEST_FINAL.csv",
        FIGURES / "FIGURE_MANIFEST_FINAL.csv",
        PA / "PAPER_WRITER_HANDOFF_FINAL.md",
        PA / "PAPER_WRITER_HANDOFF_FINAL.json",
        PA / "paper_writer_bundle_manifest.json",
    ]
    fig_manifest = read_csv(FIGURES / "FIGURE_MANIFEST_FINAL.csv")
    figure_files_ok = all((ROOT / p).exists() for col in ["svg", "pdf", "png"] for p in fig_manifest[col])
    missing = [rel(path) for path in required if not path.exists()]
    return {"missing_required": missing, "figure_files_ok": figure_files_ok, "tables": len(list(TABLES.glob("table_*.csv"))), "figures": len(fig_manifest), "pass": not missing and figure_files_ok}


def main() -> None:
    src = load_sources()
    if src["consistency"].get("blockers") != []:
        raise SystemExit("FINAL_RELEASE_BLOCKED = TRUE: full consistency audit has blockers")
    table_manifest = make_tables(src)
    figure_manifest = make_figures(src)
    metrics = make_metrics_and_claims(src, table_manifest, figure_manifest)
    make_handoff(metrics, table_manifest, figure_manifest)
    bundle = make_bundle()
    validation = validate_outputs()
    summary = {"generated_at": now(), "validation": validation, "bundle_files": len(bundle["files"])}
    (PA / "reproduction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation["pass"]:
        raise SystemExit(f"Artifact reproduction failed: {validation}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
