from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd


STAGE4_VERSION = "stage4_candidate_ranking_v0.1"
DEFAULT_CONFIG = {
    "ranking": {
        "method": "rrf",
        "rrf_k": 60,
        "top_n": 20,
        "branch_weights": {
            "R1_EXACT_FUZZY": 1.0,
            "R2_BM25": 1.0,
            "R3_BIOMEDICAL_DENSE": 1.0,
            "R4_RXNORM": 1.0,
            "R5_INDIA_KB": 1.0,
        },
    }
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def reciprocal_rank_fusion(ranks: dict[str, int], rrf_k: int = 60, weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    weights = weights or {}
    components = {}
    for branch, rank in sorted(ranks.items()):
        components[branch] = float(weights.get(branch, 1.0)) / (rrf_k + int(rank))
    return sum(components.values()), components


def weighted_rank_fusion(ranks: dict[str, int], rrf_k: int = 60, weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    return reciprocal_rank_fusion(ranks, rrf_k=rrf_k, weights=weights)


def _load_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or root / "configs/ranking/stage4_ranking_config.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True), encoding="utf-8")
        return DEFAULT_CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged["ranking"].update(config.get("ranking", {}))
    merged["ranking"]["branch_weights"].update(config.get("ranking", {}).get("branch_weights", {}))
    return merged


def _rank_mention(group: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = config["ranking"]
    rrf_k = int(ranking.get("rrf_k", 60))
    top_n = int(ranking.get("top_n", 20))
    weights = {k: float(v) for k, v in ranking.get("branch_weights", {}).items()}
    rows = []
    for candidate_id, cgroup in group.groupby("candidate_id", sort=False):
        ranks = {
            row.branch: int(float(row.rank))
            for row in cgroup.itertuples(index=False)
            if str(row.rank) and str(row.rank) != "nan"
        }
        scores = {
            row.branch: float(row.score)
            for row in cgroup.itertuples(index=False)
            if str(row.score) and str(row.score) != "nan"
        }
        score, components = reciprocal_rank_fusion(ranks, rrf_k=rrf_k, weights=weights)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_name": cgroup["candidate_name"].iloc[0],
                "candidate_type": cgroup["entity_type"].iloc[0],
                "entity_type": cgroup["entity_type"].iloc[0],
                "entity_id": cgroup["entity_id"].iloc[0],
                "ranking_score": score,
                "participating_branches": sorted(cgroup["branch"].unique()),
                "per_branch_rank": ranks,
                "per_branch_score": scores,
                "ranking_components": components,
                "source_state": "|".join(sorted(set(v for v in cgroup["source_state"].astype(str) if v))),
                "source_ids": "|".join(sorted(set(v for v in cgroup["authority"].astype(str) if v))),
                "evidence_ids": "|".join(sorted(set(v for cell in cgroup["provenance_evidence_ids"].astype(str) for v in cell.split("|") if v))),
                "raw_candidate_ids": "|".join(sorted(set(v for cell in cgroup["raw_candidate_id"].astype(str) for v in cell.split("|") if v))),
            }
        )
    ranked = sorted(rows, key=lambda row: (-row["ranking_score"], row["candidate_id"]))[:top_n]
    for i, row in enumerate(ranked, start=1):
        row["final_rank"] = i
    return ranked


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 4 Candidate Ranking Report",
        "",
        f"- version: {STAGE4_VERSION}",
        f"- generated_at: {summary['generated_at']}",
        f"- mentions_ranked: {summary['mentions_ranked']}",
        f"- engineering_default: {summary['engineering_default']}",
        f"- ready_for_evidence_assessment: {str(summary['READY_FOR_EVIDENCE_ASSESSMENT']).lower()}",
        "",
        "## Guardrails",
        "- Ranking consumed only `derived/retrieval/stage2c1_candidate_union.csv` and `stage2c1_branch_traces.csv`.",
        "- Ranking did not retrieve new candidates.",
        "- Ranking did not perform source-authority acceptance, evidence assessment, verification, or gold metric reporting.",
        "",
        "## Diagnostics",
        f"- median_input_pool_size: {summary['median_input_pool_size']}",
        f"- top_n_produced: {summary['top_n_produced']}",
        f"- rankability_rate: {summary['rankability_rate']:.3f}",
        f"- median_latency_ms: {summary['median_latency_ms']:.3f}",
        f"- single_branch_candidate_proportion: {summary['single_branch_candidate_proportion']:.3f}",
        f"- multi_branch_candidate_proportion: {summary['multi_branch_candidate_proportion']:.3f}",
        "",
        "## Outputs",
    ]
    for key, value in summary["paths"].items():
        lines.append(f"- {key}: {value}")
    if summary["blockers"]:
        lines.extend(["", "## Blockers"])
        for blocker in summary["blockers"]:
            lines.append(f"- {blocker}")
    (root / "rebuild/reports/STAGE4_CANDIDATE_RANKING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage4_ranking(root: Path | None = None, config_path: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    config = _load_config(root, config_path)
    union_path = root / "derived/retrieval/stage2c1_candidate_union.csv"
    trace_path = root / "derived/retrieval/stage2c1_branch_traces.csv"
    if not union_path.exists() or not trace_path.exists():
        raise FileNotFoundError("Stage 4 requires stage2c1 candidate union and branch traces.")
    union = pd.read_csv(union_path, dtype=str).fillna("")
    trace = pd.read_csv(trace_path, dtype=str).fillna("")
    candidate_trace = trace[trace["candidate_id"].astype(str) != ""].copy()
    candidate_trace["rank"] = pd.to_numeric(candidate_trace["rank"], errors="coerce")
    candidate_trace["score"] = pd.to_numeric(candidate_trace["score"], errors="coerce")

    out_dir = root / "derived/ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    latencies = []
    for mention_id, group in candidate_trace.groupby("mention_id", sort=False):
        start = time.perf_counter()
        ranked = _rank_mention(group, config)
        latencies.append((time.perf_counter() - start) * 1000)
        raw_text = group["raw_medication_text"].iloc[0]
        for row in ranked:
            row["mention_id"] = mention_id
            row["raw_medication_text"] = raw_text
            rows.append(row)
    ranked_df = pd.DataFrame(rows)
    for column in ["participating_branches", "per_branch_rank", "per_branch_score", "ranking_components"]:
        ranked_df[f"{column}_json"] = ranked_df[column].map(lambda value: json.dumps(value, sort_keys=True))
        ranked_df = ranked_df.drop(columns=[column])
    column_order = [
        "mention_id",
        "raw_medication_text",
        "candidate_id",
        "candidate_name",
        "candidate_type",
        "entity_type",
        "entity_id",
        "final_rank",
        "ranking_score",
        "participating_branches_json",
        "per_branch_rank_json",
        "per_branch_score_json",
        "ranking_components_json",
        "source_state",
        "source_ids",
        "evidence_ids",
        "raw_candidate_ids",
    ]
    ranked_df = ranked_df[column_order] if not ranked_df.empty else pd.DataFrame(columns=column_order)
    csv_path = out_dir / "ranked_candidates.csv"
    parquet_path = out_dir / "ranking_results.parquet"
    ranked_df.to_csv(csv_path, index=False)
    ranked_df.to_parquet(parquet_path, index=False)

    input_ids = set(union["candidate_id"])
    output_ids = set(ranked_df["candidate_id"])
    pool_sizes = union.groupby("mention_id")["candidate_id"].nunique().tolist() if not union.empty else []
    branch_counts = union["branches_returned"].str.split("|").explode().value_counts().to_dict() if not union.empty else {}
    branch_count_values = pd.to_numeric(union["branch_count"], errors="coerce").fillna(0)
    score_values = ranked_df["ranking_score"].astype(float).tolist() if not ranked_df.empty else []
    top1 = ranked_df[ranked_df["final_rank"] == 1] if not ranked_df.empty else ranked_df
    blockers = []
    if not output_ids <= input_ids:
        blockers.append("Ranking introduced candidate IDs not present in stage2c1 union.")
    if ranked_df.empty:
        blockers.append("No ranked candidates were produced.")
    summary = {
        "generated_at": _now_iso(),
        "version": STAGE4_VERSION,
        "mentions_ranked": int(ranked_df["mention_id"].nunique()) if not ranked_df.empty else 0,
        "ranking_methods_implemented": ["UNWEIGHTED_RRF", "WEIGHTED_RANK_FUSION"],
        "engineering_default": "UNWEIGHTED_RRF",
        "config": config["ranking"],
        "median_input_pool_size": float(median(pool_sizes)) if pool_sizes else 0.0,
        "top_n_produced": int(config["ranking"].get("top_n", 20)),
        "rankability_rate": (ranked_df["mention_id"].nunique() / union["mention_id"].nunique()) if not union.empty else 0.0,
        "median_latency_ms": float(median(latencies)) if latencies else 0.0,
        "branch_contribution_frequency": branch_counts,
        "single_branch_candidate_proportion": float((branch_count_values == 1).sum() / len(branch_count_values)) if len(branch_count_values) else 0.0,
        "multi_branch_candidate_proportion": float((branch_count_values > 1).sum() / len(branch_count_values)) if len(branch_count_values) else 0.0,
        "rank_score_distribution": {
            "min": min(score_values) if score_values else 0.0,
            "median": float(median(score_values)) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
        },
        "top1_candidate_type_distribution": top1["entity_type"].value_counts().to_dict() if not top1.empty else {},
        "top1_source_state_distribution": top1["source_state"].value_counts().to_dict() if not top1.empty else {},
        "input_subset_invariant_passed": bool(output_ids <= input_ids),
        "retrieved_new_candidates": False,
        "source_authority_acceptance_performed": False,
        "gold_metrics_reported": False,
        "blockers": blockers,
        "READY_FOR_EVIDENCE_ASSESSMENT": not blockers,
        "paths": {
            "ranked_candidates_csv": str(csv_path),
            "ranking_results_parquet": str(parquet_path),
            "ranking_summary": str(out_dir / "ranking_summary.json"),
            "ranking_report": str(root / "rebuild/reports/STAGE4_CANDIDATE_RANKING_REPORT.md"),
            "ranking_config": str(config_path or root / "configs/ranking/stage4_ranking_config.json"),
        },
    }
    summary_path = out_dir / "ranking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(root, summary)
    return summary

