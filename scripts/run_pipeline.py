#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.current_annotation_adapter import (  # noqa: E402
    build_context_bundle,
    build_document,
    build_mention,
    build_page,
    deduplicate_source_objects,
    discover_medication_objects,
    load_json,
)
from src.ranking.stage4 import reciprocal_rank_fusion  # noqa: E402
from src.utils.stable_ids import lexical_surface  # noqa: E402


STAGES = [
    "de-identification",
    "annotation",
    "layer-a",
    "candidate-retrieval",
    "candidate-union",
    "rrf",
    "evidence-assessment",
    "verification",
    "layer-b",
    "evaluation-export",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def normalize_surface(text: str) -> str:
    text = lexical_surface(text)
    text = re.sub(r"\b(tab|tablet|cap|capsule|syp|syrup)\b", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(mg|mcg|g|ml|iu)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_layer_a(annotations_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        data = load_json(annotation_path)
        collection_date = "SYNTHETIC"
        relpath = annotation_path.relative_to(ROOT).as_posix() if annotation_path.is_relative_to(ROOT) else str(annotation_path)
        doc = build_document(
            data=data,
            collection_date=collection_date,
            source_json_relpath=relpath,
            source_json_sha256="synthetic",
        )
        page = build_page(
            document=doc,
            page_number=1,
            raw_image_relpath="",
            raw_image_sha256="",
            anonymized_image_relpath="",
            anonymized_image_sha256="",
            lineage_status="UNVERIFIED_HEURISTIC",
        )
        build_context_bundle(doc, page, data)
        kept, _ = deduplicate_source_objects(discover_medication_objects(data))
        for source in kept:
            mention = build_mention(
                source=source,
                document=doc,
                page=page,
                source_json_relpath=relpath,
                source_json_sha256="synthetic",
            )
            if mention is None:
                continue
            item = mention.model_dump()
            item["source_document_id"] = doc.source_document_id
            rows.append(item)
    return rows


def branch_hits(surface: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    query = normalize_surface(surface)
    rows: list[dict[str, Any]] = []
    branches = ["R1_EXACT_FUZZY", "R2_BM25", "R3_BIOMEDICAL_DENSE", "R4_RXNORM", "R5_INDIA_KB"]
    for candidate in catalog:
        aliases = [normalize_surface(a) for a in candidate.get("aliases", [])]
        if query not in aliases and not any(query and (query in alias or alias in query) for alias in aliases):
            continue
        for branch_index, branch in enumerate(branches, start=1):
            rows.append(
                {
                    "branch": branch,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_name": candidate["candidate_name"],
                    "entity_type": candidate["entity_type"],
                    "rank": branch_index,
                    "score": round(1.0 / branch_index, 6),
                    "source_state": candidate.get("source_state", "SYNTHETIC_PUBLIC_EXAMPLE"),
                }
            )
    return rows


def candidate_retrieval(layer_a: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mention in layer_a:
        hits = branch_hits(mention["raw_medication_text"], catalog)
        if not hits:
            rows.append(
                {
                    "mention_id": mention["mention_id"],
                    "raw_medication_text": mention["raw_medication_text"],
                    "branch": "R5_INDIA_KB",
                    "candidate_id": "",
                    "candidate_name": "",
                    "entity_type": "",
                    "rank": "",
                    "score": "",
                    "source_state": "",
                }
            )
            continue
        for hit in hits:
            hit = dict(hit)
            hit["mention_id"] = mention["mention_id"]
            hit["raw_medication_text"] = mention["raw_medication_text"]
            rows.append(hit)
    return rows


def candidate_union(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in trace:
        if not row.get("candidate_id"):
            continue
        key = (row["mention_id"], row["candidate_id"])
        item = grouped.setdefault(
            key,
            {
                "mention_id": row["mention_id"],
                "raw_medication_text": row["raw_medication_text"],
                "candidate_id": row["candidate_id"],
                "candidate_name": row["candidate_name"],
                "entity_type": row["entity_type"],
                "branches_returned": [],
                "branch_count": 0,
                "source_state": row.get("source_state", ""),
            },
        )
        item["branches_returned"].append(row["branch"])
    out = []
    for item in grouped.values():
        branches = sorted(set(item["branches_returned"]))
        item["branches_returned"] = "|".join(branches)
        item["branch_count"] = len(branches)
        out.append(item)
    return sorted(out, key=lambda r: (r["mention_id"], r["candidate_id"]))


def rank_candidates(trace: list[dict[str, Any]], top_k: int, rrf_k: int) -> list[dict[str, Any]]:
    by_mention_candidate: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trace:
        if row.get("candidate_id"):
            by_mention_candidate.setdefault((row["mention_id"], row["candidate_id"]), []).append(row)
    by_mention: dict[str, list[dict[str, Any]]] = {}
    for (mention_id, candidate_id), rows in by_mention_candidate.items():
        ranks = {row["branch"]: int(row["rank"]) for row in rows if str(row.get("rank", "")).isdigit()}
        score, components = reciprocal_rank_fusion(ranks, rrf_k=rrf_k)
        first = rows[0]
        by_mention.setdefault(mention_id, []).append(
            {
                "mention_id": mention_id,
                "raw_medication_text": first["raw_medication_text"],
                "candidate_id": candidate_id,
                "candidate_name": first["candidate_name"],
                "entity_type": first["entity_type"],
                "ranking_score": score,
                "per_branch_rank_json": json.dumps(ranks, sort_keys=True),
                "ranking_components_json": json.dumps(components, sort_keys=True),
                "source_state": first.get("source_state", ""),
            }
        )
    ranked: list[dict[str, Any]] = []
    for rows in by_mention.values():
        for pos, row in enumerate(sorted(rows, key=lambda r: (-r["ranking_score"], r["candidate_id"]))[:top_k], start=1):
            row["final_rank"] = pos
            ranked.append(row)
    return ranked


def assess_evidence(ranked: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["candidate_id"]: item for item in catalog}
    rows = []
    for row in ranked:
        candidate = by_id.get(row["candidate_id"], {})
        lexical_match = normalize_surface(row["raw_medication_text"]) in [normalize_surface(a) for a in candidate.get("aliases", [])]
        rows.append(
            {
                "mention_id": row["mention_id"],
                "candidate_id": row["candidate_id"],
                "ranking_position": row["final_rank"],
                "lexical_status": "MATCH" if lexical_match else "PARTIAL",
                "formulation_status": "NOT_COMPARABLE",
                "provenance_status": "SYNTHETIC",
                "context_status": "NOT_COMPARABLE",
                "hard_conflicts_json": "[]",
                "missing_evidence_json": "[]",
                "candidate_facts_json": json.dumps(candidate, sort_keys=True),
            }
        )
    return rows


def verify(layer_a: list[dict[str, Any]], ranked: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_by_mention = {row["mention_id"]: row for row in ranked if row["final_rank"] == 1}
    evidence_by_pair = {(row["mention_id"], row["candidate_id"]): row for row in evidence}
    verification = []
    layer_b = []
    mention_by_id = {row["mention_id"]: row for row in layer_a}
    for mention_id, mention in mention_by_id.items():
        selected = ranked_by_mention.get(mention_id)
        if selected is None:
            decision = "NIL"
            evidence_row: dict[str, Any] = {}
        else:
            evidence_row = evidence_by_pair.get((mention_id, selected["candidate_id"]), {})
            decision = "ACCEPT" if evidence_row.get("lexical_status") == "MATCH" else "HUMAN_REVIEW"
        verification.append(
            {
                "mention_id": mention_id,
                "verification_decision": decision,
                "selected_candidate_id": selected.get("candidate_id", "") if selected else "",
                "review_reason_codes_json": "[]" if decision == "ACCEPT" else json.dumps(["INSUFFICIENT_SYNTHETIC_EVIDENCE"]),
            }
        )
        facts = json.loads(evidence_row.get("candidate_facts_json", "{}")) if evidence_row else {}
        layer_b.append(
            {
                "mention_id": mention_id,
                "document_uid": mention.get("document_uid", ""),
                "page_uid": mention.get("page_uid", ""),
                "raw_medication_text": mention.get("raw_medication_text", ""),
                "verification_decision": decision,
                "resolution_level": selected.get("entity_type", "NO_SUPPORTED_RESOLUTION") if selected else "NO_SUPPORTED_RESOLUTION",
                "primary_candidate_id": selected.get("candidate_id", "") if decision == "ACCEPT" and selected else "",
                "primary_candidate_name": selected.get("candidate_name", "") if decision == "ACCEPT" and selected else "",
                "ingredient_components_json": json.dumps(facts.get("ingredient_components", []), sort_keys=True),
                "rxnorm_rxcui": facts.get("rxnorm_rxcui", ""),
                "atc_codes_json": json.dumps(facts.get("atc_codes", []), sort_keys=True),
                "pipeline_version": "public_synthetic_smoke_v1",
            }
        )
    return verification, layer_b


def run(args: argparse.Namespace) -> int:
    config = read_json(Path(args.config))
    selected_stages = STAGES if args.stage == "all" else STAGES[STAGES.index(args.stage) :]
    if args.dry_run:
        print(json.dumps({"dry_run": True, "stages": selected_stages, "output": args.output}, indent=2))
        return 0

    annotations_dir = Path(args.annotations_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    layer_a = load_layer_a(annotations_dir)
    trace = candidate_retrieval(layer_a, config["candidate_catalog"])
    union = candidate_union(trace)
    ranked = rank_candidates(trace, int(config["ranking"]["top_k"]), int(config["ranking"]["rrf_k"]))
    evidence = assess_evidence(ranked, config["candidate_catalog"])
    verification, layer_b = verify(layer_a, ranked, evidence)

    write_csv(output / "layer_a_medication_mentions.csv", layer_a)
    write_csv(output / "candidate_branch_traces.csv", trace)
    write_csv(output / "candidate_union.csv", union)
    write_csv(output / "ranked_candidates.csv", ranked)
    write_csv(output / "evidence_assessments.csv", evidence)
    write_csv(output / "verification_results.csv", verification)
    write_csv(output / "layer_b.csv", layer_b)
    write_json(
        output / "evaluation_export.json",
        {
            "pipeline_version": config["pipeline_version"],
            "annotation_mode": "precomputed",
            "documents": len({row["source_document_id"] for row in layer_a}),
            "mentions": len(layer_a),
            "layer_b_rows": len(layer_b),
            "stages": selected_stages,
        },
    )
    print(json.dumps({"status": "SUCCESS", "output": str(output), "mentions": len(layer_a), "layer_b_rows": len(layer_b)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Public-safe AgenticMedNorm-IN pipeline wrapper.")
    parser.add_argument("--input", required=True, help="Input data root. Retained for CLI contract and future raw-image mode.")
    parser.add_argument("--annotations-dir", required=True, help="Directory of precomputed annotation JSON files.")
    parser.add_argument("--config", required=True, help="Pipeline configuration JSON.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--stage", default="all", choices=["all", *STAGES])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
