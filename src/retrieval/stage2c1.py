from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from src.knowledge.parsers import normalize_text
from src.retrieval.stage2c import BRANCHES, TOP_N, _pairwise_overlap, _read_csv


STAGE2C1_VERSION = "stage2c1_identity_harmonization_v0.1"
ENTITY_PREFIXES = {
    "BPROD": ("BRAND_PRODUCT", "brand_products.csv"),
    "BFAM": ("BRAND_FAMILY", "brand_families.csv"),
    "ING": ("INGREDIENT", "ingredients.csv"),
    "FORM": ("CLINICAL_FORMULATION", "clinical_formulations.csv"),
    "RXCUI": ("RXNORM_CONCEPT", "RxNorm/RxNav"),
    "CDSCO": ("OFFICIAL_SOURCE_RECORD", "cdsco_structured_records.csv"),
    "NLEM": ("OFFICIAL_SOURCE_RECORD", "nlem_entries.csv"),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prefix(candidate_id: str) -> str:
    return candidate_id.split("_", 1)[0] if "_" in candidate_id else candidate_id


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", normalize_text(text)) if len(tok) > 1]


def _clean_surface(text: str) -> str:
    norm = normalize_text(text)
    norm = re.sub(r"\b(tab|tabs|tablet|cap|caps|capsule|syp|syr|syrup|inj|injection|drop|drops|cream|oint|ointment)\b\.?", " ", norm)
    norm = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?|%)\b", " ", norm)
    return re.sub(r"\s+", " ", norm).strip() or normalize_text(text)


class IdentityMapper:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.ingredients = _read_csv(root / "knowledge/canonical/ingredients.csv")
        self.ingredients_by_norm = dict(zip(self.ingredients["normalized_name"], self.ingredients["ingredient_id"], strict=False))
        self.ingredient_names = dict(zip(self.ingredients["ingredient_id"], self.ingredients["canonical_name"], strict=False))
        self.nlem_to_ingredient = self._build_nlem_map()
        self.cdsco_to_ingredient = self._build_cdsco_map()

    def _build_nlem_map(self) -> dict[str, str]:
        nlem_path = self.root / "knowledge/canonical/nlem_entries.csv"
        if not nlem_path.exists():
            return {}
        nlem = _read_csv(nlem_path)
        return {
            row.nlem_entry_id: self.ingredients_by_norm[row.normalized_ingredient]
            for row in nlem.itertuples(index=False)
            if row.normalized_ingredient in self.ingredients_by_norm
        }

    def _build_cdsco_map(self) -> dict[str, str]:
        records_path = self.root / "knowledge/canonical/cdsco_structured_records.csv"
        components_path = self.root / "knowledge/canonical/cdsco_formulation_components.csv"
        if not records_path.exists():
            return {}
        records = _read_csv(records_path)
        mapping = {
            row.cdsco_record_id: self.ingredients_by_norm[row.normalized_drug_name]
            for row in records.itertuples(index=False)
            if row.normalized_drug_name in self.ingredients_by_norm
        }
        if components_path.exists():
            components = _read_csv(components_path)
            counts = components.groupby("cdsco_record_id")["normalized_ingredient"].agg(list)
            for record_id, norms in counts.items():
                usable = [norm for norm in norms if norm in self.ingredients_by_norm]
                if len(norms) == 1 and len(usable) == 1:
                    mapping.setdefault(record_id, self.ingredients_by_norm[usable[0]])
        return mapping

    def map_id(self, candidate_id: str, candidate_type: str, candidate_name: str = "") -> dict[str, str]:
        if not candidate_id:
            return {
                "raw_candidate_id": "",
                "entity_type": "",
                "entity_id": "",
                "candidate_id": "",
                "candidate_name": candidate_name,
                "identity_mapping_basis": "",
            }
        prefix = _prefix(candidate_id)
        if prefix == "NLEM" and candidate_id in self.nlem_to_ingredient:
            entity_id = self.nlem_to_ingredient[candidate_id]
            return {
                "raw_candidate_id": candidate_id,
                "entity_type": "INGREDIENT",
                "entity_id": entity_id,
                "candidate_id": f"ENTITY:INGREDIENT:{entity_id}",
                "candidate_name": self.ingredient_names.get(entity_id, candidate_name),
                "identity_mapping_basis": "nlem_normalized_ingredient_exact_local_ingredient",
            }
        if prefix == "CDSCO" and candidate_id in self.cdsco_to_ingredient:
            entity_id = self.cdsco_to_ingredient[candidate_id]
            return {
                "raw_candidate_id": candidate_id,
                "entity_type": "INGREDIENT",
                "entity_id": entity_id,
                "candidate_id": f"ENTITY:INGREDIENT:{entity_id}",
                "candidate_name": self.ingredient_names.get(entity_id, candidate_name),
                "identity_mapping_basis": "cdsco_drug_or_single_component_exact_local_ingredient",
            }
        entity_type = ENTITY_PREFIXES.get(prefix, ("UNKNOWN", ""))[0]
        entity_id = candidate_id.replace("RXCUI_", "", 1) if prefix == "RXCUI" else candidate_id
        return {
            "raw_candidate_id": candidate_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "candidate_id": f"ENTITY:{entity_type}:{entity_id}",
            "candidate_name": candidate_name,
            "identity_mapping_basis": "native_canonical_entity_id" if entity_type != "OFFICIAL_SOURCE_RECORD" else "no_supported_entity_crosswalk_preserved_as_source_record",
        }


class StructuredIndiaRetriever:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[dict[str, str]] = []
        self.exact: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.by_token: dict[str, set[int]] = defaultdict(set)
        self.by_initial: dict[str, set[int]] = defaultdict(set)
        self._load_entries()

    def _add(self, candidate_id: str, candidate_name: str, candidate_type: str, field: str, text: str, source_state: str, authority: str, evidence_id: str = "") -> None:
        norm = normalize_text(text)
        if not norm or not candidate_id:
            return
        entry = {
            "candidate_id": candidate_id,
            "candidate_name": candidate_name or text,
            "candidate_type": candidate_type,
            "matched_field": field,
            "matched_alias": text,
            "source_state": source_state or "UNKNOWN",
            "authority": authority or "",
            "provenance_evidence_ids": evidence_id,
            "norm": norm,
        }
        idx = len(self.entries)
        self.entries.append(entry)
        self.exact[norm].append(entry)
        self.by_initial[norm[:1]].add(idx)
        for token in _tokenize(norm)[:6]:
            self.by_token[token].add(idx)

    def _load_entries(self) -> None:
        canonical = self.root / "knowledge/canonical"
        products = _read_csv(canonical / "brand_products.csv")
        families = _read_csv(canonical / "brand_families.csv")
        ingredients = _read_csv(canonical / "ingredients.csv")
        formulations = _read_csv(canonical / "clinical_formulations.csv")
        source_evidence = _read_csv(canonical / "source_evidence.csv")
        evidence_by_entity = source_evidence.groupby("entity_id")["evidence_id"].first().to_dict() if not source_evidence.empty else {}

        for row in products.itertuples(index=False):
            self._add(row.brand_product_id, row.raw_brand_name, "BrandProduct", "india_structured.brand_product.raw_brand_name", row.raw_brand_name, row.kg_state, row.authority, evidence_by_entity.get(row.brand_product_id, ""))
            self._add(row.brand_product_id, row.raw_brand_name, "BrandProduct", "india_structured.brand_product.normalized_brand_name", row.normalized_brand_name, row.kg_state, row.authority, evidence_by_entity.get(row.brand_product_id, ""))
        for row in families.itertuples(index=False):
            self._add(row.brand_family_id, row.canonical_name, "BrandFamily", "india_structured.brand_family.canonical_name", row.canonical_name, row.kg_state, row.authority)
        for row in ingredients.itertuples(index=False):
            self._add(row.ingredient_id, row.canonical_name, "Ingredient", "india_structured.ingredient.canonical_name", row.canonical_name, row.kg_state, row.authority)
        for row in formulations.itertuples(index=False):
            label = row.normalized_component_signature
            self._add(row.formulation_id, label, "ClinicalFormulation", "india_structured.formulation.normalized_component_signature", label, row.kg_state, row.authority)

        nlem_path = canonical / "nlem_entries.csv"
        if nlem_path.exists():
            nlem = _read_csv(nlem_path)
            for row in nlem.itertuples(index=False):
                text = " ".join([row.ingredient, row.strength, row.dosage_form, row.section_category])
                self._add(row.nlem_entry_id, row.ingredient, "NLEMEntry", "india_structured.nlem.ingredient_strength_form", text, "AUTHORITATIVE_NLEM_CONTEXT", "NLEM_2022", row.evidence_id)
        cdsco_path = canonical / "cdsco_structured_records.csv"
        if cdsco_path.exists():
            cdsco = _read_csv(cdsco_path)
            for row in cdsco.itertuples(index=False):
                text = " ".join([row.drug_name, row.source_document_title, row.applicant_or_company])
                self._add(row.cdsco_record_id, row.drug_name, "CDSCORecord", "india_structured.cdsco.drug_name", text, "AUTHORITATIVE_CDSCO_CONTEXT", row.source_id, row.evidence_id)

    def _choice_indices(self, query: str) -> list[int]:
        token_sets = [self.by_token.get(token, set()) for token in _tokenize(query)]
        token_sets = [items for items in token_sets if items]
        if token_sets:
            choices = set().union(*token_sets)
        else:
            choices = self.by_initial.get(query[:1], set())
        return sorted(choices)[:25000]

    def search(self, surface: str, top_n: int = TOP_N) -> tuple[list[dict[str, Any]], float]:
        start = time.perf_counter()
        query = normalize_text(surface)
        cleaned = _clean_surface(surface)
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for q in dict.fromkeys([query, cleaned]):
            for entry in self.exact.get(q, []):
                if entry["candidate_id"] in seen:
                    continue
                seen.add(entry["candidate_id"])
                item = dict(entry)
                item["score"] = 1.0
                item["score_semantics"] = "india_structured_exact_normalized_lookup"
                hits.append(item)
        if len(hits) < top_n:
            choices = {idx: self.entries[idx]["norm"] for idx in self._choice_indices(cleaned)}
            for _, score, idx in process.extract(cleaned, choices, scorer=fuzz.WRatio, limit=top_n * 4, score_cutoff=84):
                entry = self.entries[idx]
                if entry["candidate_id"] in seen:
                    continue
                seen.add(entry["candidate_id"])
                item = dict(entry)
                item["score"] = float(score) / 100.0
                item["score_semantics"] = "india_structured_field_rapidfuzz_wratio_0_to_1"
                hits.append(item)
                if len(hits) >= top_n:
                    break
        hits = sorted(hits, key=lambda item: (-float(item["score"]), item["candidate_id"]))[:top_n]
        return hits, (time.perf_counter() - start) * 1000


def _replace_r5(trace: pd.DataFrame, root: Path) -> pd.DataFrame:
    retriever = StructuredIndiaRetriever(root)
    mention_frame = trace[["mention_id", "raw_medication_text"]].drop_duplicates().reset_index(drop=True)
    surface_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
    rows: list[dict[str, Any]] = []
    surfaces = mention_frame["raw_medication_text"].drop_duplicates().tolist()
    for i, surface in enumerate(surfaces, start=1):
        if i % 50 == 0:
            print(f"stage2c1 structured R5 surfaces processed: {i}/{len(surfaces)}", flush=True)
        surface_cache[surface] = retriever.search(surface, TOP_N)
    for mention in mention_frame.itertuples(index=False):
        hits, latency_ms = surface_cache[mention.raw_medication_text]
        if not hits:
            rows.append(
                {
                    "mention_id": mention.mention_id,
                    "raw_medication_text": mention.raw_medication_text,
                    "branch": "R5_INDIA_KB",
                    "candidate_id": "",
                    "candidate_name": "",
                    "candidate_type": "",
                    "rank": "",
                    "score": "",
                    "score_semantics": "",
                    "matched_field": "",
                    "matched_alias": "",
                    "source_state": "",
                    "authority": "",
                    "provenance_evidence_ids": "",
                    "latency_ms": round(latency_ms, 3),
                    "status": "EMPTY",
                    "error": "",
                }
            )
            continue
        for rank, hit in enumerate(hits, start=1):
            row = {key: hit.get(key, "") for key in ["candidate_id", "candidate_name", "candidate_type", "score", "score_semantics", "matched_field", "matched_alias", "source_state", "authority", "provenance_evidence_ids"]}
            row.update(
                {
                    "mention_id": mention.mention_id,
                    "raw_medication_text": mention.raw_medication_text,
                    "branch": "R5_INDIA_KB",
                    "rank": rank,
                    "latency_ms": round(latency_ms, 3),
                    "status": "SUCCESS",
                    "error": "",
                }
            )
            rows.append(row)
    return pd.concat([trace[trace["branch"] != "R5_INDIA_KB"], pd.DataFrame(rows)], ignore_index=True)


def _canonicalize_trace(trace: pd.DataFrame, mapper: IdentityMapper) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for row in trace.to_dict("records"):
        mapped = mapper.map_id(row.get("candidate_id", ""), row.get("candidate_type", ""), row.get("candidate_name", ""))
        new_row = dict(row)
        new_row["raw_candidate_id"] = mapped["raw_candidate_id"]
        new_row["candidate_id"] = mapped["candidate_id"]
        new_row["candidate_name"] = mapped["candidate_name"] or row.get("candidate_name", "")
        new_row["entity_type"] = mapped["entity_type"]
        new_row["entity_id"] = mapped["entity_id"]
        new_row["identity_mapping_basis"] = mapped["identity_mapping_basis"]
        rows.append(new_row)
    enriched = pd.DataFrame(rows)
    candidate_rows = enriched[enriched["candidate_id"].astype(str) != ""].copy()
    empty_rows = enriched[enriched["candidate_id"].astype(str) == ""].copy()
    candidate_rows["score_numeric"] = pd.to_numeric(candidate_rows["score"], errors="coerce").fillna(0.0)
    candidate_rows["rank_numeric"] = pd.to_numeric(candidate_rows["rank"], errors="coerce").fillna(999999)
    group_cols = ["mention_id", "raw_medication_text", "branch", "candidate_id"]
    candidate_rows = candidate_rows.sort_values(group_cols + ["rank_numeric", "score_numeric"], ascending=[True, True, True, True, True, False], kind="stable")
    best = candidate_rows.drop_duplicates(group_cols, keep="first").copy()

    def join_unique(series: pd.Series, limit: int | None = None) -> str:
        values = sorted(set(v for cell in series.astype(str) for v in cell.split("|") if v))
        if limit is not None:
            values = values[:limit]
        return "|".join(values)

    aggs = candidate_rows.groupby(group_cols, sort=False).agg(
        matched_alias_all=("matched_alias", lambda s: " || ".join(sorted(set(v for v in s.astype(str) if v))[:12])),
        raw_candidate_id_all=("raw_candidate_id", join_unique),
        provenance_evidence_ids_all=("provenance_evidence_ids", join_unique),
        collapsed_surface_hit_count=("candidate_id", "size"),
        collapsed_raw_candidate_count=("raw_candidate_id", lambda s: len(set(v for cell in s.astype(str) for v in cell.split("|") if v))),
    ).reset_index()
    best = best.merge(aggs, on=group_cols, how="left")
    best["matched_alias"] = best["matched_alias_all"]
    best["raw_candidate_id"] = best["raw_candidate_id_all"]
    best["provenance_evidence_ids"] = best["provenance_evidence_ids_all"]
    best["rank"] = best["rank_numeric"].astype(int)
    best["score"] = best["score_numeric"].astype(float)
    best = best.drop(columns=["score_numeric", "rank_numeric", "matched_alias_all", "raw_candidate_id_all", "provenance_evidence_ids_all"])
    collapse_records = best[["mention_id", "branch", "candidate_id", "raw_candidate_id", "collapsed_surface_hit_count", "collapsed_raw_candidate_count"]].rename(columns={"raw_candidate_id": "raw_candidate_ids"})
    collapsed = pd.concat([best, empty_rows], ignore_index=True) if not best.empty else empty_rows
    branch_order = {branch: i for i, branch in enumerate(BRANCHES)}
    collapsed["_branch_order"] = collapsed["branch"].map(branch_order)
    collapsed["_rank_order"] = pd.to_numeric(collapsed["rank"], errors="coerce").fillna(0)
    collapsed = collapsed.sort_values(["mention_id", "_branch_order", "_rank_order"], kind="stable").drop(columns=["_branch_order", "_rank_order"])
    return collapsed, collapse_records


def _union(trace: pd.DataFrame) -> pd.DataFrame:
    candidates = trace[trace["candidate_id"].astype(str) != ""].copy()
    rows = []
    for (mention_id, candidate_id), group in candidates.groupby(["mention_id", "candidate_id"], sort=False):
        rows.append(
            {
                "mention_id": mention_id,
                "raw_medication_text": group["raw_medication_text"].iloc[0],
                "candidate_id": candidate_id,
                "candidate_name": group["candidate_name"].iloc[0],
                "candidate_type": group["entity_type"].iloc[0],
                "entity_type": group["entity_type"].iloc[0],
                "entity_id": group["entity_id"].iloc[0],
                "branches_returned": "|".join(sorted(group["branch"].unique())),
                "branch_count": group["branch"].nunique(),
                "source_states": "|".join(sorted(set(v for v in group["source_state"].astype(str) if v))),
                "source_ids": "|".join(sorted(set(v for v in group["authority"].astype(str) if v))),
                "evidence_ids": "|".join(sorted(set(v for cell in group["provenance_evidence_ids"].astype(str) for v in cell.split("|") if v))),
                "raw_candidate_ids": "|".join(sorted(set(v for cell in group["raw_candidate_id"].astype(str) for v in cell.split("|") if v))),
                "is_ranked": "false",
            }
        )
    return pd.DataFrame(rows)


def _audit_csv(before: pd.DataFrame, after: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for branch, group in before[before["candidate_id"].astype(str) != ""].groupby("branch"):
        for prefix, pgroup in group.groupby(group["candidate_id"].map(_prefix)):
            entity_type, source_table = ENTITY_PREFIXES.get(prefix, ("UNKNOWN", ""))
            rows.append(
                {
                    "stage": "stage2c_raw",
                    "branch": branch,
                    "candidate_id_prefix": prefix,
                    "candidate_type_values": "|".join(sorted(pgroup["candidate_type"].unique())),
                    "canonical_entity_type": entity_type,
                    "source_table_or_namespace": source_table,
                    "rows": len(pgroup),
                    "unique_raw_candidate_ids": pgroup["candidate_id"].nunique(),
                    "example_candidate_ids": "|".join(pgroup["candidate_id"].drop_duplicates().head(5)),
                    "id_represents": _id_represents(prefix),
                }
            )
    for branch, group in after[after["candidate_id"].astype(str) != ""].groupby("branch"):
        for entity_type, egroup in group.groupby("entity_type"):
            rows.append(
                {
                    "stage": "stage2c1_canonicalized",
                    "branch": branch,
                    "candidate_id_prefix": "ENTITY",
                    "candidate_type_values": "|".join(sorted(egroup["candidate_type"].unique())),
                    "canonical_entity_type": entity_type,
                    "source_table_or_namespace": "canonical_candidate_key",
                    "rows": len(egroup),
                    "unique_raw_candidate_ids": egroup["raw_candidate_id"].nunique(),
                    "example_candidate_ids": "|".join(egroup["candidate_id"].drop_duplicates().head(5)),
                    "id_represents": f"ENTITY:{entity_type}:<entity_id>",
                }
            )
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def _id_represents(prefix: str) -> str:
    return {
        "BPROD": "Indian open-dataset brand product record",
        "BFAM": "Canonical brand-family entity",
        "ING": "Canonical ingredient entity",
        "FORM": "Canonical clinical formulation entity",
        "RXCUI": "RxNorm concept identifier",
        "CDSCO": "CDSCO structured official source record; maps to ingredient only with exact crosswalk",
        "NLEM": "NLEM 2022 structured source record; maps to ingredient only with exact crosswalk",
    }.get(prefix, "Unknown namespace")


def _median_union(path: Path) -> float:
    if not path.exists():
        return 0.0
    df = _read_csv(path)
    values = df.groupby("mention_id")["candidate_id"].nunique().tolist() if not df.empty else []
    return float(median(values)) if values else 0.0


def _write_report(root: Path, summary: dict[str, Any], audit: pd.DataFrame) -> None:
    before_overlap = pd.read_csv(root / "derived/retrieval/stage2c_pairwise_overlap.csv", dtype=str).fillna("")
    after_overlap = pd.read_csv(root / "derived/retrieval/stage2c1_pairwise_overlap.csv", dtype=str).fillna("")
    lines = [
        "# Candidate Identity Namespace Audit",
        "",
        f"- version: {STAGE2C1_VERSION}",
        f"- generated_at: {summary['generated_at']}",
        f"- stage2c1_ready_for_ranking: {str(summary['READY_FOR_RANKING']).lower()}",
        "",
        "## Candidate ID Meaning By Branch",
    ]
    for row in audit.to_dict("records"):
        if row["stage"] != "stage2c_raw":
            continue
        lines.append(
            f"- {row['branch']} `{row['candidate_id_prefix']}_*`: {row['id_represents']} "
            f"({row['rows']} rows, {row['unique_raw_candidate_ids']} unique; source `{row['source_table_or_namespace']}`)."
        )
    lines.extend(
        [
            "",
            "## Identity Decisions",
            f"- R3 namespace issue found: {str(summary['r3_namespace_issue_found']).lower()}",
            f"- R5 duplicate-BM25 issue found: {str(summary['r2_r5_near_duplication_found']).lower()}",
            f"- R5 independence status: {summary['r2_r5_independence_status']}",
            f"- Aliases/surfaces collapsed rows: {summary['aliases_surfaces_collapsed_rows']}",
            f"- Median union before: {summary['median_union_before']}",
            f"- Median union after: {summary['median_union_after']}",
            "",
            "R3 near-zero overlap was largely semantic-level divergence, with dense retrieval returning CDSCO/NLEM official context records and ingredient concepts while R2/R5 focused on Indian brand products. Stage 2C.1 maps official records to ingredients only where an exact local ingredient crosswalk exists and otherwise preserves source records explicitly.",
            "",
            "R2 remains BM25 over `retrieval_documents.csv`. R5 has been rebuilt for Stage 2C.1 as structured India-KB lookups over brand product, brand family, ingredient, formulation, NLEM, and CDSCO fields using exact and controlled fuzzy field matching, not BM25 over the R2 corpus.",
            "",
            "R4 low return rate is expected: the branch returns actual RxNorm concepts from RxNorm/RxNav/crosswalk coverage, while most Layer-A surfaces are Indian brands or abbreviations. Stage 2C.1 does not inflate RxNorm coverage by inventing local-brand concepts.",
            "",
            "## Pairwise Overlap Before",
        ]
    )
    for row in before_overlap.to_dict("records"):
        lines.append(f"- {row['branch_a']} vs {row['branch_b']}: jaccard={row['jaccard']}")
    lines.append("")
    lines.append("## Pairwise Overlap After")
    for row in after_overlap.to_dict("records"):
        lines.append(f"- {row['branch_a']} vs {row['branch_b']}: jaccard={row['jaccard']}")
    lines.extend(["", "## Outputs"])
    for key, value in summary["paths"].items():
        lines.append(f"- {key}: {value}")
    (root / "rebuild/reports/CANDIDATE_IDENTITY_NAMESPACE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2c1(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    derived = root / "derived/retrieval"
    trace_before = _read_csv(derived / "stage2c_branch_traces.csv")
    union_before_path = derived / "stage2c_candidate_union.csv"
    r2_r5_before = set(trace_before[(trace_before["branch"] == "R2_BM25") & (trace_before["candidate_id"].astype(str) != "")]["candidate_id"]) & set(
        trace_before[(trace_before["branch"] == "R5_INDIA_KB") & (trace_before["candidate_id"].astype(str) != "")]["candidate_id"]
    )
    r5_rebuilt = _replace_r5(trace_before, root)
    mapper = IdentityMapper(root)
    trace_after, collapse = _canonicalize_trace(r5_rebuilt, mapper)

    trace_path = derived / "stage2c1_branch_traces.csv"
    trace_after.to_csv(trace_path, index=False)
    union = _union(trace_after)
    union_path = derived / "stage2c1_candidate_union.csv"
    union.to_csv(union_path, index=False)
    overlap = _pairwise_overlap(trace_after)
    overlap_path = derived / "stage2c1_pairwise_overlap.csv"
    overlap.to_csv(overlap_path, index=False)
    collapse_path = derived / "stage2c1_identity_collapse_log.csv"
    collapse.to_csv(collapse_path, index=False)
    audit_path = derived / "candidate_identity_namespace_audit.csv"
    audit = _audit_csv(trace_before, trace_after, audit_path)

    union_sizes = union.groupby("mention_id")["candidate_id"].nunique().tolist() if not union.empty else []
    before_median = _median_union(union_before_path)
    after_median = float(median(union_sizes)) if union_sizes else 0.0
    status = trace_after.groupby(["mention_id", "branch"], as_index=False).agg(status=("status", "first"), candidates=("candidate_id", lambda s: int((s.astype(str) != "").sum())))
    status_counts = status.groupby("branch")["status"].value_counts().rename("count").reset_index().to_dict("records")
    r3_source_record_raw = trace_before[(trace_before["branch"] == "R3_BIOMEDICAL_DENSE") & (trace_before["candidate_id"].str.startswith(("CDSCO_", "NLEM_"), na=False))]
    r3_source_record_after = trace_after[(trace_after["branch"] == "R3_BIOMEDICAL_DENSE") & (trace_after["raw_candidate_id"].str.contains("CDSCO_|NLEM_", regex=True, na=False))]
    summary = {
        "generated_at": _now_iso(),
        "version": STAGE2C1_VERSION,
        "mentions_processed": int(trace_after["mention_id"].nunique()),
        "unique_surfaces": int(trace_after["raw_medication_text"].nunique()),
        "r3_namespace_issue_found": bool(not r3_source_record_raw.empty),
        "r3_source_record_rows_before": int(len(r3_source_record_raw)),
        "r3_source_record_rows_after_entity_mapping": int(len(r3_source_record_after)),
        "aliases_surfaces_collapsed_rows": int(collapse["collapsed_surface_hit_count"].sub(1).clip(lower=0).sum()) if not collapse.empty else 0,
        "raw_candidate_ids_collapsed_rows": int(collapse["collapsed_raw_candidate_count"].sub(1).clip(lower=0).sum()) if not collapse.empty else 0,
        "median_union_before": before_median,
        "median_union_after": after_median,
        "branch_status_counts": status_counts,
        "r2_r5_near_duplication_found": bool(len(r2_r5_before) > 0),
        "r2_r5_independence_status": "R5_REFACTORED_STAGE2C1_STRUCTURED_INDIA_KB_NOT_R2_BM25",
        "r2_corpus": "knowledge/canonical/retrieval_documents.csv",
        "r2_scoring": "BM25 Okapi over deterministic retrieval documents",
        "r5_corpus": "structured brand_products, brand_families, ingredients, clinical_formulations, NLEM, CDSCO fields",
        "r5_scoring": "exact normalized field lookup plus controlled rapidfuzz WRatio over structured India-KB fields",
        "r4_low_return_explanation": "Expected for Indian brand-heavy surfaces; R4 returns only actual RxNorm concepts from RxNorm/RxNav/crosswalk coverage and does not invent Indian brand concepts.",
        "READY_FOR_RANKING": bool(after_median > 0 and not union.empty),
        "paths": {
            "stage2c1_branch_traces": str(trace_path),
            "stage2c1_candidate_union": str(union_path),
            "stage2c1_pairwise_overlap": str(overlap_path),
            "identity_collapse_log": str(collapse_path),
            "namespace_audit_csv": str(audit_path),
            "namespace_audit_report": str(root / "rebuild/reports/CANDIDATE_IDENTITY_NAMESPACE_AUDIT.md"),
        },
    }
    summary_path = derived / "stage2c1_summary.json"
    summary["paths"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(root, summary, audit)
    return summary
