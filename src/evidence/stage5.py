from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.knowledge.parsers import normalize_text, normalize_strength


STAGE5_VERSION = "stage5_evidence_assessment_v0.1"
EVIDENCE_CONFIG_DEFAULT = {"evidence": {"depth": 20, "strong_lexical_threshold": 0.9}}
FORM_SYNONYMS = {
    "tab": "tablet",
    "tabs": "tablet",
    "tablet": "tablet",
    "cap": "capsule",
    "caps": "capsule",
    "capsule": "capsule",
    "syp": "syrup",
    "syr": "syrup",
    "syrup": "syrup",
    "inj": "injection",
    "injection": "injection",
    "drop": "drops",
    "drops": "drops",
    "cream": "cream",
    "gel": "gel",
    "ointment": "ointment",
    "oint": "ointment",
}
RELEASE_MODIFIERS = {"sr": "SR", "cr": "CR", "er": "ER", "xr": "ER", "mr": "MR", "ec": "EC", "dr": "DR", "dsr": "DSR", "od": "OD"}


@dataclass(frozen=True)
class ParsedStrength:
    value: str
    unit: str
    raw: str
    normalized: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "||".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, **kwargs).fillna("")


def normalize_dosage_form(text: object) -> str:
    norm = normalize_text(text)
    for token in re.split(r"[^a-z0-9]+", norm):
        if token in FORM_SYNONYMS:
            return FORM_SYNONYMS[token]
    return ""


def extract_release_modifier(text: object) -> str:
    tokens = re.split(r"[^a-z0-9]+", normalize_text(text))
    for token in tokens:
        if token in RELEASE_MODIFIERS:
            return RELEASE_MODIFIERS[token]
    return ""


def parse_strength(text: object) -> ParsedStrength | None:
    norm = normalize_text(text)
    if not norm:
        return None
    match = re.search(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|iu|units?|%)\b", norm)
    if not match:
        return None
    value = match.group("value").rstrip("0").rstrip(".") if "." in match.group("value") else match.group("value")
    unit = match.group("unit").replace("units", "unit")
    raw = match.group(0)
    return ParsedStrength(value=value, unit=unit, raw=raw, normalized=f"{value}{unit}")


def parse_observed_formulation(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    getter = row.get if isinstance(row, dict) else row.get
    raw = getter("raw_medication_text", "")
    strength_text = getter("raw_strength_text", "")
    dosage_text = getter("raw_dosage_text", "")
    surface = " ".join([str(raw), str(strength_text), str(dosage_text)])
    strength = parse_strength(strength_text) or parse_strength(raw)
    form = normalize_dosage_form(raw) or normalize_dosage_form(dosage_text)
    release = extract_release_modifier(raw)
    cleaned = normalize_text(raw)
    fdc_markers = bool(re.search(r"\s\+\s|/| with ", cleaned))
    component_count = None
    if fdc_markers:
        pieces = [p for p in re.split(r"\s+\+\s+|\s+with\s+", cleaned) if p.strip()]
        component_count = len(pieces) if len(pieces) > 1 else None
    return {
        "raw_medication_text": raw,
        "observed_strength_raw": strength.raw if strength else "",
        "observed_strength_normalized": strength.normalized if strength else "",
        "observed_strength_value": strength.value if strength else "",
        "observed_strength_unit": strength.unit if strength else "",
        "observed_dosage_form": form,
        "observed_release_modifier": release,
        "observed_fdc_visible": fdc_markers,
        "observed_component_count": component_count,
        "observed_components": [],
    }


def compare_strength(observed: str, candidate_strengths: list[str]) -> tuple[str, str]:
    if not observed:
        return "NOT_COMPARABLE", "observed_strength_missing"
    normalized = [normalize_strength(s) for s in candidate_strengths if normalize_strength(s)]
    if not normalized:
        return "NOT_COMPARABLE", "candidate_strength_missing"
    return ("MATCH", "strength_equivalent") if normalize_strength(observed) in normalized else ("CONFLICT", "STRENGTH_CONFLICT")


def compare_dosage_form(observed: str, candidate: str) -> tuple[str, str]:
    obs = normalize_dosage_form(observed)
    cand = normalize_dosage_form(candidate)
    if not obs or not cand:
        return "NOT_COMPARABLE", "dosage_form_missing"
    return ("MATCH", "dosage_form_equivalent") if obs == cand else ("CONFLICT", "DOSAGE_FORM_CONFLICT")


def compare_component_count(observed_count: int | None, candidate_count: int | None) -> tuple[str, str]:
    if observed_count is None or candidate_count is None:
        return "NOT_COMPARABLE", "component_count_missing"
    return ("MATCH", "component_count_equal") if observed_count == candidate_count else ("CONFLICT", "COMPONENT_COUNT_CONFLICT")


class CandidateFactIndex:
    def __init__(self, root: Path) -> None:
        canonical = root / "knowledge/canonical"
        crosswalks = root / "knowledge/crosswalks"
        self.products = _read_csv(canonical / "brand_products.csv")
        self.families = _read_csv(canonical / "brand_families.csv")
        self.formulations = _read_csv(canonical / "clinical_formulations.csv")
        self.components = _read_csv(canonical / "formulation_components.csv")
        self.ingredients = _read_csv(canonical / "ingredients.csv")
        self.rxnorm = _read_csv(crosswalks / "rxnorm_ingredient_mappings.csv")
        self.atc = _read_csv(crosswalks / "rxclass_atc_mappings.csv")
        self.source_evidence = _read_csv(canonical / "source_evidence.csv", usecols=["evidence_id", "entity_id", "source_id", "kg_state", "authority"])
        self.product_by_id = self.products.set_index("brand_product_id").to_dict("index")
        self.family_by_id = self.families.set_index("brand_family_id").to_dict("index")
        self.formulation_by_id = self.formulations.set_index("formulation_id").to_dict("index")
        self.ingredient_by_id = self.ingredients.set_index("ingredient_id").to_dict("index")
        self.components_by_form = {k: v.to_dict("records") for k, v in self.components.groupby("formulation_id")}
        self.evidence_by_entity = {
            k: v.to_dict("records")
            for k, v in self.source_evidence.groupby("entity_id")
        }
        accepted_rx = self.rxnorm[self.rxnorm["mapping_status"].isin(["EXACT", "NORMALIZED_SUPPORTED"])]
        self.rx_by_ing = {k: v.to_dict("records") for k, v in accepted_rx.groupby("ingredient_id")}
        self.rx_by_rxcui = {k: v.to_dict("records") for k, v in accepted_rx.groupby("rxcui")}
        self.atc_by_ing = {k: v.to_dict("records") for k, v in self.atc.groupby("ingredient_id")}

    def _component_facts(self, formulation_id: str) -> list[dict[str, str]]:
        rows = []
        for comp in self.components_by_form.get(formulation_id, []):
            ing = self.ingredient_by_id.get(comp.get("ingredient_id", ""), {})
            rows.append(
                {
                    "ingredient_id": comp.get("ingredient_id", ""),
                    "ingredient_name": ing.get("canonical_name", ""),
                    "strength_text": comp.get("strength_text", ""),
                    "normalized_strength": comp.get("normalized_strength", ""),
                    "component_order": comp.get("component_order", ""),
                    "raw_component_text": comp.get("raw_component_text", ""),
                }
            )
        return rows

    def facts_for(self, entity_type: str, entity_id: str, candidate_name: str, source_state: str, evidence_ids: str) -> dict[str, Any]:
        facts: dict[str, Any] = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "candidate_name": candidate_name,
            "brand_family_id": "",
            "brand_family_name": "",
            "brand_product_id": "",
            "brand_product_name": "",
            "formulation_id": "",
            "dosage_form": "",
            "release_modifier": "",
            "components": [],
            "component_count": None,
            "fdc_status": "UNKNOWN",
            "company": "",
            "source_state": source_state,
            "source_evidence_ids": [v for v in str(evidence_ids).split("|") if v],
            "rxnorm": [],
            "atc": [],
        }
        if entity_type == "BRAND_PRODUCT":
            prod = self.product_by_id.get(entity_id, {})
            facts.update(
                {
                    "brand_product_id": entity_id,
                    "brand_product_name": prod.get("raw_brand_name", candidate_name),
                    "brand_family_id": prod.get("brand_family_id", ""),
                    "formulation_id": prod.get("formulation_id", ""),
                    "source_state": prod.get("kg_state", source_state),
                }
            )
            fam = self.family_by_id.get(facts["brand_family_id"], {})
            facts["brand_family_name"] = fam.get("canonical_name", "")
            form = self.formulation_by_id.get(facts["formulation_id"], {})
            facts["dosage_form"] = form.get("dosage_form", "")
            facts["release_modifier"] = form.get("release_modifier", "")
            facts["components"] = self._component_facts(facts["formulation_id"])
            facts["source_evidence_ids"] += [e["evidence_id"] for e in self.evidence_by_entity.get(entity_id, [])]
        elif entity_type == "BRAND_FAMILY":
            fam = self.family_by_id.get(entity_id, {})
            facts["brand_family_id"] = entity_id
            facts["brand_family_name"] = fam.get("canonical_name", candidate_name)
        elif entity_type == "CLINICAL_FORMULATION":
            form = self.formulation_by_id.get(entity_id, {})
            facts["formulation_id"] = entity_id
            facts["dosage_form"] = form.get("dosage_form", "")
            facts["release_modifier"] = form.get("release_modifier", "")
            facts["components"] = self._component_facts(entity_id)
        elif entity_type == "INGREDIENT":
            ing = self.ingredient_by_id.get(entity_id, {})
            facts["components"] = [{"ingredient_id": entity_id, "ingredient_name": ing.get("canonical_name", candidate_name), "strength_text": "", "normalized_strength": "", "component_order": "1", "raw_component_text": ing.get("canonical_name", candidate_name)}]
        elif entity_type == "RXNORM_CONCEPT":
            rows = self.rx_by_rxcui.get(entity_id, [])
            if rows:
                ing_id = rows[0].get("ingredient_id", "")
                ing = self.ingredient_by_id.get(ing_id, {})
                facts["components"] = [{"ingredient_id": ing_id, "ingredient_name": ing.get("canonical_name", rows[0].get("ingredient_name", candidate_name)), "strength_text": "", "normalized_strength": "", "component_order": "1", "raw_component_text": rows[0].get("rxnorm_name", candidate_name)}]
                facts["rxnorm"] = rows
        elif entity_type == "OFFICIAL_SOURCE_RECORD":
            facts["fdc_status"] = "EVIDENCE_ONLY"
        facts["component_count"] = len(facts["components"]) if facts["components"] else None
        if facts["component_count"] and facts["component_count"] > 1:
            facts["fdc_status"] = "FDC"
        elif facts["component_count"] == 1:
            facts["fdc_status"] = "SINGLE_COMPONENT"
        ingredient_ids = [c.get("ingredient_id", "") for c in facts["components"] if c.get("ingredient_id", "")]
        if not facts["rxnorm"]:
            for ing_id in ingredient_ids:
                facts["rxnorm"] += self.rx_by_ing.get(ing_id, [])
        for ing_id in ingredient_ids:
            facts["atc"] += self.atc_by_ing.get(ing_id, [])
        facts["source_evidence_ids"] = sorted(set(v for v in facts["source_evidence_ids"] if v))
        return facts


def _lexical_evidence(row: pd.Series, trace_group: pd.DataFrame, threshold: float) -> dict[str, Any]:
    surface = row["raw_medication_text"]
    candidate_name = row["candidate_name"]
    similarity = fuzz.WRatio(normalize_text(surface), normalize_text(candidate_name)) / 100.0
    exact = normalize_text(surface) == normalize_text(candidate_name)
    branch_ranks = json.loads(row["per_branch_rank_json"])
    status = "MATCH" if exact or similarity >= threshold or min(branch_ranks.values() or [999]) <= 3 else "NOT_COMPARABLE"
    return {
        "status": status,
        "observed_text": surface,
        "candidate_text": candidate_name,
        "exact_normalized_match": exact,
        "fuzzy_similarity": similarity,
        "branch_ranks": branch_ranks,
        "matched_fields": sorted(set(v for v in trace_group["matched_field"].astype(str) if v)),
        "matched_aliases": sorted(set(v for v in trace_group["matched_alias"].astype(str) if v))[:10],
    }


def _semantic_evidence(row: pd.Series, trace_group: pd.DataFrame) -> dict[str, Any]:
    r3 = trace_group[trace_group["branch"] == "R3_BIOMEDICAL_DENSE"]
    if r3.empty:
        return {"status": "NOT_COMPARABLE", "semantic_support": False}
    best = r3.sort_values("rank_numeric").iloc[0]
    return {
        "status": "MATCH",
        "semantic_support": True,
        "semantic_rank": int(best["rank_numeric"]),
        "semantic_score": float(best["score_numeric"]),
        "matched_semantic_document": best["matched_alias"],
        "underlying_canonical_entity": best["candidate_id"],
    }


def _formulation_evidence(observed: dict[str, Any], facts: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    missing: list[str] = []
    hard: list[str] = []
    candidate_strengths = [c.get("normalized_strength") or c.get("strength_text", "") for c in facts["components"]]
    strength_status, strength_reason = compare_strength(observed.get("observed_strength_normalized", ""), candidate_strengths)
    form_status, form_reason = compare_dosage_form(observed.get("observed_dosage_form", ""), facts.get("dosage_form", ""))
    count_status, count_reason = compare_component_count(observed.get("observed_component_count"), facts.get("component_count"))
    if strength_status == "CONFLICT":
        hard.append("STRENGTH_CONFLICT")
    elif strength_status == "NOT_COMPARABLE":
        missing.append(strength_reason)
    if form_status == "CONFLICT":
        hard.append("DOSAGE_FORM_CONFLICT")
    elif form_status == "NOT_COMPARABLE":
        missing.append(form_reason)
    if count_status == "CONFLICT":
        hard.append("COMPONENT_COUNT_CONFLICT")
    elif count_status == "NOT_COMPARABLE":
        missing.append(count_reason)
    release_status = "NOT_COMPARABLE"
    release_reason = "release_modifier_missing"
    if observed.get("observed_release_modifier") and facts.get("release_modifier"):
        release_status = "MATCH" if observed["observed_release_modifier"] == facts["release_modifier"] else "CONFLICT"
        release_reason = "release_modifier_equivalent" if release_status == "MATCH" else "RELEASE_FORM_CONFLICT"
        if release_status == "CONFLICT":
            hard.append("RELEASE_FORM_CONFLICT")
    else:
        missing.append(release_reason)
    fdc_status = "NOT_COMPARABLE"
    if observed.get("observed_fdc_visible") and facts.get("component_count"):
        fdc_status = "MATCH" if facts["component_count"] and facts["component_count"] > 1 else "CONFLICT"
        if fdc_status == "CONFLICT":
            hard.append("FDC_STRUCTURE_CONFLICT")
    return (
        {
            "ingredient_set": "NOT_COMPARABLE",
            "component_count": count_status,
            "strength": strength_status,
            "strength_reason": strength_reason,
            "dosage_form": form_status,
            "dosage_form_reason": form_reason,
            "release_modifier": release_status,
            "release_modifier_reason": release_reason,
            "fdc_structure": fdc_status,
            "candidate_components": facts["components"],
            "observed": observed,
        },
        sorted(set(hard)),
        sorted(set(missing)),
    )


def _provenance_evidence(facts: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    states = sorted(set(v for v in str(row["source_state"]).split("|") if v))
    source_ids = sorted(set(v for v in str(row["source_ids"]).split("|") if v))
    evidence_ids = sorted(set(facts["source_evidence_ids"] + [v for v in str(row["evidence_ids"]).split("|") if v]))
    if any("AUTHORITATIVE" in state for state in states) or row["entity_type"] == "RXNORM_CONCEPT":
        status = "MATCH"
    elif states:
        status = "NOT_COMPARABLE"
    else:
        status = "UNKNOWN"
    return {"status": status, "source_states": states, "source_ids": source_ids, "evidence_ids": evidence_ids, "field_specific_authority": True}


def _context_evidence(context_bundle_id: str) -> dict[str, Any]:
    if context_bundle_id:
        return {"implemented": False, "status": "NOT_COMPARABLE", "reason": "context_bundle_present_but_no_deterministic_drug_context_mapping_available", "context_bundle_id": context_bundle_id}
    return {"implemented": False, "status": "NOT_COMPARABLE", "reason": "context_bundle_missing_or_not_applicable", "context_bundle_id": ""}


def _resolution_level(entity_type: str) -> str:
    return {
        "BRAND_PRODUCT": "LOCAL_BRAND_PRODUCT",
        "BRAND_FAMILY": "LOCAL_BRAND_FAMILY",
        "CLINICAL_FORMULATION": "CLINICAL_FORMULATION",
        "INGREDIENT": "INGREDIENT",
        "RXNORM_CONCEPT": "TERMINOLOGY_CONCEPT",
        "OFFICIAL_SOURCE_RECORD": "OFFICIAL_EVIDENCE_ONLY",
    }.get(entity_type, "UNKNOWN")


def _load_config(root: Path) -> dict[str, Any]:
    path = root / "configs/evidence/stage5_evidence_config.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(EVIDENCE_CONFIG_DEFAULT, indent=2, sort_keys=True), encoding="utf-8")
        return EVIDENCE_CONFIG_DEFAULT
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(EVIDENCE_CONFIG_DEFAULT))
    merged["evidence"].update(data.get("evidence", {}))
    return merged


def run_stage5_evidence(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    config = _load_config(root)
    depth = int(config["evidence"]["depth"])
    threshold = float(config["evidence"]["strong_lexical_threshold"])
    out_dir = root / "derived/evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = _read_csv(root / "derived/ranking/ranked_candidates.csv")
    ranked = ranked[pd.to_numeric(ranked["final_rank"], errors="coerce") <= depth].copy()
    layer_a = _read_csv(root / "derived/layer_a_medication_mentions.csv")
    mention_meta = layer_a.set_index("mention_id").to_dict("index")
    trace = _read_csv(root / "derived/retrieval/stage2c1_branch_traces.csv")
    trace["rank_numeric"] = pd.to_numeric(trace["rank"], errors="coerce").fillna(999999)
    trace["score_numeric"] = pd.to_numeric(trace["score"], errors="coerce").fillna(0.0)
    trace_groups = {(m, c): g for (m, c), g in trace[trace["candidate_id"].astype(str) != ""].groupby(["mention_id", "candidate_id"], sort=False)}
    facts = CandidateFactIndex(root)

    rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    for row in ranked.itertuples(index=False):
        start = time.perf_counter()
        r = row._asdict()
        meta = mention_meta.get(r["mention_id"], {})
        observed = parse_observed_formulation({"raw_medication_text": r["raw_medication_text"], **meta})
        fact = facts.facts_for(r["entity_type"], r["entity_id"], r["candidate_name"], r["source_state"], r["evidence_ids"])
        tg = trace_groups.get((r["mention_id"], r["candidate_id"]), pd.DataFrame())
        lexical = _lexical_evidence(pd.Series(r), tg, threshold)
        semantic = _semantic_evidence(pd.Series(r), tg)
        formulation, hard, missing = _formulation_evidence(observed, fact)
        provenance = _provenance_evidence(fact, pd.Series(r))
        context = _context_evidence(meta.get("context_bundle_id", ""))
        supporting_ids = sorted(set(provenance["evidence_ids"] + [v for v in str(r["evidence_ids"]).split("|") if v]))
        summary_bits = [
            f"lexical={lexical['status']}",
            f"semantic={semantic['status']}",
            f"strength={formulation['strength']}",
            f"form={formulation['dosage_form']}",
            f"hard_conflicts={','.join(hard) if hard else 'none'}",
        ]
        rows.append(
            {
                "assessment_id": _stable_id("EASS", r["mention_id"], r["candidate_id"], r["final_rank"]),
                "mention_id": r["mention_id"],
                "raw_medication_text": r["raw_medication_text"],
                "candidate_id": r["candidate_id"],
                "candidate_type": r["candidate_type"],
                "entity_type": r["entity_type"],
                "entity_id": r["entity_id"],
                "resolution_level": _resolution_level(r["entity_type"]),
                "ranking_position": int(r["final_rank"]),
                "ranking_score": float(r["ranking_score"]),
                "lexical_status": lexical["status"],
                "semantic_status": semantic["status"],
                "strength_status": formulation["strength"],
                "dosage_form_status": formulation["dosage_form"],
                "fdc_status": formulation["fdc_structure"],
                "component_count_status": formulation["component_count"],
                "provenance_status": provenance["status"],
                "context_status": context["status"],
                "context_implemented": context["implemented"],
                "lexical_evidence_json": json.dumps(lexical, sort_keys=True),
                "semantic_evidence_json": json.dumps(semantic, sort_keys=True),
                "formulation_evidence_json": json.dumps(formulation, sort_keys=True),
                "provenance_evidence_json": json.dumps(provenance, sort_keys=True),
                "context_evidence_json": json.dumps(context, sort_keys=True),
                "candidate_facts_json": json.dumps(fact, sort_keys=True),
                "observed_formulation_json": json.dumps(observed, sort_keys=True),
                "hard_conflicts_json": json.dumps(hard, sort_keys=True),
                "supporting_evidence_ids_json": json.dumps(supporting_ids, sort_keys=True),
                "missing_evidence_json": json.dumps(missing, sort_keys=True),
                "evidence_summary": "; ".join(summary_bits),
            }
        )
        latencies.append((time.perf_counter() - start) * 1000)
    evidence = pd.DataFrame(rows)
    csv_path = out_dir / "evidence_assessments.csv"
    parquet_path = out_dir / "evidence_assessments.parquet"
    evidence.to_csv(csv_path, index=False)
    evidence.to_parquet(parquet_path, index=False)

    hard_count = int(evidence["hard_conflicts_json"].map(lambda s: len(json.loads(s)) > 0).sum()) if not evidence.empty else 0
    context_rate = float(evidence["context_implemented"].sum() / len(evidence)) if len(evidence) else 0.0
    source_complete = float(evidence["supporting_evidence_ids_json"].map(lambda s: len(json.loads(s)) > 0).sum() / len(evidence)) if len(evidence) else 0.0
    summary = {
        "generated_at": _now_iso(),
        "version": STAGE5_VERSION,
        "mentions_assessed": int(evidence["mention_id"].nunique()) if not evidence.empty else 0,
        "candidates_assessed": int(len(evidence)),
        "evidence_dimensions_operational": ["lexical", "semantic", "formulation", "provenance", "context"],
        "depth": depth,
        "status_counts": {
            "lexical": evidence["lexical_status"].value_counts().to_dict(),
            "strength": evidence["strength_status"].value_counts().to_dict(),
            "dosage_form": evidence["dosage_form_status"].value_counts().to_dict(),
            "fdc": evidence["fdc_status"].value_counts().to_dict(),
            "component_count": evidence["component_count_status"].value_counts().to_dict(),
        },
        "hard_conflict_count": hard_count,
        "context_evidence_operational_rate": context_rate,
        "source_evidence_completeness": source_complete,
        "median_latency_ms": float(median(latencies)) if latencies else 0.0,
        "retrieved_new_candidates": False,
        "gold_metrics_reported": False,
        "paths": {
            "evidence_assessments_csv": str(csv_path),
            "evidence_assessments_parquet": str(parquet_path),
            "summary": str(out_dir / "evidence_summary.json"),
            "report": str(root / "rebuild/reports/STAGE5_EVIDENCE_ASSESSMENT_REPORT.md"),
            "config": str(root / "configs/evidence/stage5_evidence_config.json"),
        },
        "READY_FOR_VERIFICATION": bool(len(evidence) > 0),
    }
    (out_dir / "evidence_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(root, summary)
    return summary


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 5 Evidence Assessment Report",
        "",
        f"- version: {STAGE5_VERSION}",
        f"- generated_at: {summary['generated_at']}",
        f"- mentions_assessed: {summary['mentions_assessed']}",
        f"- candidates_assessed: {summary['candidates_assessed']}",
        f"- evidence_depth: {summary['depth']}",
        f"- ready_for_verification: {str(summary['READY_FOR_VERIFICATION']).lower()}",
        "",
        "## Guardrails",
        "- Evidence Assessment consumed ranked candidates and Stage 2C.1 retrieval traces only.",
        "- No candidate retrieval, candidate generation, verification decision, or gold metric reporting is performed here.",
        "- Fuzzy/semantic support is retained as evidence but cannot override hard formulation conflicts.",
        "",
        "## Status Counts",
    ]
    for dimension, counts in summary["status_counts"].items():
        lines.append(f"- {dimension}: {counts}")
    lines.extend(
        [
            "",
            f"- hard_conflict_count: {summary['hard_conflict_count']}",
            f"- source_evidence_completeness: {summary['source_evidence_completeness']:.3f}",
            f"- context_evidence_operational_rate: {summary['context_evidence_operational_rate']:.3f}",
            "",
            "## Outputs",
        ]
    )
    for key, value in summary["paths"].items():
        lines.append(f"- {key}: {value}")
    (root / "rebuild/reports/STAGE5_EVIDENCE_ASSESSMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

