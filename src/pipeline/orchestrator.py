from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from src.annotation.agent import image_to_base64, call_annotation_model
from src.deidentification.stage2a import read_image, write_image
from src.evidence.stage5 import (
    CandidateFactIndex,
    _context_evidence,
    _formulation_evidence,
    _lexical_evidence,
    _provenance_evidence,
    _resolution_level,
    parse_observed_formulation,
)
from src.ranking.stage4 import DEFAULT_CONFIG as RANKING_DEFAULT_CONFIG
from src.ranking.stage4 import _rank_mention
from src.retrieval.stage2c import BRANCHES, TOP_N, R1ExactFuzzy, R2BM25, R3BiomedicalDense, R4RxNorm, RetrievalHit, _trace_rows_for_mention
from src.retrieval.stage2c1 import IdentityMapper, StructuredIndiaRetriever, _canonicalize_trace, _union
from src.utils.stable_ids import stable_hash
from src.verification.stage6 import _adequate_for_accept, _j, _layer_b_row, _review_reasons, _resolution
from src.verification.stage6_1 import _nppa_supported_families


STAGE7_VERSION = "stage7_operational_orchestrator_v0.1"


class PipelineStatus(StrEnum):
    SUCCESS = "SUCCESS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class PipelineState(StrEnum):
    RAW_IMAGE = "RAW_IMAGE"
    DEIDENTIFIED = "DEIDENTIFIED"
    ANNOTATED = "ANNOTATED"
    LAYER_A = "LAYER_A"
    RETRIEVED = "RETRIEVED"
    RANKED = "RANKED"
    EVIDENCE_ASSESSED = "EVIDENCE_ASSESSED"
    VERIFIED = "VERIFIED"
    LAYER_B = "LAYER_B"


@dataclass
class PipelineResult:
    status: PipelineStatus
    document_uid: str
    output_dir: Path
    layer_b_records: list[dict[str, Any]]
    provenance: dict[str, Any]
    error_code: str = ""
    error_message: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _document_uid(*parts: object) -> str:
    return "DOC_ONLINE_" + stable_hash(*parts, length=20)


def _mention_id(document_uid: str, text: str, index: int) -> str:
    return "MENT_ONLINE_" + stable_hash(document_uid, text, index, length=20)


def _page_uid(document_uid: str) -> str:
    return "PAGE_ONLINE_" + stable_hash(document_uid, length=20)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_layer_a_json(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        document_uid = _document_uid(path, sha256_file(path))
        rows = data
    elif "mentions" in data:
        document_uid = data.get("document_uid") or _document_uid(path, sha256_file(path))
        rows = data["mentions"]
    elif "raw_entities" in data:
        document_uid = data.get("document_uid") or _document_uid(path, sha256_file(path))
        meds = data.get("raw_entities", {}).get("medications", [])
        rows = meds
    else:
        document_uid = data.get("document_uid") or _document_uid(path, sha256_file(path))
        rows = [data] if data.get("raw_medication_text") else []
    mentions = []
    for i, row in enumerate(rows):
        raw = row.get("raw_medication_text") or row.get("medicine") or row.get("medicine_name") or ""
        if not raw:
            continue
        mentions.append(
            {
                "mention_id": row.get("mention_id") or _mention_id(document_uid, raw, i),
                "document_uid": row.get("document_uid") or document_uid,
                "page_uid": row.get("page_uid") or _page_uid(document_uid),
                "raw_medication_text": raw,
                "raw_strength_text": row.get("raw_strength_text", ""),
                "raw_dosage_text": row.get("raw_dosage_text", ""),
                "context_bundle_id": row.get("context_bundle_id", ""),
            }
        )
    return document_uid, mentions


class PipelineOrchestrator:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.cache_dir = self.root / "knowledge/cache/stage2c"
        self.outputs_dir = self.root / "outputs"
        self.state_db = self.root / "state/pipeline_orchestrator.sqlite"
        self.r1: R1ExactFuzzy | None = None
        self.r2: R2BM25 | None = None
        self.r3: R3BiomedicalDense | None = None
        self.r4: R4RxNorm | None = None
        self.r5: StructuredIndiaRetriever | None = None
        self.identity: IdentityMapper | None = None
        self.fact_index: CandidateFactIndex | None = None
        self.nppa_families: set[str] | None = None
        self._ensure_state_db()

    def _ensure_state_db(self) -> None:
        self.state_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.state_db)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS processed_files (sha256 TEXT PRIMARY KEY, document_uid TEXT NOT NULL, output_dir TEXT NOT NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            conn.commit()
        finally:
            conn.close()

    def _load_retrieval(self) -> None:
        if self.r1 is not None:
            return
        self.r1 = R1ExactFuzzy(self.root)
        self.r2 = R2BM25(self.root, self.cache_dir)
        self.r3 = R3BiomedicalDense(self.root, self.cache_dir)
        self.r4 = R4RxNorm(self.root, self.cache_dir)
        self.r5 = StructuredIndiaRetriever(self.root)
        self.identity = IdentityMapper(self.root)

    def _facts(self) -> CandidateFactIndex:
        if self.fact_index is None:
            self.fact_index = CandidateFactIndex(self.root)
        return self.fact_index

    def _nppa(self) -> set[str]:
        if self.nppa_families is None:
            self.nppa_families = _nppa_supported_families(self.root)
        return self.nppa_families

    def deidentify_image(self, raw_image: Path, output_dir: Path) -> dict[str, Any]:
        output = output_dir / f"{raw_image.stem}_deidentified{raw_image.suffix or '.png'}"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            image = read_image(raw_image)
            height = image.shape[0]
            redaction_height = max(1, int(height * 0.20))
            image[:redaction_height, :] = 0
            write_image(output, image)
        except Exception as exc:
            return {
                "status": "FAILED",
                "error_code": "DEIDENTIFICATION_FAILED",
                "error_message": f"{type(exc).__name__}: {exc}",
                "source_sha256": sha256_file(raw_image),
            }
        return {
            "status": "SUCCESS",
            "deidentified_image_path": str(output),
            "method": "stage7_local_header_redaction",
            "source_sha256": sha256_file(raw_image),
            "raw_image_copied": False,
        }

    def annotate_image(self, deidentified_image: Path, output_dir: Path) -> dict[str, Any]:
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANNOTATION_API_KEY")):
            return {"status": "BLOCKED", "error_code": "CREDENTIALS_MISSING", "error_message": "No OPENAI_API_KEY or ANNOTATION_API_KEY environment variable set"}
        result = call_annotation_model(image_to_base64(deidentified_image))
        if result.annotation is None:
            return {"status": result.status, "error_code": result.error_code or "ANNOTATION_FAILED", "error_message": result.error_message or ""}
        payload = result.annotation.model_dump()
        _write_json(output_dir / "annotation.json", payload)
        return {"status": result.status, "annotation": payload}

    def retrieve_candidates(self, mention: dict[str, Any]) -> pd.DataFrame:
        self._load_retrieval()
        surface = mention["raw_medication_text"]
        results: dict[str, tuple[str, list[RetrievalHit], float, str]] = {}
        branch_calls = {
            "R1_EXACT_FUZZY": self.r1.search,
            "R2_BM25": self.r2.search,
            "R3_BIOMEDICAL_DENSE": self.r3.search,
            "R4_RXNORM": self.r4.search,
        }
        for branch, fn in branch_calls.items():
            start = time.perf_counter()
            try:
                hits = fn(surface, TOP_N)
                results[branch] = ("SUCCESS" if hits else "EMPTY", hits, (time.perf_counter() - start) * 1000, "")
            except Exception as exc:
                results[branch] = ("FAILED", [], (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}")
        start = time.perf_counter()
        try:
            r5_hits, latency = self.r5.search(surface, TOP_N)
            results["R5_INDIA_KB"] = (
                "SUCCESS" if r5_hits else "EMPTY",
                [
                    RetrievalHit(
                        candidate_id=hit.get("candidate_id", ""),
                        candidate_name=hit.get("candidate_name", ""),
                        candidate_type=hit.get("candidate_type", ""),
                        score=float(hit.get("score", 0.0)),
                        score_semantics=hit.get("score_semantics", ""),
                        matched_field=hit.get("matched_field", ""),
                        matched_alias=hit.get("matched_alias", ""),
                        source_state=hit.get("source_state", ""),
                        authority=hit.get("authority", ""),
                        provenance_evidence_ids=hit.get("provenance_evidence_ids", ""),
                        metadata={},
                    )
                    for hit in r5_hits
                ],
                latency,
                "",
            )
        except Exception as exc:
            results["R5_INDIA_KB"] = ("FAILED", [], (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}")
        raw = pd.DataFrame(_trace_rows_for_mention(pd.Series(mention), results))
        harmonized, _ = _canonicalize_trace(raw, self.identity)
        return harmonized

    def rank_candidates(self, trace: pd.DataFrame) -> pd.DataFrame:
        candidate_trace = trace[trace["candidate_id"].astype(str) != ""].copy()
        if candidate_trace.empty:
            return pd.DataFrame()
        candidate_trace["rank"] = pd.to_numeric(candidate_trace["rank"], errors="coerce")
        candidate_trace["score"] = pd.to_numeric(candidate_trace["score"], errors="coerce")
        rows = _rank_mention(candidate_trace, RANKING_DEFAULT_CONFIG)
        ranked = pd.DataFrame(rows)
        if ranked.empty:
            return ranked
        ranked["mention_id"] = candidate_trace["mention_id"].iloc[0]
        ranked["raw_medication_text"] = candidate_trace["raw_medication_text"].iloc[0]
        for column in ["participating_branches", "per_branch_rank", "per_branch_score", "ranking_components"]:
            ranked[f"{column}_json"] = ranked[column].map(lambda value: json.dumps(value, sort_keys=True))
            ranked = ranked.drop(columns=[column])
        return ranked

    def assess_evidence(self, ranked: pd.DataFrame, trace: pd.DataFrame, mention: dict[str, Any]) -> pd.DataFrame:
        if ranked.empty:
            return pd.DataFrame()
        trace = trace.copy()
        trace["rank_numeric"] = pd.to_numeric(trace["rank"], errors="coerce").fillna(999999)
        trace["score_numeric"] = pd.to_numeric(trace["score"], errors="coerce").fillna(0.0)
        trace_groups = {(m, c): g for (m, c), g in trace[trace["candidate_id"].astype(str) != ""].groupby(["mention_id", "candidate_id"], sort=False)}
        rows = []
        for row in ranked.itertuples(index=False):
            r = row._asdict()
            observed = parse_observed_formulation({**mention, "raw_medication_text": r["raw_medication_text"]})
            fact = self._facts().facts_for(r["entity_type"], r["entity_id"], r["candidate_name"], r["source_state"], r["evidence_ids"])
            tg = trace_groups.get((r["mention_id"], r["candidate_id"]), pd.DataFrame())
            lexical = _lexical_evidence(pd.Series(r), tg, 0.9)
            semantic = pd.Series({"status": "NOT_COMPARABLE", "semantic_support": False}).to_dict()
            if not tg.empty and "R3_BIOMEDICAL_DENSE" in set(tg["branch"]):
                from src.evidence.stage5 import _semantic_evidence

                semantic = _semantic_evidence(pd.Series(r), tg)
            formulation, hard, missing = _formulation_evidence(observed, fact)
            provenance = _provenance_evidence(fact, pd.Series(r))
            context = _context_evidence(mention.get("context_bundle_id", ""))
            rows.append(
                {
                    "assessment_id": "EASS_ONLINE_" + stable_hash(r["mention_id"], r["candidate_id"], r["final_rank"], length=20),
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
                    "supporting_evidence_ids_json": json.dumps(sorted(set(provenance["evidence_ids"] + [v for v in str(r["evidence_ids"]).split("|") if v])), sort_keys=True),
                    "missing_evidence_json": json.dumps(missing, sort_keys=True),
                    "evidence_summary": f"lexical={lexical['status']}; semantic={semantic['status']}; strength={formulation['strength']}; form={formulation['dosage_form']}; hard_conflicts={','.join(hard) if hard else 'none'}",
                }
            )
        return pd.DataFrame(rows)

    def verify(self, evidence: pd.DataFrame, mention: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if evidence.empty:
            layer = _layer_b_row(None, "NIL", mention, ["NO_DEFENSIBLE_CANDIDATE"], [], [])
            return pd.DataFrame([{"mention_id": mention["mention_id"], "verification_decision": "NIL", "selected_candidate_id": "", "resolution_level": "NO_SUPPORTED_RESOLUTION", "decision_reason_codes_json": json.dumps(["NO_DEFENSIBLE_CANDIDATE"]), "hard_conflicts_json": "[]", "missing_evidence_json": "[]", "evidence_ids_json": "[]", "verification_method": "stage7_online", "verification_version": STAGE7_VERSION, "timestamp": _now_iso()}]), [layer]
        group = evidence.sort_values("ranking_position", kind="stable")
        selected = None
        decision = "NIL"
        reason_codes: list[str] = []
        for _, row in group.iterrows():
            ok, _ = _adequate_for_accept(row)
            if ok:
                selected = row
                decision = "ACCEPT"
                break
            facts = _j(row["candidate_facts_json"], {})
            family_id = row["entity_id"] if row["entity_type"] == "BRAND_FAMILY" else facts.get("brand_family_id", "")
            if family_id and family_id in self._nppa() and row["lexical_status"] == "MATCH" and not _j(row["hard_conflicts_json"], []):
                selected = row.copy()
                selected["entity_type"] = "BRAND_FAMILY"
                selected["entity_id"] = family_id
                selected["candidate_id"] = f"ENTITY:BRAND_FAMILY:{family_id}"
                selected["supporting_evidence_ids_json"] = json.dumps(sorted(set(_j(row["supporting_evidence_ids_json"], []) + ["NPPA_BRAND_INDEX_L1_MATCH"])))
                decision = "ACCEPT"
                reason_codes = []
                break
            if selected is None and row["entity_type"] != "OFFICIAL_SOURCE_RECORD":
                selected = row
                decision = "HUMAN_REVIEW"
                reason_codes = _review_reasons(row)
        if selected is None:
            return self.verify(pd.DataFrame(), mention)
        hard = _j(selected["hard_conflicts_json"], [])
        missing = _j(selected["missing_evidence_json"], [])
        layer = _layer_b_row(selected, decision, mention, reason_codes, hard, missing)
        if decision == "ACCEPT" and selected["entity_type"] == "BRAND_FAMILY":
            facts = _j(selected["candidate_facts_json"], {})
            layer["resolution_level"] = "LOCAL_BRAND_FAMILY"
            layer["primary_candidate_id"] = selected["candidate_id"]
            layer["local_brand_family_id"] = selected["entity_id"]
            layer["local_brand_family_name"] = facts.get("brand_family_name", "") or facts.get("candidate_name", "")
            layer["local_brand_product_id"] = ""
            layer["local_brand_product_name"] = ""
            layer["review_reason_codes_json"] = "[]"
            layer["pipeline_version"] = STAGE7_VERSION
            layer["provenance_json"] = json.dumps(
                {
                    "selected_from_ranked_candidate": True,
                    "stage6_1_online_brand_family_fallback": True,
                    "nppa_brand_family_supported": True,
                    "verification_generated_at": _now_iso(),
                },
                sort_keys=True,
            )
        verification = {
            "mention_id": mention["mention_id"],
            "verification_decision": decision,
            "selected_candidate_id": "" if decision == "NIL" else layer["primary_candidate_id"],
            "resolution_level": layer["resolution_level"],
            "decision_reason_codes_json": json.dumps(reason_codes, sort_keys=True),
            "hard_conflicts_json": json.dumps(hard, sort_keys=True),
            "missing_evidence_json": json.dumps(missing, sort_keys=True),
            "evidence_ids_json": selected["supporting_evidence_ids_json"],
            "verification_method": "stage7_online_deterministic_rules",
            "verification_version": STAGE7_VERSION,
            "timestamp": _now_iso(),
        }
        return pd.DataFrame([verification]), [layer]

    def process_mentions(self, mentions: list[dict[str, Any]], document_uid: str | None = None, mode: str = "mention_text") -> PipelineResult:
        document_uid = document_uid or _document_uid(mode, json.dumps(mentions, sort_keys=True))
        output_dir = self.outputs_dir / document_uid
        output_dir.mkdir(parents=True, exist_ok=True)
        layer_a_payload = {"document_uid": document_uid, "mentions": mentions}
        _write_json(output_dir / "layer_a.json", layer_a_payload)
        all_trace, all_ranked, all_evidence, all_verification, all_layer_b = [], [], [], [], []
        latencies = {"retrieval": [], "ranking": [], "evidence": [], "verification": []}
        for i, mention in enumerate(mentions):
            mention = dict(mention)
            mention.setdefault("document_uid", document_uid)
            mention.setdefault("page_uid", _page_uid(document_uid))
            mention.setdefault("mention_id", _mention_id(document_uid, mention["raw_medication_text"], i))
            t = time.perf_counter()
            trace = self.retrieve_candidates(mention)
            latencies["retrieval"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            ranked = self.rank_candidates(trace)
            latencies["ranking"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            evidence = self.assess_evidence(ranked, trace, mention)
            latencies["evidence"].append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            verification, layer_b = self.verify(evidence, mention)
            latencies["verification"].append((time.perf_counter() - t) * 1000)
            all_trace.append(trace)
            all_ranked.append(ranked)
            all_evidence.append(evidence)
            all_verification.append(verification)
            all_layer_b.extend(layer_b)
        trace_df = pd.concat(all_trace, ignore_index=True) if all_trace else pd.DataFrame()
        ranked_df = pd.concat(all_ranked, ignore_index=True) if all_ranked else pd.DataFrame()
        evidence_df = pd.concat(all_evidence, ignore_index=True) if all_evidence else pd.DataFrame()
        verification_df = pd.concat(all_verification, ignore_index=True) if all_verification else pd.DataFrame()
        layer_b = all_layer_b
        trace_df.to_csv(output_dir / "retrieval_trace.csv", index=False)
        ranked_df.to_csv(output_dir / "ranked_candidates.csv", index=False)
        evidence_df.to_csv(output_dir / "evidence_trace.csv", index=False)
        verification_df.to_csv(output_dir / "verification_results.csv", index=False)
        _write_json(output_dir / "layer_b_normalized.json", {"document_uid": document_uid, "medications": layer_b})
        status = PipelineStatus.NEEDS_REVIEW if not verification_df.empty and (verification_df["verification_decision"] == "HUMAN_REVIEW").any() else PipelineStatus.SUCCESS
        provenance = {"pipeline_version": STAGE7_VERSION, "mode": mode, "states": [state.value for state in [PipelineState.LAYER_A, PipelineState.RETRIEVED, PipelineState.RANKED, PipelineState.EVIDENCE_ASSESSED, PipelineState.VERIFIED, PipelineState.LAYER_B]], "status": status.value, "latency_medians_ms": {k: float(median(v)) if v else 0 for k, v in latencies.items()}, "generated_at": _now_iso()}
        _write_json(output_dir / "pipeline_provenance.json", provenance)
        return PipelineResult(status, document_uid, output_dir, layer_b, provenance)

    def process_mention_text(self, text: str) -> PipelineResult:
        document_uid = _document_uid("mention_text", text)
        return self.process_mentions([{"raw_medication_text": text}], document_uid=document_uid, mode="mention_text")

    def process_layer_a_json(self, path: Path) -> PipelineResult:
        document_uid, mentions = _read_layer_a_json(path)
        return self.process_mentions(mentions, document_uid=document_uid, mode="layer_a_json")

    def process_deidentified_image(self, path: Path) -> PipelineResult:
        document_uid = _document_uid("deidentified_image", sha256_file(path))
        output_dir = self.outputs_dir / document_uid
        output_dir.mkdir(parents=True, exist_ok=True)
        annotation = self.annotate_image(path, output_dir)
        if annotation["status"] == "BLOCKED":
            provenance = {"pipeline_version": STAGE7_VERSION, "mode": "deidentified_image", "states": [PipelineState.DEIDENTIFIED.value], "blocked_at": PipelineState.ANNOTATED.value, "error_code": annotation["error_code"], "generated_at": _now_iso()}
            _write_json(output_dir / "pipeline_provenance.json", provenance)
            return PipelineResult(PipelineStatus.BLOCKED, document_uid, output_dir, [], provenance, annotation["error_code"], annotation["error_message"])
        _write_json(output_dir / "annotation.json", annotation.get("annotation", {}))
        _, mentions = _read_layer_a_json(output_dir / "annotation.json")
        return self.process_mentions(mentions, document_uid=document_uid, mode="deidentified_image")

    def process_raw_image(self, path: Path) -> PipelineResult:
        document_uid = _document_uid("raw_image", sha256_file(path))
        output_dir = self.outputs_dir / document_uid
        deid = self.deidentify_image(path, output_dir)
        _write_json(output_dir / "deidentification_result.json", deid)
        if deid["status"] != "SUCCESS":
            provenance = {
                "pipeline_version": STAGE7_VERSION,
                "mode": "raw_image",
                "states": [PipelineState.RAW_IMAGE.value],
                "failed_at": PipelineState.DEIDENTIFIED.value,
                "error_code": deid["error_code"],
                "generated_at": _now_iso(),
            }
            _write_json(output_dir / "pipeline_provenance.json", provenance)
            return PipelineResult(PipelineStatus.FAILED, document_uid, output_dir, [], provenance, deid["error_code"], deid["error_message"])
        annotation = self.annotate_image(Path(deid["deidentified_image_path"]), output_dir)
        if annotation["status"] == "BLOCKED":
            provenance = {
                "pipeline_version": STAGE7_VERSION,
                "mode": "raw_image",
                "states": [PipelineState.RAW_IMAGE.value, PipelineState.DEIDENTIFIED.value],
                "blocked_at": PipelineState.ANNOTATED.value,
                "error_code": annotation["error_code"],
                "generated_at": _now_iso(),
            }
            _write_json(output_dir / "pipeline_provenance.json", provenance)
            return PipelineResult(PipelineStatus.BLOCKED, document_uid, output_dir, [], provenance, annotation["error_code"], annotation["error_message"])
        _write_json(output_dir / "annotation.json", annotation.get("annotation", {}))
        _, mentions = _read_layer_a_json(output_dir / "annotation.json")
        return self.process_mentions(mentions, document_uid=document_uid, mode="raw_image")

    def register_processed_file(self, source_path: Path, result: PipelineResult) -> None:
        conn = sqlite3.connect(self.state_db)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO processed_files (sha256, document_uid, output_dir, status, updated_at) VALUES (?, ?, ?, ?, ?)",
                (sha256_file(source_path), result.document_uid, str(result.output_dir), result.status.value, _now_iso()),
            )
            conn.commit()
        finally:
            conn.close()

    def lookup_processed_file(self, source_path: Path) -> dict[str, str] | None:
        digest = sha256_file(source_path)
        conn = sqlite3.connect(self.state_db)
        try:
            row = conn.execute("SELECT sha256, document_uid, output_dir, status, updated_at FROM processed_files WHERE sha256=?", (digest,)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {"sha256": row[0], "document_uid": row[1], "output_dir": row[2], "status": row[3], "updated_at": row[4]}


def process_mention_text(text: str, root: Path | None = None) -> PipelineResult:
    return PipelineOrchestrator(root).process_mention_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process a prescription or Layer-A mention through semantic normalization.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw-image", type=Path)
    mode.add_argument("--deidentified-image", type=Path)
    mode.add_argument("--layer-a-json", type=Path)
    mode.add_argument("--mention-text")
    args = parser.parse_args(argv)
    orch = PipelineOrchestrator()
    if args.raw_image:
        result = orch.process_raw_image(args.raw_image)
    elif args.deidentified_image:
        result = orch.process_deidentified_image(args.deidentified_image)
    elif args.layer_a_json:
        result = orch.process_layer_a_json(args.layer_a_json)
    else:
        result = orch.process_mention_text(args.mention_text)
    print(json.dumps({"status": result.status.value, "document_uid": result.document_uid, "output_dir": str(result.output_dir), "error_code": result.error_code, "error_message": result.error_message, "layer_b_records": result.layer_b_records}, indent=2, sort_keys=True))
    return 0 if result.status in {PipelineStatus.SUCCESS, PipelineStatus.NEEDS_REVIEW, PipelineStatus.BLOCKED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
