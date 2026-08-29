from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.schemas.deidentification import DeidentificationResult
from src.schemas.provenance import ArtifactProvenance
from src.utils.stable_ids import stable_hash, task_id


ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "derived"
MASK_DIR = DERIVED / "deid_reference_masks"
STAGE_DIR = DERIVED / "deid_stage2a"
REPORT_DIR = ROOT / "rebuild" / "reports"
REVIEW_DIR = ROOT / "review"
PILOT_DIR = ROOT / "generated" / "anonymized_stage2a_pilot"
HIST_RECON_DIR = ROOT / "generated" / "anonymized_stage2a_historical_recon"
STATE_DB = ROOT / "state" / "pipeline_state.sqlite"

TOOL_VERSION = "stage2a_local_deid_v0.1"
RUN_ID = "RUN_STAGE2A_LOCAL_DEID_V0"
RELIABLE_LINEAGE = {"VERIFIED_EXACT_METADATA", "VERIFIED_VISUAL_HIGH"}
STATUS_SUCCESS = "SUCCESS"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"
STATUS_FAILED = "FAILED"


@dataclass(frozen=True)
class MaskBox:
    x: int
    y: int
    w: int
    h: int

    def normalized(self, width: int, height: int) -> dict[str, float]:
        return {
            "x": self.x / width,
            "y": self.y / height,
            "w": self.w / width,
            "h": self.h / height,
        }


@dataclass(frozen=True)
class HistoricalMaskRecord:
    page_uid: str
    raw_path: str
    anon_path: str
    collection_date: str
    lineage_confidence: str
    width: int
    height: int
    orientation: str
    layout_signature: str
    layout_profile: str
    mask_path: str
    mask_area_fraction: float
    redaction_boxes_json: str
    mask_confidence_status: str
    unusual_reason: str
    split: str


@dataclass(frozen=True)
class LayoutProfile:
    layout_profile: str
    orientation: str
    width_bucket: int
    height_bucket: int
    layout_signature: str
    development_count: int
    validation_count: int
    common_layout_fraction: float
    median_mask_area_fraction: float
    normalized_boxes: list[dict[str, float]]
    reason_code: str


@dataclass(frozen=True)
class ValidationMetric:
    page_uid: str
    raw_path: str
    historical_anon_path: str
    reconstructed_anon_path: str
    layout_profile: str
    predicted_status: str
    qc_status: str
    reason_codes: str
    mask_recall: float
    mask_precision: float
    mask_iou: float
    false_negative_area_fraction: float
    false_positive_area_fraction: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    return str(path.relative_to(ROOT))


def ensure_dirs() -> None:
    for path in [MASK_DIR, STAGE_DIR, REPORT_DIR, REVIEW_DIR, PILOT_DIR, HIST_RECON_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"IMAGE_CORRUPT: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Failed to encode {path}")
    encoded.tofile(str(path))


def resize_to_raw(raw: np.ndarray, anon: np.ndarray) -> np.ndarray:
    if raw.shape[:2] == anon.shape[:2]:
        return anon
    return cv2.resize(anon, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)


def derive_historical_mask(raw: np.ndarray, anon: np.ndarray) -> tuple[np.ndarray, list[MaskBox], str]:
    anon = resize_to_raw(raw, anon)
    gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    gray_anon = cv2.cvtColor(anon, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_raw, gray_anon)

    dark_replacement = (gray_anon < 55) & (gray_raw > 70) & (diff > 30)
    strong_change = (diff > 80) & (gray_anon < 95)
    mask = (dark_replacement | strong_change).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = raw.shape[0] * raw.shape[1]
    min_area = max(120, int(image_area * 0.00003))
    kept = np.zeros(mask.shape, dtype=np.uint8)
    boxes: list[MaskBox] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w < 8 or h < 5:
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.18:
            continue
        boxes.append(MaskBox(int(x), int(y), int(w), int(h)))
        kept[labels == label] = 255
    kept = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, kernel, iterations=1)
    status = "ATTRIBUTABLE_TO_DEID" if boxes else "MASK_ALIGNMENT_UNCERTAIN"
    return kept, merge_boxes(boxes), status


def merge_boxes(boxes: list[MaskBox]) -> list[MaskBox]:
    if not boxes:
        return []
    rects = [[box.x, box.y, box.x + box.w, box.y + box.h] for box in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[list[int]] = []
        used = [False] * len(rects)
        for i, a in enumerate(rects):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = a
            used[i] = True
            for j, b in enumerate(rects):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = b
                horiz_gap = max(0, max(bx1 - ax2, ax1 - bx2))
                vert_gap = max(0, max(by1 - ay2, ay1 - by2))
                if horiz_gap <= 18 and vert_gap <= 14:
                    ax1, ay1, ax2, ay2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    changed = True
            merged.append([ax1, ay1, ax2, ay2])
        rects = merged
    return [MaskBox(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in rects]


def layout_signature(raw: np.ndarray) -> tuple[str, int, int, str]:
    h, w = raw.shape[:2]
    orientation = "portrait" if h >= w else "landscape"
    width_bucket = int(round(w / 100.0) * 100)
    height_bucket = int(round(h / 100.0) * 100)
    signature = f"{orientation}_{width_bucket}x{height_bucket}"
    return orientation, width_bucket, height_bucket, signature


def diagnostic_header_signature(raw: np.ndarray) -> str:
    h, w = raw.shape[:2]
    orientation = "portrait" if h >= w else "landscape"
    width_bucket = int(round(w / 100.0) * 100)
    height_bucket = int(round(h / 100.0) * 100)
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    top = gray[: max(1, int(h * 0.33)), :]
    edges = cv2.Canny(top, 60, 160)
    grid_h, grid_w = 3, 4
    bits: list[str] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            patch = edges[
                int(gy * top.shape[0] / grid_h) : int((gy + 1) * top.shape[0] / grid_h),
                int(gx * w / grid_w) : int((gx + 1) * w / grid_w),
            ]
            bits.append("1" if float(np.mean(patch > 0)) > 0.025 else "0")
    return f"{orientation}_{width_bucket}x{height_bucket}_{''.join(bits)}"


def assign_profile_name(signature: str) -> str:
    return "LAYOUT_" + stable_hash(signature, length=8).upper()


def page_uid_map() -> dict[str, str]:
    if not STATE_DB.exists():
        return {}
    conn = sqlite3.connect(STATE_DB)
    try:
        rows = conn.execute("SELECT page_uid, raw_image_relpath FROM pages").fetchall()
    finally:
        conn.close()
    return {raw_path: page_uid for page_uid, raw_path in rows if raw_path}


def reliable_lineage() -> pd.DataFrame:
    lineage = pd.read_csv(ROOT / "rebuild" / "manifests" / "raw_anon_verified_lineage.csv")
    return lineage[lineage["lineage_confidence"].isin(RELIABLE_LINEAGE)].copy()


def split_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["layout_profile"]].append(record)
    split_records_out: list[dict[str, Any]] = []
    for layout, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (row["collection_date"], row["raw_path"]))
        for idx, row in enumerate(rows):
            row = dict(row)
            row["split"] = "validation" if idx % 5 == 0 and len(rows) >= 5 else "development"
            split_records_out.append(row)
    return split_records_out


def build_historical_masks() -> tuple[list[HistoricalMaskRecord], dict[str, LayoutProfile]]:
    ensure_dirs()
    uid_map = page_uid_map()
    rows: list[dict[str, Any]] = []
    for idx, record in reliable_lineage().reset_index(drop=True).iterrows():
        raw_path = ROOT / record["raw_path"]
        anon_path = ROOT / record["anon_path"]
        try:
            raw = read_image(raw_path)
            anon = read_image(anon_path)
            mask, boxes, confidence_status = derive_historical_mask(raw, anon)
            orientation, width_bucket, height_bucket, signature = layout_signature(raw)
            profile = assign_profile_name(signature)
            page_uid = uid_map.get(record["raw_path"], "PAGE_DEID_" + stable_hash(record["raw_path"], record["anon_path"], length=20))
            mask_path = MASK_DIR / f"{page_uid}.png"
            write_image(mask_path, mask)
            h, w = raw.shape[:2]
            area_fraction = float(np.count_nonzero(mask)) / float(w * h)
            unusual_reason = "" if confidence_status == "ATTRIBUTABLE_TO_DEID" else "MASK_ALIGNMENT_UNCERTAIN"
            rows.append(
                {
                    "page_uid": page_uid,
                    "raw_path": record["raw_path"],
                    "anon_path": record["anon_path"],
                    "collection_date": record["collection_date"],
                    "lineage_confidence": record["lineage_confidence"],
                    "width": w,
                    "height": h,
                    "orientation": orientation,
                    "width_bucket": width_bucket,
                    "height_bucket": height_bucket,
                    "layout_signature": signature,
                    "layout_profile": profile,
                    "mask_path": relpath(mask_path),
                    "mask_area_fraction": area_fraction,
                    "redaction_boxes_json": json.dumps([box.normalized(w, h) for box in boxes], sort_keys=True),
                    "mask_confidence_status": confidence_status,
                    "unusual_reason": unusual_reason,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "page_uid": "PAGE_DEID_" + stable_hash(record["raw_path"], record["anon_path"], length=20),
                    "raw_path": record["raw_path"],
                    "anon_path": record["anon_path"],
                    "collection_date": record["collection_date"],
                    "lineage_confidence": record["lineage_confidence"],
                    "width": 0,
                    "height": 0,
                    "orientation": "",
                    "width_bucket": 0,
                    "height_bucket": 0,
                    "layout_signature": "",
                    "layout_profile": "LAYOUT_UNREADABLE",
                    "mask_path": "",
                    "mask_area_fraction": 0.0,
                    "redaction_boxes_json": "[]",
                    "mask_confidence_status": "FAILED",
                    "unusual_reason": str(exc),
                }
            )
    split_rows = split_records(rows)
    records = [
        HistoricalMaskRecord(
            page_uid=row["page_uid"],
            raw_path=row["raw_path"],
            anon_path=row["anon_path"],
            collection_date=row["collection_date"],
            lineage_confidence=row["lineage_confidence"],
            width=int(row["width"]),
            height=int(row["height"]),
            orientation=row["orientation"],
            layout_signature=row["layout_signature"],
            layout_profile=row["layout_profile"],
            mask_path=row["mask_path"],
            mask_area_fraction=float(row["mask_area_fraction"]),
            redaction_boxes_json=row["redaction_boxes_json"],
            mask_confidence_status=row["mask_confidence_status"],
            unusual_reason=row["unusual_reason"],
            split=row["split"],
        )
        for row in split_rows
    ]
    profiles = derive_layout_profiles(records)
    write_csv(STAGE_DIR / "historical_mask_audit.csv", [asdict(record) for record in records])
    (STAGE_DIR / "layout_profiles.json").write_text(
        json.dumps({key: asdict(value) for key, value in profiles.items()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return records, profiles


def derive_layout_profiles(records: list[HistoricalMaskRecord]) -> dict[str, LayoutProfile]:
    by_profile: dict[str, list[HistoricalMaskRecord]] = defaultdict(list)
    total = len(records)
    for record in records:
        if record.mask_confidence_status == "ATTRIBUTABLE_TO_DEID":
            by_profile[record.layout_profile].append(record)
    profiles: dict[str, LayoutProfile] = {}
    for profile, rows in by_profile.items():
        dev = [row for row in rows if row.split == "development"]
        val = [row for row in rows if row.split == "validation"]
        source_rows = dev or rows
        dims = Counter((row.orientation, row.width, row.height, row.layout_signature) for row in source_rows)
        orientation, width, height, signature = dims.most_common(1)[0][0]
        grouped_boxes: list[list[dict[str, float]]] = []
        for row in source_rows:
            grouped_boxes.append(json.loads(row.redaction_boxes_json))
        normalized_boxes = median_boxes(grouped_boxes)
        area_values = [row.mask_area_fraction for row in source_rows]
        profiles[profile] = LayoutProfile(
            layout_profile=profile,
            orientation=orientation,
            width_bucket=int(round(width / 100.0) * 100),
            height_bucket=int(round(height / 100.0) * 100),
            layout_signature=signature,
            development_count=len(dev),
            validation_count=len(val),
            common_layout_fraction=len(rows) / total if total else 0.0,
            median_mask_area_fraction=float(np.median(area_values)) if area_values else 0.0,
            normalized_boxes=normalized_boxes,
            reason_code="KNOWN_LAYOUT_MASK_APPLIED" if normalized_boxes else "UNUSUAL_LAYOUT",
        )
    return profiles


def median_boxes(box_sets: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for boxes in box_sets:
        candidates.extend(boxes)
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda box: (box["y"], box["x"]))
    clusters: list[list[dict[str, float]]] = []
    for box in candidates:
        assigned = False
        cx = box["x"] + box["w"] / 2.0
        cy = box["y"] + box["h"] / 2.0
        for cluster in clusters:
            centers = [(item["x"] + item["w"] / 2.0, item["y"] + item["h"] / 2.0) for item in cluster]
            mx = float(np.median([center[0] for center in centers]))
            my = float(np.median([center[1] for center in centers]))
            if abs(cx - mx) < 0.035 and abs(cy - my) < 0.035:
                cluster.append(box)
                assigned = True
                break
        if not assigned:
            clusters.append([box])
    min_support = max(1, int(math.ceil(len(box_sets) * 0.35)))
    medianed: list[dict[str, float]] = []
    for cluster in clusters:
        if len(cluster) < min_support:
            continue
        medianed.append(
            {
                "x": max(0.0, float(np.median([box["x"] for box in cluster]))),
                "y": max(0.0, float(np.median([box["y"] for box in cluster]))),
                "w": min(1.0, float(np.median([box["w"] for box in cluster]))),
                "h": min(1.0, float(np.median([box["h"] for box in cluster]))),
            }
        )
    return sorted(medianed, key=lambda box: (box["y"], box["x"]))


def predict_mask(raw: np.ndarray, profiles: dict[str, LayoutProfile]) -> tuple[np.ndarray, str, list[str]]:
    h, w = raw.shape[:2]
    _, _, _, signature = layout_signature(raw)
    profile_name = assign_profile_name(signature)
    profile = profiles.get(profile_name)
    mask = np.zeros((h, w), dtype=np.uint8)
    if not profile or not profile.normalized_boxes:
        return mask, "LAYOUT_UNASSIGNED", ["UNUSUAL_LAYOUT"]
    for box in profile.normalized_boxes:
        x1 = int(max(0, min(w - 1, round(box["x"] * w))))
        y1 = int(max(0, min(h - 1, round(box["y"] * h))))
        x2 = int(max(x1 + 1, min(w, round((box["x"] + box["w"]) * w))))
        y2 = int(max(y1 + 1, min(h, round((box["y"] + box["h"]) * h))))
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask, profile_name, ["KNOWN_LAYOUT_MASK_APPLIED"]


def apply_mask(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = raw.copy()
    out[mask > 0] = (0, 0, 0)
    return out


def compare_masks(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred = predicted > 0
    ref = reference > 0
    tp = float(np.logical_and(pred, ref).sum())
    fp = float(np.logical_and(pred, ~ref).sum())
    fn = float(np.logical_and(~pred, ref).sum())
    union = float(np.logical_or(pred, ref).sum())
    image_area = float(predicted.shape[0] * predicted.shape[1])
    return {
        "mask_recall": tp / (tp + fn) if tp + fn else 1.0,
        "mask_precision": tp / (tp + fp) if tp + fp else 1.0,
        "mask_iou": tp / union if union else 1.0,
        "false_negative_area_fraction": fn / image_area,
        "false_positive_area_fraction": fp / image_area,
    }


def validate_historical(records: list[HistoricalMaskRecord], profiles: dict[str, LayoutProfile]) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []
    for record in records:
        if record.split != "validation" or not record.mask_path:
            continue
        raw = read_image(ROOT / record.raw_path)
        reference = read_image(ROOT / record.mask_path)
        reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
        predicted, profile, reasons = predict_mask(raw, profiles)
        reconstructed = apply_mask(raw, predicted)
        out_path = HIST_RECON_DIR / f"{record.page_uid}.jpg"
        write_image(out_path, reconstructed)
        scores = compare_masks(predicted, reference_gray)
        status = STATUS_SUCCESS if "KNOWN_LAYOUT_MASK_APPLIED" in reasons else STATUS_NEEDS_REVIEW
        if scores["mask_recall"] < 0.72:
            status = STATUS_NEEDS_REVIEW
            reasons.append("LOW_VISUAL_SIMILARITY")
        metrics.append(
            ValidationMetric(
                page_uid=record.page_uid,
                raw_path=record.raw_path,
                historical_anon_path=record.anon_path,
                reconstructed_anon_path=relpath(out_path),
                layout_profile=profile,
                predicted_status=status,
                qc_status="NOT_CHECKED",
                reason_codes=";".join(sorted(set(reasons))),
                mask_recall=scores["mask_recall"],
                mask_precision=scores["mask_precision"],
                mask_iou=scores["mask_iou"],
                false_negative_area_fraction=scores["false_negative_area_fraction"],
                false_positive_area_fraction=scores["false_positive_area_fraction"],
            )
        )
    write_csv(STAGE_DIR / "historical_validation_metrics.csv", [asdict(metric) for metric in metrics])
    return metrics


def select_qc_sample(records: list[HistoricalMaskRecord], metrics: list[ValidationMetric], target: int = 60) -> list[dict[str, Any]]:
    metric_by_page = {metric.page_uid: metric for metric in metrics}
    grouped: dict[str, list[HistoricalMaskRecord]] = defaultdict(list)
    for record in records:
        grouped[record.layout_profile].append(record)
    sample: list[HistoricalMaskRecord] = []
    for _, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        rows = sorted(rows, key=lambda row: (row.split != "validation", -row.mask_area_fraction, row.raw_path))
        sample.extend(rows[: max(1, min(4, len(rows)))])
    remaining = [record for record in records if record not in set(sample)]
    remaining = sorted(remaining, key=lambda row: (row.collection_date, row.layout_profile, -row.mask_area_fraction, row.raw_path))
    sample.extend(remaining[: max(0, target - len(sample))])
    sample = sample[:target]
    rows_out: list[dict[str, Any]] = []
    for record in sample:
        metric = metric_by_page.get(record.page_uid)
        rows_out.append(
            {
                "page_uid": record.page_uid,
                "raw_path": record.raw_path,
                "historical_anon_path": record.anon_path,
                "reconstructed_anon_path": metric.reconstructed_anon_path if metric else "",
                "layout_profile": record.layout_profile,
                "predicted_status": metric.predicted_status if metric else "NOT_CHECKED",
                "mask_area_fraction": record.mask_area_fraction,
                "visual_lineage_confidence": record.lineage_confidence,
                "collection_date": record.collection_date,
                "unusual_pages": record.unusual_reason,
                "all_identifiers_masked": "",
                "medication_content_preserved": "",
                "over_redaction_problem": "",
                "under_redaction_problem": "",
                "overall_pass": "",
                "notes": "",
            }
        )
    write_csv(REVIEW_DIR / "deid_qc_sample.csv", rows_out)
    return rows_out


def pilot_candidates(profiles: dict[str, LayoutProfile], count: int = 25) -> pd.DataFrame:
    queue = pd.read_csv(ROOT / "rebuild" / "queues" / "deidentification_queue.csv")
    queue = queue.sort_values(["collection_date", "raw_path"]).copy()
    unique = queue.drop_duplicates("raw_sha256", keep="first").copy()
    selected: list[pd.Series] = []
    by_date = {date_value: rows for date_value, rows in unique.groupby("collection_date", sort=True)}
    while len(selected) < count and by_date:
        progressed = False
        for date_value in sorted(list(by_date.keys())):
            rows = by_date[date_value]
            if rows.empty:
                by_date.pop(date_value)
                continue
            selected.append(rows.iloc[0])
            by_date[date_value] = rows.iloc[1:]
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return pd.DataFrame(selected)


def run_pilot(profiles: dict[str, LayoutProfile], count: int = 25) -> tuple[list[DeidentificationResult], list[dict[str, Any]]]:
    selected = pilot_candidates(profiles, count=count)
    results: list[DeidentificationResult] = []
    provenance_rows: list[dict[str, Any]] = []
    duplicate_reuse_rows: list[dict[str, Any]] = []
    queue = pd.read_csv(ROOT / "rebuild" / "queues" / "deidentification_queue.csv")
    selected_shas = set(selected["raw_sha256"])
    duplicate_rows = queue[queue["raw_sha256"].isin(selected_shas)].copy()
    for _, row in selected.iterrows():
        raw_rel = row["raw_path"]
        raw_path = ROOT / raw_rel
        raw_hash_before = sha256_file(raw_path)
        task_stable = str(row["page_uid"])
        try:
            raw = read_image(raw_path)
            predicted, profile, reasons = predict_mask(raw, profiles)
            status = STATUS_SUCCESS if "KNOWN_LAYOUT_MASK_APPLIED" in reasons and np.count_nonzero(predicted) > 0 else STATUS_NEEDS_REVIEW
            if status == STATUS_NEEDS_REVIEW and "UNUSUAL_LAYOUT" not in reasons:
                reasons.append("UNUSUAL_LAYOUT")
            output_path = PILOT_DIR / row["collection_date"] / (Path(raw_rel).stem + "_stage2a.jpg")
            output = apply_mask(raw, predicted)
            write_image(output_path, output)
            output_sha = sha256_file(output_path)
            raw_hash_after = sha256_file(raw_path)
            if raw_hash_before != raw_hash_after:
                status = STATUS_FAILED
                reasons.append("SOURCE_IMAGE_MUTATED")
            artifact = ArtifactProvenance(
                artifact_id="ART_DEID_" + stable_hash(row["page_uid"], output_sha, length=20),
                artifact_type="stage2a_pilot_anonymized_image",
                parent_artifact_ids=[],
                source_paths=[raw_rel],
                source_sha256=[raw_hash_before],
                run_id=RUN_ID,
                pipeline_version=TOOL_VERSION,
                model_provider=None,
                model_id=None,
                model_settings={"local_only": True, "layout_profile": profile},
                human_review_status="PENDING" if status != STATUS_SUCCESS else "NOT_REQUIRED",
            )
            result = DeidentificationResult(
                page_uid=str(row["page_uid"]),
                output_path=relpath(output_path),
                output_sha256=output_sha,
                status=status,
                qc_status="NOT_CHECKED",
                redaction_metadata={
                    "layout_profile": profile,
                    "reason_codes": sorted(set(reasons)),
                    "mask_area_fraction": float(np.count_nonzero(predicted)) / float(predicted.shape[0] * predicted.shape[1]),
                    "raw_sha256_before": raw_hash_before,
                    "raw_sha256_after": raw_hash_after,
                },
                tool_version=TOOL_VERSION,
                run_id=RUN_ID,
                error_code=None if status != STATUS_FAILED else "SOURCE_IMAGE_MUTATED",
                provenance=artifact,
            )
            results.append(result)
            provenance_rows.append(flatten_result(result))
        except Exception as exc:
            result = DeidentificationResult(
                page_uid=str(row["page_uid"]),
                status=STATUS_FAILED,
                qc_status="NOT_CHECKED",
                redaction_metadata={"reason_codes": ["IMAGE_CORRUPT"], "raw_path": raw_rel},
                tool_version=TOOL_VERSION,
                run_id=RUN_ID,
                error_code=str(exc),
            )
            results.append(result)
            provenance_rows.append(flatten_result(result))

    for raw_sha, rows in duplicate_rows.groupby("raw_sha256"):
        if len(rows) <= 1:
            continue
        representative = rows.iloc[0]
        for _, dup in rows.iloc[1:].iterrows():
            duplicate_reuse_rows.append(
                {
                    "raw_sha256": raw_sha,
                    "duplicate_page_uid": dup["page_uid"],
                    "duplicate_raw_path": dup["raw_path"],
                    "derived_from_duplicate_representative": representative["page_uid"],
                    "representative_raw_path": representative["raw_path"],
                    "same_deidentification_configuration": True,
                    "reuse_allowed": True,
                }
            )
    write_csv(STAGE_DIR / "pilot_results.csv", provenance_rows)
    write_csv(STAGE_DIR / "duplicate_reuse_provenance.csv", duplicate_reuse_rows)
    write_state_records(results)
    return results, duplicate_reuse_rows


def flatten_result(result: DeidentificationResult) -> dict[str, Any]:
    provenance = result.provenance
    return {
        "page_uid": result.page_uid,
        "output_path": result.output_path or "",
        "output_sha256": result.output_sha256 or "",
        "status": result.status,
        "qc_status": result.qc_status,
        "redaction_metadata_json": json.dumps(result.redaction_metadata, sort_keys=True),
        "tool_version": result.tool_version or "",
        "run_id": result.run_id,
        "timestamp": result.timestamp.isoformat(),
        "error_code": result.error_code or "",
        "artifact_id": provenance.artifact_id if provenance else "",
        "source_paths_json": json.dumps(provenance.source_paths if provenance else []),
        "source_sha256_json": json.dumps(provenance.source_sha256 if provenance else []),
    }


def write_state_records(results: list[DeidentificationResult]) -> None:
    conn = sqlite3.connect(STATE_DB)
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute("DELETE FROM artifacts WHERE run_id = ?", (RUN_ID,))
        conn.execute("DELETE FROM tasks WHERE stage_name = 'stage2a_deidentification_pilot'")
        conn.execute(
            "INSERT OR REPLACE INTO stage_runs (run_id, stage_name, status, started_at, completed_at) VALUES (?, ?, ?, ?, ?)",
            (RUN_ID, "stage2a_deidentification_pilot", STATUS_SUCCESS, now, now),
        )
        for result in results:
            if result.provenance and result.output_path and result.output_sha256:
                conn.execute(
                    "INSERT OR REPLACE INTO artifacts (artifact_id, artifact_type, path, sha256, run_id, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        result.provenance.artifact_id,
                        result.provenance.artifact_type,
                        result.output_path,
                        result.output_sha256,
                        result.run_id,
                        result.status,
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text("", encoding="utf-8")


def summarize(records: list[HistoricalMaskRecord], profiles: dict[str, LayoutProfile], metrics: list[ValidationMetric], pilot: list[DeidentificationResult], duplicate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reliable = len(records)
    profile_counts = Counter(record.layout_profile for record in records)
    common_fraction = max(profile_counts.values()) / reliable if reliable else 0.0
    area_values = [record.mask_area_fraction for record in records if record.mask_confidence_status == "ATTRIBUTABLE_TO_DEID"]
    def median_metric(name: str) -> float:
        values = [getattr(metric, name) for metric in metrics]
        return float(np.median(values)) if values else 0.0
    pilot_status = Counter(result.status for result in pilot)
    return {
        "historical_pairs_used": reliable,
        "layout_clusters_discovered": len(profiles),
        "common_layout_fraction": common_fraction,
        "median_redaction_area_fraction": float(np.median(area_values)) if area_values else 0.0,
        "mask_uncertain_cases": sum(1 for record in records if record.mask_confidence_status != "ATTRIBUTABLE_TO_DEID"),
        "validation_images": len(metrics),
        "median_mask_recall": median_metric("mask_recall"),
        "median_mask_precision": median_metric("mask_precision"),
        "median_mask_iou": median_metric("mask_iou"),
        "median_false_negative_area_fraction": median_metric("false_negative_area_fraction"),
        "median_false_positive_area_fraction": median_metric("false_positive_area_fraction"),
        "pilot_count": len(pilot),
        "pilot_success": pilot_status.get(STATUS_SUCCESS, 0),
        "pilot_needs_review": pilot_status.get(STATUS_NEEDS_REVIEW, 0),
        "pilot_failed": pilot_status.get(STATUS_FAILED, 0),
        "duplicate_savings": len(duplicate_rows),
        "bulk_deid_ready": False,
    }


def write_reports(summary: dict[str, Any], records: list[HistoricalMaskRecord], profiles: dict[str, LayoutProfile], metrics: list[ValidationMetric], pilot: list[DeidentificationResult]) -> None:
    profile_counts = Counter(record.layout_profile for record in records)
    profile_lines = "\n".join(
        f"- `{profile}`: {count} historical reliable pairs, median mask area {profiles[profile].median_mask_area_fraction:.4f}, boxes {len(profiles[profile].normalized_boxes)}"
        for profile, count in profile_counts.most_common()
        if profile in profiles
    )
    unusual = [record for record in records if record.mask_confidence_status != "ATTRIBUTABLE_TO_DEID"]
    reasons = Counter()
    for result in pilot:
        reasons.update(result.redaction_metadata.get("reason_codes", []))
    pilot_reason_lines = "\n".join(f"- `{reason}`: {count}" for reason, count in reasons.most_common()) or "- none"
    REPORT_DIR.joinpath("HISTORICAL_DEID_MASK_AUDIT.md").write_text(
        f"""# Historical De-Identification Mask Audit

## Benchmark Set

Reliable historical pairs analyzed: {summary['historical_pairs_used']} (`VERIFIED_EXACT_METADATA` and `VERIFIED_VISUAL_HIGH` only). Ambiguous and medium-confidence pairs were excluded from ground truth.

## Mask Derivation

Masks were derived locally with OpenCV by comparing reliable raw/anonymized pairs, emphasizing dark/redacted changed regions and filtering small compression/noise components. Not every pixel difference was treated as PHI masking.

## Layout Clusters

Distinct layout clusters: {summary['layout_clusters_discovered']}

Common layout fraction: {summary['common_layout_fraction']:.3f}

{profile_lines}

## Redaction Geometry

Median redaction area fraction: {summary['median_redaction_area_fraction']:.5f}

Uncertain historical mask cases: {summary['mask_uncertain_cases']}

Typical redaction locations are concentrated in the upper/header bands for the dominant portrait/landscape prescription forms, with smaller secondary blocks where historical masks indicate identifiers.

## Unusual Cases

{len(unusual)} reliable pairs could not be confidently attributed to de-identification masking. These are retained in the audit CSV with `MASK_ALIGNMENT_UNCERTAIN` or `FAILED` reason codes and are not used to promote bulk readiness.
""",
        encoding="utf-8",
    )
    REPORT_DIR.joinpath("STAGE2A_DEIDENTIFICATION_VALIDATION_REPORT.md").write_text(
        f"""# Stage 2A De-Identification Validation Report

## Historical Validation

- Historical reliable pairs used: {summary['historical_pairs_used']}
- Layout clusters discovered: {summary['layout_clusters_discovered']}
- Validation images: {summary['validation_images']}
- Median mask recall: {summary['median_mask_recall']:.4f}
- Median mask precision: {summary['median_mask_precision']:.4f}
- Median mask IoU: {summary['median_mask_iou']:.4f}
- Median false-negative redaction area fraction: {summary['median_false_negative_area_fraction']:.6f}
- Median false-positive redaction area fraction: {summary['median_false_positive_area_fraction']:.6f}

Historical masks are behavior references, not perfect PHI ground truth. Manual QC is required before any bulk run.

## Manual QC

Manual QC sheet: `review/deid_qc_sample.csv`

Reviewer judgment columns were intentionally left blank.

## Pilot

- Pilot images processed: {summary['pilot_count']}
- SUCCESS: {summary['pilot_success']}
- NEEDS_REVIEW: {summary['pilot_needs_review']}
- FAILED: {summary['pilot_failed']}
- Duplicate reuse savings recorded: {summary['duplicate_savings']}

Pilot reason codes:

{pilot_reason_lines}

## Safety Concerns

- PHI safety: template masks are derived from historical behavior and may miss identifiers on unusual layouts or handwritten fields outside expected zones.
- Medication over-redaction: the local method masks only historical/profile regions, but manual QC must verify medication content remains readable near header boundaries.
- Cloud safety: no OpenAI, Gemini, Claude, external OCR, external VLM, or cloud document AI calls are used.

## Decision

`BULK_DEID_READY = false`

Recommended next action: complete manual QC on `review/deid_qc_sample.csv`, inspect all pilot `NEEDS_REVIEW` cases, then decide whether to tune layout profiles or authorize a second small pilot. Do not bulk-run the 501 queue yet.
""",
        encoding="utf-8",
    )


def run_stage2a() -> dict[str, Any]:
    records, profiles = build_historical_masks()
    metrics = validate_historical(records, profiles)
    select_qc_sample(records, metrics)
    pilot, duplicate_rows = run_pilot(profiles, count=25)
    summary = summarize(records, profiles, metrics, pilot, duplicate_rows)
    (STAGE_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_reports(summary, records, profiles, metrics, pilot)
    return summary
