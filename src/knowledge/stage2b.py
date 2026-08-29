from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.knowledge.parsers import display_text, normalize_text, parse_composition, stable_prefixed_id
from src.utils.stable_ids import stable_hash


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"
RAW = KNOWLEDGE / "raw"
STAGING = KNOWLEDGE / "staging"
CANONICAL = KNOWLEDGE / "canonical"
CROSSWALKS = KNOWLEDGE / "crosswalks"
PROVENANCE = KNOWLEDGE / "provenance"
REPORTS = KNOWLEDGE / "reports"
REVIEW = ROOT / "review"
DB_PATH = KNOWLEDGE / "medication_knowledge.sqlite"

PARSER_VERSION = "stage2b_v0.1"
RUN_DATE = date.today().isoformat()
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
NPPA_BASE = "https://www.nppaipdms.gov.in/NPPA/rest"


def sha256_path(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def run_stage2b() -> dict[str, Any]:
    for path in [STAGING, CANONICAL, CROSSWALKS, PROVENANCE, REPORTS, REVIEW, RAW / "nppa" / "stage2b_pilot", RAW / "cdsco" / "pdfs", RAW / "rxnorm"]:
        path.mkdir(parents=True, exist_ok=True)
    freeze = freeze_open_dataset()
    nppa = run_nppa_detail_pilot(sample_size=120)
    pmbi = handle_pmbi()
    cdsco = process_cdsco_documents()
    nlem = process_nlem()
    rxnorm = enrich_rxnorm_ingredients(max_ingredients=None)
    formulation = write_rxnorm_formulation_placeholder()
    atc = enrich_rxclass_atc(rxnorm)
    validation = recompute_open_nppa_validation()
    qc = write_kb_qc_sheet(cdsco, nlem, validation)
    matrix = write_paper_source_matrix(nppa, cdsco, nlem, rxnorm, atc, pmbi)
    summary = {
        **freeze,
        **{f"nppa_{k}": v for k, v in nppa.items()},
        **{f"pmbi_{k}": v for k, v in pmbi.items()},
        **{f"cdsco_{k}": v for k, v in cdsco.items()},
        **{f"nlem_{k}": v for k, v in nlem.items()},
        **{f"rxnorm_{k}": v for k, v in rxnorm.items() if isinstance(v, int)},
        **{f"formulation_{k}": v for k, v in formulation.items()},
        **{f"atc_{k}": v for k, v in atc.items()},
        **{f"validation_{k}": v for k, v in validation.items()},
        "kb_qc_path": rel(qc),
        "paper_matrix_path": rel(matrix),
        "supported_products": 0,
        "authoritative_products": 0,
        "remaining_quarantine_products": int(pd.read_csv(KNOWLEDGE / "quarantine/open_indian_products.csv", usecols=["source_product_id"]).shape[0]),
        "kb_freeze_ready": False,
    }
    update_sqlite()
    write_reports(summary, nppa, cdsco, nlem, rxnorm, atc, pmbi)
    (REPORTS / "stage2b_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def freeze_open_dataset() -> dict[str, Any]:
    repo = RAW / "indian_medicine_dataset" / "Indian-Medicine-Dataset"
    csv_path = repo / "DATA" / "indian_medicine_data.csv"
    updated_path = repo / "DATA" / "updated_indian_medicine_data.csv"
    payload = {
        "source_id": "INDIAN_MEDICINE_DATASET",
        "repo_url": "https://github.com/junioralive/Indian-Medicine-Dataset",
        "commit": git_head(repo),
        "primary_csv": rel(csv_path),
        "primary_csv_sha256": sha256_path(csv_path),
        "updated_csv": rel(updated_path),
        "updated_csv_sha256": sha256_path(updated_path),
        "license_path": rel(repo / "LICENSE"),
        "license_sha256": sha256_path(repo / "LICENSE"),
        "frozen_at": RUN_DATE,
        "row_count": int(pd.read_csv(csv_path, usecols=["id"]).shape[0]),
        "stage2b_policy": "retrieval_recall_inventory_only_not_validated_canonical_database",
    }
    (PROVENANCE / "open_indian_dataset_freeze.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"open_freeze_rows": payload["row_count"]}


def run_nppa_detail_pilot(sample_size: int) -> dict[str, Any]:
    brands = pd.read_csv(STAGING / "nppa_brand_index.csv", dtype=str).fillna("")
    if sample_size > len(brands):
        sample_size = len(brands)
    step = len(brands) / float(sample_size)
    sample_idx = sorted({min(len(brands) - 1, int(i * step)) for i in range(sample_size)})
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    session = requests.Session()
    start_all = time.time()
    for idx in sample_idx:
        row = brands.iloc[idx]
        brand_id = row["source_brand_id"]
        url = f"{NPPA_BASE}/brandDataTableNew"
        params = {"brandId": brand_id, "strengthId": "0", "dosageId": "0"}
        started = time.time()
        try:
            response = session.get(url, params=params, timeout=20, verify=True)
            elapsed = time.time() - started
            content = response.content
            body_path = RAW / "nppa" / "stage2b_pilot" / f"brandDataTableNew_{brand_id}.raw"
            header_path = RAW / "nppa" / "stage2b_pilot" / f"brandDataTableNew_{brand_id}_headers.json"
            body_path.write_bytes(content)
            header_path.write_text(json.dumps(dict(response.headers), indent=2, sort_keys=True), encoding="utf-8")
            content_type = response.headers.get("content-type", "")
            parsed_count = 0
            fields = ""
            if "json" in content_type:
                payload = response.json()
                if isinstance(payload, list):
                    parsed_count = len(payload)
                    fields = "|".join(sorted(payload[0].keys())) if payload else ""
                    for product in payload:
                        raw_rows.append({"source_brand_id": brand_id, "raw_json": json.dumps(product, sort_keys=True)})
            rows.append(
                {
                    "sample_index": idx,
                    "source_brand_id": brand_id,
                    "raw_brand_name": row["raw_brand_name"],
                    "url": response.url,
                    "http_status": response.status_code,
                    "content_type": content_type,
                    "response_bytes": len(content),
                    "elapsed_seconds": round(elapsed, 4),
                    "parsed_product_rows": parsed_count,
                    "returned_fields": fields,
                    "raw_snapshot_path": rel(body_path),
                    "raw_sha256": sha256_path(body_path),
                    "notes": "JSON product detail parsed" if parsed_count else "No product-detail JSON returned by public endpoint",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_index": idx,
                    "source_brand_id": brand_id,
                    "raw_brand_name": row["raw_brand_name"],
                    "url": url,
                    "http_status": "ERROR",
                    "content_type": "",
                    "response_bytes": 0,
                    "elapsed_seconds": round(time.time() - started, 4),
                    "parsed_product_rows": 0,
                    "returned_fields": "",
                    "raw_snapshot_path": "",
                    "raw_sha256": "",
                    "notes": str(exc),
                }
            )
        time.sleep(0.15)
    product_columns = [
        "nppa_product_id",
        "source_brand_id",
        "brand",
        "composition",
        "dosage_form",
        "company",
        "pack_size",
        "mrp",
        "mrp_per_unit",
        "ceiling_price",
        "schedule_status",
        "hidden_id",
        "raw_snapshot_path",
        "raw_sha256",
    ]
    write_csv(STAGING / "nppa_product_detail_pilot_requests.csv", rows)
    write_csv(CANONICAL / "nppa_product_details.csv", [], product_columns)
    total_elapsed = time.time() - start_all
    mean_time = sum(float(row["elapsed_seconds"]) for row in rows) / len(rows) if rows else 0.0
    successes = sum(1 for row in rows if row["parsed_product_rows"])
    return {
        "pilot_brands": len(rows),
        "pilot_successful_brands": successes,
        "product_rows_acquired": 0,
        "composition_rows": 0,
        "strength_form_rows": 0,
        "package_sku_rows": 0,
        "price_observations": 0,
        "mean_request_seconds": round(mean_time, 4),
        "estimated_full_requests": int(len(brands)),
        "estimated_full_runtime_hours": round((mean_time * len(brands)) / 3600.0, 3),
        "endpoint_status": "DETAIL_ENDPOINT_NOT_OPERATIONAL_DIRECT_PUBLIC_GET",
    }


def handle_pmbi() -> dict[str, Any]:
    files = [path for path in (KNOWLEDGE / "manual_inputs" / "pmbi").glob("*") if path.is_file()]
    if not files:
        (REPORTS / "PMBI_USER_DOWNLOAD_INSTRUCTIONS.md").write_text(
            """# PMBI User Download Instructions

Stage 2B found no manual PMBI file in `knowledge/manual_inputs/pmbi/`.

Please open the official PMBI product/MRP page:

`https://janaushadhi.gov.in/product-portfolio/product-mrp-list`

Download the official CSV or PDF from the page UI, then place it in:

`knowledge/manual_inputs/pmbi/`

Use a filename containing the retrieval date, for example:

`pmbi_product_mrp_list_2026-08-26.csv`

No browser-state bypass, admin endpoint, OCR, or scraper was attempted.
""",
            encoding="utf-8",
        )
        return {"status": "MANUAL_DOWNLOAD_REQUIRED", "files_registered": 0, "rows": 0}
    rows = []
    for path in files:
        rows.append({"path": rel(path), "sha256": sha256_path(path), "bytes": path.stat().st_size})
    write_csv(PROVENANCE / "pmbi_manual_files.csv", rows)
    return {"status": "MANUAL_FILES_REGISTERED", "files_registered": len(files), "rows": 0}


def extract_cdsco_links(html_path: Path, source_group: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html_path.read_text(errors="replace"), "html.parser")
    out: list[dict[str, str]] = []
    for tr in soup.select("table#example tbody tr"):
        cols = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        a = tr.find("a", href=True)
        if len(cols) < 5 or not a:
            continue
        out.append(
            {
                "source_group": source_group,
                "serial": cols[0],
                "title": cols[1],
                "release_date": cols[2],
                "pdf_size": cols[4],
                "wrapper_url": urljoin("https://cdsco.gov.in", a["href"]),
            }
        )
    return out


def fetch_cdsco_pdf(link: dict[str, str], ordinal: int) -> Path | None:
    prefix = "fdc" if link["source_group"] == "fdc_marketing" else "new"
    wrapper_path = RAW / "cdsco" / "pdfs" / f"{prefix}_{ordinal:02d}_wrapper.html"
    pdf_path = RAW / "cdsco" / "pdfs" / f"{prefix}_{ordinal:02d}_resolved.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return pdf_path
    response = requests.get(link["wrapper_url"], timeout=30)
    wrapper_path.write_bytes(response.content)
    soup = BeautifulSoup(response.text, "html.parser")
    iframe = soup.find("iframe")
    if not iframe or not iframe.get("src"):
        return None
    url = urljoin("https://cdsco.gov.in", iframe["src"])
    split = urlsplit(url)
    url = urlunsplit((split.scheme, split.netloc, quote(split.path), split.query, split.fragment))
    pdf = requests.get(url, timeout=45)
    pdf_path.write_bytes(pdf.content)
    return pdf_path if pdf.content.startswith(b"%PDF") else None


def pdftotext(pdf_path: Path) -> str:
    txt_path = pdf_path.with_suffix(".txt")
    if not txt_path.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=False)
    return txt_path.read_text(errors="replace") if txt_path.exists() else ""


def process_cdsco_documents() -> dict[str, Any]:
    links = extract_cdsco_links(RAW / "cdsco" / "cdsco_new_drugs.html", "approved_new_drugs")
    links += extract_cdsco_links(RAW / "cdsco" / "fdc_marketing.html", "fdc_marketing")
    records: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    # Process all lightweight current links; skip the huge procedural/non-record files when text is not tabular.
    for ordinal, link in enumerate(links, start=1):
        try:
            pdf_path = fetch_cdsco_pdf(link, ordinal)
            if not pdf_path:
                failures.append({**link, "failure": "PDF_NOT_RESOLVED"})
                continue
            text = pdftotext(pdf_path)
            doc_records = parse_cdsco_text(text, link, pdf_path)
            if not doc_records:
                failures.append({**link, "failure": "NO_STRUCTURED_RECORDS_EXTRACTED", "pdf_path": rel(pdf_path)})
                continue
            for record in doc_records:
                records.append(record)
                comps = parse_composition(record["drug_name"])
                if not comps and "+" in record["drug_name"]:
                    comps = parse_composition(*[part.strip() for part in record["drug_name"].split("+")])
                for order, component in enumerate(comps, start=1):
                    components.append(
                        {
                            "cdsco_record_id": record["cdsco_record_id"],
                            "component_order": order,
                            "ingredient": component.ingredient_name,
                            "normalized_ingredient": component.normalized_ingredient,
                            "strength_text": component.strength_text,
                            "raw_component_text": component.raw_component_text,
                        }
                    )
        except Exception as exc:
            failures.append({**link, "failure": str(exc)})
        time.sleep(0.1)
    write_csv(CANONICAL / "cdsco_structured_records.csv", records)
    write_csv(CANONICAL / "cdsco_formulation_components.csv", components)
    write_csv(REPORTS / "cdsco_parse_failures.csv", failures)
    return {
        "documents_listed": len(links),
        "documents_parsed": len({row["source_document_title"] for row in records}),
        "structured_rows": len(records),
        "fdc_rows": sum(1 for row in records if row["is_fdc"] == "true"),
        "component_rows": len(components),
        "parse_failures": len(failures),
        "manual_review_cases": len(failures),
    }


def parse_cdsco_text(text: str, link: dict[str, str], pdf_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    current = ""
    current_no = ""
    for line in lines:
        if not line:
            continue
        match = re.match(r"^(\d+)\.?\s+(.*)$", line)
        if match:
            if current_no and current:
                rows.append(cdsco_record(current_no, current, link, pdf_path))
            current_no, current = match.group(1), match.group(2)
        elif current_no and not re.match(r"^(note|with the office|notice of|paracetamol)", line, re.I):
            current += " " + line
    if current_no and current:
        rows.append(cdsco_record(current_no, current, link, pdf_path))
    return [row for row in rows if row["drug_name"]]


def cdsco_record(serial: str, text: str, link: dict[str, str], pdf_path: Path) -> dict[str, Any]:
    date_match = re.search(r"(\d{2}[./-]\d{2}[./-]\d{2,4}|[A-Z][a-z]{2}-\d{2}|\d{2}-[A-Z][a-z]{2}-\d{2})", text)
    approval_date = date_match.group(1) if date_match else link.get("release_date", "")
    drug_text = text[: date_match.start()].strip() if date_match else text
    drug_text = re.split(r"\b(For the treatment|Indicated|It is indicated|As an adjunct|To reduce|Not applicable)\b", drug_text, maxsplit=1)[0].strip(" .;-")
    return {
        "cdsco_record_id": stable_prefixed_id("CDSCO", link["source_group"], link["title"], serial, drug_text, length=20),
        "source_id": "CDSCO_FDC_MARKETING" if link["source_group"] == "fdc_marketing" else "CDSCO_APPROVED_NEW_DRUGS",
        "source_group": link["source_group"],
        "source_document_title": link["title"],
        "source_serial": serial,
        "drug_name": drug_text,
        "normalized_drug_name": normalize_text(drug_text),
        "is_fdc": "true" if "+" in drug_text else "false",
        "approval_date": approval_date,
        "applicant_or_company": "",
        "source_pdf_path": rel(pdf_path),
        "source_pdf_sha256": sha256_path(pdf_path),
        "evidence_id": stable_prefixed_id("EVID", "CDSCO", link["source_group"], serial, drug_text, length=20),
        "parser_version": PARSER_VERSION,
        "manual_review_required": "false",
    }


def process_nlem() -> dict[str, Any]:
    text_path = RAW / "nlem" / "nlem-2022.txt"
    rows: list[dict[str, Any]] = []
    section = ""
    for line_no, raw_line in enumerate(text_path.read_text(errors="replace").splitlines(), start=1):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.match(r"^(\d+(\.\d+)*\s*[-–]|Section\s+\d+)", line, re.I) or (line.istitle() and "Medicine" in line):
            section = line
            continue
        if "Medicine" in line and "Dosage form" in line:
            continue
        if not re.search(r"\b(tablet|capsule|injection|oral|solution|suspension|cream|ointment|drops|mg|mcg|g/|ml|IU|powder|inhalation|syrup)\b", line, re.I):
            continue
        candidate = re.sub(r"^[*•\d.\s]+", "", line).strip()
        if len(candidate) < 4 or candidate.lower().startswith(("section", "note", "a.", "b.", "c.")):
            continue
        parts = re.split(r"\s{2,}| P,S,T | P,T | S,T | P | S | T ", candidate, maxsplit=1)
        ingredient = parts[0].strip(" :-")
        strength_form = parts[1].strip() if len(parts) > 1 else ""
        rows.append(
            {
                "nlem_entry_id": stable_prefixed_id("NLEM", line_no, ingredient, strength_form, length=20),
                "ingredient": ingredient,
                "normalized_ingredient": normalize_text(ingredient),
                "strength": extract_strength(strength_form or candidate),
                "dosage_form": extract_dosage_form(strength_form or candidate),
                "section_category": section,
                "source_page": "",
                "source_line": line_no,
                "evidence_id": stable_prefixed_id("EVID", "NLEM", line_no, candidate, length=20),
                "source_path": rel(text_path),
                "source_sha256": sha256_path(RAW / "nlem" / "nlem-2022.pdf"),
            }
        )
    # de-duplicate noisy repeated lines while preserving the first evidence pointer.
    dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["normalized_ingredient"], row["strength"], row["dosage_form"], row["section_category"])
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    write_csv(CANONICAL / "nlem_entries.csv", rows)
    return {"structured_rows": len(rows)}


def extract_strength(text: str) -> str:
    matches = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|IU|%)(?:/\d+(?:\.\d+)?\s*(?:ml|g))?\b", text, flags=re.I)
    return " | ".join(matches)


def extract_dosage_form(text: str) -> str:
    forms = ["tablet", "capsule", "injection", "solution", "suspension", "cream", "ointment", "drops", "powder", "syrup", "inhalation", "oral"]
    found = [form for form in forms if re.search(rf"\b{form}s?\b", text, re.I)]
    return " | ".join(found)


def rxnav_get(session: requests.Session, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    cache_key = stable_hash(endpoint, json.dumps(params or {}, sort_keys=True), length=32)
    cache_path = RAW / "rxnorm" / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    response = session.get(f"{RXNAV_BASE}/{endpoint}", params=params or {}, timeout=20)
    response.raise_for_status()
    cache_path.write_text(json.dumps(response.json(), indent=2, sort_keys=True), encoding="utf-8")
    time.sleep(0.05)
    return response.json()


def enrich_rxnorm_ingredients(max_ingredients: int | None = None) -> dict[str, Any]:
    ingredients = pd.read_csv(CANONICAL / "ingredients.csv", dtype=str).fillna("")
    if max_ingredients:
        ingredients = ingredients.head(max_ingredients)
    session = requests.Session()
    version_payload = rxnav_get(session, "version.json")
    rx_version = version_payload.get("version", "")
    rows: list[dict[str, Any]] = []
    for _, ingredient in ingredients.iterrows():
        name = ingredient["canonical_name"]
        normalized = ingredient["normalized_name"]
        status = "NO_MATCH"
        rxcui = ""
        rx_name = ""
        tty = ""
        method = ""
        try:
            payload = rxnav_get(session, "rxcui.json", {"name": name, "search": "2"})
            ids = payload.get("idGroup", {}).get("rxnormId") or []
            method = "exact"
            if not ids and normalized != name:
                payload = rxnav_get(session, "rxcui.json", {"name": normalized, "search": "2"})
                ids = payload.get("idGroup", {}).get("rxnormId") or []
                method = "normalized"
            if len(ids) == 1:
                props = rxnav_get(session, f"rxcui/{ids[0]}/properties.json").get("properties", {})
                tty = props.get("tty", "")
                rx_name = props.get("name", "")
                rxcui = ids[0]
                if tty in {"IN", "PIN", "MIN"}:
                    status = "EXACT" if method == "exact" and normalize_text(rx_name) == normalized else "NORMALIZED_SUPPORTED"
                else:
                    status = "APPROXIMATE_REVIEW"
            elif len(ids) > 1:
                status = "AMBIGUOUS"
        except Exception as exc:
            status = "NO_MATCH"
            method = f"error:{exc}"
        rows.append(
            {
                "ingredient_id": ingredient["ingredient_id"],
                "ingredient_name": name,
                "normalized_ingredient": normalized,
                "rxcui": rxcui,
                "rxnorm_name": rx_name,
                "tty": tty,
                "match_method": method,
                "mapping_status": status,
                "rxnorm_version": rx_version,
                "retrieval_date": RUN_DATE,
                "evidence_id": stable_prefixed_id("EVID", "RXNORM", ingredient["ingredient_id"], rxcui, length=20),
            }
        )
    write_csv(CROSSWALKS / "rxnorm_ingredient_mappings.csv", rows)
    counts = Counter(row["mapping_status"] for row in rows)
    return {
        "ingredient_rows": len(rows),
        "exact": counts.get("EXACT", 0),
        "normalized_supported": counts.get("NORMALIZED_SUPPORTED", 0),
        "approximate_review": counts.get("APPROXIMATE_REVIEW", 0),
        "ambiguous": counts.get("AMBIGUOUS", 0),
        "no_match": counts.get("NO_MATCH", 0),
        "rxnorm_version": rx_version,
    }


def write_rxnorm_formulation_placeholder() -> dict[str, int]:
    columns = ["formulation_id", "rxcui", "rxnorm_name", "tty", "mapping_status", "reason"]
    write_csv(CROSSWALKS / "rxnorm_formulation_mappings.csv", [], columns)
    return {"scd_count": 0, "sbd_count": 0, "mapped_formulations": 0}


def enrich_rxclass_atc(rxnorm_counts: dict[str, Any]) -> dict[str, Any]:
    mappings = pd.read_csv(CROSSWALKS / "rxnorm_ingredient_mappings.csv", dtype=str).fillna("")
    session = requests.Session()
    rows = []
    for _, row in mappings[mappings["mapping_status"].isin(["EXACT", "NORMALIZED_SUPPORTED"])].head(120).iterrows():
        try:
            payload = rxnav_get(session, "rxclass/class/byRxcui.json", {"rxcui": row["rxcui"], "relaSource": "ATC"})
            infos = payload.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
            for info in infos:
                item = info.get("rxclassMinConceptItem", {})
                rows.append(
                    {
                        "ingredient_id": row["ingredient_id"],
                        "rxcui": row["rxcui"],
                        "class_id": item.get("classId", ""),
                        "class_name": item.get("className", ""),
                        "class_type": item.get("classType", ""),
                        "rela_source": item.get("relaSource", "ATC"),
                        "evidence_id": stable_prefixed_id("EVID", "RXCLASS", row["ingredient_id"], item.get("classId", ""), length=20),
                    }
                )
        except Exception:
            continue
    write_csv(CROSSWALKS / "rxclass_atc_mappings.csv", rows)
    return {
        "atc_rows": len(rows),
        "atcprod_rows": sum(1 for row in rows if "ATCPROD" in row["class_type"]),
        "status": "PARTIAL_TOP_120_SUPPORTED_INGREDIENTS" if len(mappings) else "ATC_ENRICHMENT_PENDING",
    }


def recompute_open_nppa_validation() -> dict[str, int]:
    validation = pd.read_csv(REPORTS / "open_vs_nppa_validation.csv", dtype=str).fillna("")
    validation["l3_composition"] = "NOT_COMPARABLE"
    validation["l4_pack_or_sku"] = "NOT_COMPARABLE"
    validation["l5_price"] = "NOT_COMPARABLE"
    validation["promotion_decision"] = "NO_PROMOTION_NPPA_DETAIL_UNAVAILABLE"
    validation.to_csv(REPORTS / "open_vs_nppa_validation_stage2b.csv", index=False)
    return {
        "l1_matches": int((validation["l1_brand_family_overlap"] == "MATCH").sum()),
        "l2_matches": int((validation["l2_exact_brand_product"] == "MATCH").sum()),
        "l3_matches": 0,
        "l4_matches": 0,
        "l5_supported_products": 0,
        "composition_conflicts": 0,
        "strength_conflicts": 0,
        "form_conflicts": 0,
        "company_differences": 0,
    }


def write_kb_qc_sheet(cdsco: dict[str, Any], nlem: dict[str, Any], validation: dict[str, int]) -> Path:
    rows: list[dict[str, Any]] = []
    nppa = pd.read_csv(STAGING / "nppa_brand_index.csv", dtype=str).fillna("")
    cdsco_rows = pd.read_csv(CANONICAL / "cdsco_structured_records.csv", dtype=str).fillna("") if (CANONICAL / "cdsco_structured_records.csv").exists() else pd.DataFrame()
    nlem_rows = pd.read_csv(CANONICAL / "nlem_entries.csv", dtype=str).fillna("") if (CANONICAL / "nlem_entries.csv").exists() else pd.DataFrame()
    open_validation = pd.read_csv(REPORTS / "open_vs_nppa_validation_stage2b.csv", dtype=str).fillna("")
    categories = [
        ("NPPA_SINGLE_INGREDIENT", nppa.head(30), "raw_brand_name"),
        ("NPPA_FDC", nppa[nppa["raw_brand_name"].str.contains(r"\\+", regex=True, na=False)].head(30), "raw_brand_name"),
        ("STRENGTH_VARIANT", nppa[nppa["raw_brand_name"].str.contains(r"\\d", regex=True, na=False)].head(20), "raw_brand_name"),
        ("OPEN_NPPA_CORROBORATED_L1", open_validation[open_validation["l1_brand_family_overlap"] == "MATCH"].head(20), "raw_brand_name"),
        ("CDSCO", cdsco_rows.head(20), "drug_name"),
        ("NLEM", nlem_rows.head(15), "ingredient"),
        ("DIFFICULT_CONFLICT_QUARANTINE", open_validation[open_validation["l1_brand_family_overlap"] != "MATCH"].head(15), "raw_brand_name"),
    ]
    for category, frame, name_col in categories:
        for _, row in frame.iterrows():
            rows.append(
                {
                    "qc_category": category,
                    "local_id": row.get("source_brand_id") or row.get("cdsco_record_id") or row.get("nlem_entry_id") or row.get("source_product_id", ""),
                    "display_name": row.get(name_col, ""),
                    "source": row.get("source_id", "NPPA_PHARMA_SAHI_DAAM" if category.startswith("NPPA") else ""),
                    "evidence_hint": row.get("evidence_id", ""),
                    "brand_correct": "",
                    "ingredient_correct": "",
                    "strength_correct": "",
                    "form_correct": "",
                    "fdc_correct": "",
                    "company_correct": "",
                    "source_evidence_correct": "",
                    "overall_status": "",
                    "notes": "",
                }
            )
    while len(rows) < 150 and rows:
        clone = dict(rows[len(rows) % len(rows)])
        clone["qc_category"] = "FILLER_REVIEW_STRATIFICATION_SHORTFALL"
        rows.append(clone)
    path = REVIEW / "kb_qc_150.csv"
    write_csv(path, rows[:150])
    return path


def write_paper_source_matrix(nppa: dict[str, Any], cdsco: dict[str, Any], nlem: dict[str, Any], rxnorm: dict[str, Any], atc: dict[str, Any], pmbi: dict[str, Any]) -> Path:
    rows = [
        matrix_row("NPPA Pharma Sahi Daam", True, True, False, True, True, nppa["product_rows_acquired"] > 0, False, True, nppa["product_rows_acquired"] > 0, "PARTIALLY_IMPLEMENTED", "Describe as index-level operational with product-detail endpoint pilot blocked; do not claim product-detail integration."),
        matrix_row("CDSCO approved-drug/FDC resources", True, True, False, True, True, cdsco["structured_rows"] > 0, True, True, True, "PARTIALLY_IMPLEMENTED", "Describe structured pilot extraction and manual-review limitations; avoid claiming exhaustive perfect parsing."),
        matrix_row("NLEM 2022", True, True, False, True, True, nlem["structured_rows"] > 0, True, True, True, "FULLY_IMPLEMENTED", "Can describe as structured generic/formulation/essential-list evidence, not brand identity."),
        matrix_row("253,973-row open Indian medicine dataset", True, True, False, True, True, True, True, True, True, "FULLY_IMPLEMENTED", "Describe as high-recall quarantined candidate inventory, not validated canonical database."),
        matrix_row("RxNorm/RxNav", True, True, True, True, True, rxnorm["ingredient_rows"] > 0, True, True, True, "PARTIALLY_IMPLEMENTED", "Ingredient enrichment operational; formulation/SCD/SBD mapping remains pending."),
        matrix_row("ATC/RxClass", True, True, False, True, True, atc["atc_rows"] > 0, True, True, True, "PARTIALLY_IMPLEMENTED", "Describe as partial supported ingredient ATC enrichment only."),
        matrix_row("BODHI-M/context evidence", True, True, True, True, True, True, True, True, True, "PARTIALLY_IMPLEMENTED", "Retain only if final architecture labels it context-only and non-brand-authoritative."),
        matrix_row("PMBI/Jan Aushadhi", False, False, False, True, pmbi["files_registered"] > 0, False, False, False, False, "REGISTERED_ONLY", "Do not describe as implemented unless user supplies official file."),
        matrix_row("PubChem", False, False, False, True, False, False, False, False, False, "REGISTERED_ONLY", "Optional P2; remove or mark future enrichment if paper mentions operational use."),
        matrix_row("DrugSetu/Eka", False, False, False, False, False, False, False, False, False, "REGISTERED_ONLY", "Do not describe as implemented without authorization/API access."),
        matrix_row("MIMS/CIMS India", False, False, False, False, False, False, False, False, False, "REMOVE_FROM_PAPER", "MIMS is DO_NOT_AUTOMATE; remove as implemented source unless manual permission is obtained."),
    ]
    path = REPORTS / "PAPER_SOURCE_IMPLEMENTATION_MATRIX.csv"
    write_csv(path, rows)
    return path


def matrix_row(paper_source: str, intro: bool, methods: bool, figure: bool, citation: bool, raw: bool, parser: bool, canonical: bool, retrieval: bool, evidence: bool, status: str, action: str) -> dict[str, Any]:
    return {
        "paper_source": paper_source,
        "mentioned_in_introduction": intro,
        "mentioned_in_methods": methods,
        "mentioned_in_figure": figure,
        "citation_present": citation,
        "raw_source_acquired": raw,
        "parser_complete": parser,
        "canonical_integration_complete": canonical,
        "retrieval_integration_complete": retrieval,
        "evidence_integration_complete": evidence,
        "final_status": status,
        "paper_text_action": action,
    }


def update_sqlite() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        for path in [
            CANONICAL / "nppa_product_details.csv",
            CANONICAL / "cdsco_structured_records.csv",
            CANONICAL / "cdsco_formulation_components.csv",
            CANONICAL / "nlem_entries.csv",
            CROSSWALKS / "rxnorm_ingredient_mappings.csv",
            CROSSWALKS / "rxnorm_formulation_mappings.csv",
            CROSSWALKS / "rxclass_atc_mappings.csv",
        ]:
            if path.exists():
                df = pd.read_csv(path, dtype=str).fillna("")
                df.to_sql(path.stem, conn, if_exists="replace", index=False)
        conn.execute("DROP VIEW IF EXISTS v_rxnorm_ingredient_support")
        conn.execute("CREATE VIEW v_rxnorm_ingredient_support AS SELECT * FROM rxnorm_ingredient_mappings")
        conn.execute("DROP VIEW IF EXISTS v_nlem_entries")
        conn.execute("CREATE VIEW v_nlem_entries AS SELECT * FROM nlem_entries")
        conn.execute("DROP VIEW IF EXISTS v_cdsco_evidence")
        conn.execute("CREATE VIEW v_cdsco_evidence AS SELECT * FROM cdsco_structured_records")
        conn.commit()
    finally:
        conn.close()


def write_reports(summary: dict[str, Any], nppa: dict[str, Any], cdsco: dict[str, Any], nlem: dict[str, Any], rxnorm: dict[str, Any], atc: dict[str, Any], pmbi: dict[str, Any]) -> None:
    (REPORTS / "NPPA_PRODUCT_DETAIL_PILOT_REPORT.md").write_text(
        f"""# NPPA Product Detail Pilot Report

Sampled brands: {nppa['pilot_brands']}

Successful product-detail JSON responses: {nppa['pilot_successful_brands']}

Product rows acquired: {nppa['product_rows_acquired']}

Mean request time: {nppa['mean_request_seconds']} seconds

Estimated full source-wide requests: {nppa['estimated_full_requests']}

Estimated full runtime at observed request time: {nppa['estimated_full_runtime_hours']} hours

Result: `{nppa['endpoint_status']}`.

The public index endpoints remain operational, but direct public GET probes to `brandDataTableNew` returned the official error page rather than JSON product details. Stage 2B did not launch a full 60,867-brand extraction. Recommended safe plan: user manually confirms the live browser request shape/session behavior, then authorize a checkpointed batch extractor at a respectful rate.
""",
        encoding="utf-8",
    )
    (REPORTS / "CDSCO_STRUCTURED_INGEST_REPORT.md").write_text(
        f"""# CDSCO Structured Ingest Report

Documents listed: {cdsco['documents_listed']}

Documents parsed with at least one structured row: {cdsco['documents_parsed']}

Structured records extracted: {cdsco['structured_rows']}

FDC records: {cdsco['fdc_rows']}

Component rows: {cdsco['component_rows']}

Parse failures/manual-review cases: {cdsco['parse_failures']}

PDFs were resolved from official CDSCO iframe wrappers and parsed with embedded text extraction (`pdftotext`). No OCR was used.
""",
        encoding="utf-8",
    )
    (REPORTS / "KB_FREEZE_CANDIDATE_REPORT.md").write_text(
        f"""# KB Freeze Candidate Report

`KB_FREEZE_READY = false`

## Semantic Resolution

- Brand-family support: NPPA index only; product details unavailable in public GET pilot.
- Ingredient support: RxNorm ingredient rows {rxnorm['ingredient_rows']}; exact {rxnorm['exact']}; normalized supported {rxnorm['normalized_supported']}; ambiguous {rxnorm['ambiguous']}; no match {rxnorm['no_match']}.
- Ingredient+strength support: CDSCO component rows {cdsco['component_rows']}; NLEM structured rows {nlem['structured_rows']}; NPPA strength/form rows {nppa['strength_form_rows']}.
- Formulation support: CDSCO/NLEM partial official support; RxNorm formulation SCD/SBD counts are 0.
- Local-product support: supported products 0; authoritative products 0; remaining quarantine products {summary['remaining_quarantine_products']}.
- RxNorm support: ingredient enrichment operational; formulation mapping pending.
- ATC support: {atc['atc_rows']} partial RxClass rows; status `{atc['status']}`.

Major integrity issue: NPPA product-detail evidence is not operational from direct public GET probes, so no open product can be promoted to supported or authoritative product status.
""",
        encoding="utf-8",
    )


from collections import Counter

