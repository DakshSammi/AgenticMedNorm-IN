from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.knowledge.models import (
    Alias,
    AliasEvidenceLink,
    Authority,
    BrandFamily,
    BrandProduct,
    ClinicalFormulation,
    CompanyEntity,
    CompanyRelationship,
    FormulationComponent,
    Ingredient,
    KGState,
    PackageSKU,
    PriceObservation,
    SourceEvidence,
    SourceTier,
    row_dict,
)
from src.knowledge.parsers import (
    PARSER_VERSION,
    display_text,
    extract_brand_family,
    formulation_signature,
    infer_dosage_form,
    normalize_text,
    parse_composition,
    stable_prefixed_id,
)


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE = ROOT / "knowledge"
RAW = KNOWLEDGE / "raw"
STAGING = KNOWLEDGE / "staging"
CANONICAL = KNOWLEDGE / "canonical"
QUARANTINE = KNOWLEDGE / "quarantine"
CROSSWALKS = KNOWLEDGE / "crosswalks"
PROVENANCE = KNOWLEDGE / "provenance"
REPORTS = KNOWLEDGE / "reports"
CONTEXT = KNOWLEDGE / "context"
MANUAL_PMBI = KNOWLEDGE / "manual_inputs" / "pmbi"
DB_PATH = KNOWLEDGE / "medication_knowledge.sqlite"

OPEN_CSV = RAW / "indian_medicine_dataset" / "Indian-Medicine-Dataset" / "DATA" / "indian_medicine_data.csv"
OPEN_UPDATED_CSV = RAW / "indian_medicine_dataset" / "Indian-Medicine-Dataset" / "DATA" / "updated_indian_medicine_data.csv"
OPEN_LICENSE = RAW / "indian_medicine_dataset" / "Indian-Medicine-Dataset" / "LICENSE"
OPEN_README = RAW / "indian_medicine_dataset" / "Indian-Medicine-Dataset" / "README.md"
NPPA_BRANDS = RAW / "nppa" / "brandComboNew.json"
NPPA_FORMULATIONS = RAW / "nppa" / "formulationListNew.json"
NPPA_SCHEDULED = RAW / "nppa" / "scheduledFormulationComboNew.json"
PMBI_PROBE = RAW / "pmbi" / "getAllProductForWeb_probe.json"
CDSCO_NEW = RAW / "cdsco" / "cdsco_new_drugs.html"
CDSCO_FDC = RAW / "cdsco" / "fdc_marketing.html"
NLEM_PAGE = RAW / "nlem" / "nppa_nlem2022.html"
NLEM_PDF = RAW / "nlem" / "nlem-2022.pdf"
NLEM_TEXT = RAW / "nlem" / "nlem-2022.txt"
BODHI_TRIPLES = RAW / "bodhi_m" / "ekacare_BODHI-M" / "data" / "triples.jsonl"
BODHI_README = RAW / "bodhi_m" / "ekacare_BODHI-M" / "README.md"

SOURCE_REGISTRY_COLUMNS = [
    "source_id",
    "source_name",
    "source_tier",
    "source_type",
    "authority_scope",
    "official_or_secondary",
    "url",
    "access_method",
    "requires_auth",
    "license",
    "license_verified",
    "bulk_download_allowed",
    "automated_query_allowed",
    "redistribution_allowed",
    "ai_use_restriction",
    "retrieval_date",
    "source_version",
    "raw_snapshot_path",
    "raw_sha256",
    "parser_version",
    "notes",
]


def main() -> None:
    ensure_dirs()
    counts: dict[str, Any] = {}
    raw_hashes = collect_raw_hashes()
    write_build_inputs_manifest(raw_hashes)
    write_source_registry(raw_hashes)

    nppa = parse_nppa_indexes(raw_hashes)
    counts.update({f"nppa_{key}": value for key, value in nppa["counts"].items()})

    cdsco_counts = parse_cdsco_indexes(raw_hashes)
    counts.update({f"cdsco_{key}": value for key, value in cdsco_counts.items()})

    nlem_counts = parse_nlem(raw_hashes)
    counts.update({f"nlem_{key}": value for key, value in nlem_counts.items()})

    bodhi_counts = parse_bodhi_context()
    counts.update({f"bodhi_{key}": value for key, value in bodhi_counts.items()})

    open_counts = build_from_open_dataset(raw_hashes, set(nppa["brand_family_norms"]), set(nppa["brand_name_norms"]))
    counts.update(open_counts)

    write_duckdb_if_available()
    write_access_reports(raw_hashes, counts)
    write_stage_report(raw_hashes, counts)
    print(json.dumps(counts, indent=2, sort_keys=True))


def ensure_dirs() -> None:
    for path in [
        RAW / "nppa",
        RAW / "pmbi",
        RAW / "cdsco",
        RAW / "nlem",
        RAW / "indian_medicine_dataset",
        RAW / "bodhi_m",
        RAW / "pubchem",
        RAW / "rxnorm",
        RAW / "drugsetu",
        STAGING,
        CANONICAL,
        QUARANTINE,
        CROSSWALKS,
        PROVENANCE,
        REPORTS,
        KNOWLEDGE / "cache",
        MANUAL_PMBI,
        CONTEXT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_path(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_raw_hashes() -> dict[str, str]:
    raw_hashes: dict[str, str] = {}
    for path in sorted(RAW.rglob("*")):
        if path.is_file() and ".git/" not in str(path):
            raw_hashes[rel(path)] = sha256_path(path)
    return raw_hashes


def write_build_inputs_manifest(raw_hashes: dict[str, str]) -> None:
    rows = [
        {"input_path": path, "sha256": sha, "used_for_stage": "stage1b_knowledge_kb"}
        for path, sha in sorted(raw_hashes.items())
        if path.startswith("knowledge/raw/")
    ]
    pd.DataFrame(rows).to_csv(PROVENANCE / "build_inputs_manifest.csv", index=False)


def raw_sha(raw_hashes: dict[str, str], path: Path) -> str:
    return raw_hashes.get(rel(path), "")


def write_source_registry(raw_hashes: dict[str, str]) -> None:
    today = date.today().isoformat()
    rows = [
        source_row(
            "INDIAN_MEDICINE_DATASET",
            "junioralive/Indian-Medicine-Dataset",
            SourceTier.C_OPEN_HIGH_RECALL,
            "CSV",
            "High-recall Indian product candidate inventory only",
            "secondary",
            "https://github.com/junioralive/Indian-Medicine-Dataset",
            "git clone",
            "false",
            "MIT",
            "true" if OPEN_LICENSE.exists() else "false",
            "true",
            "false",
            "check_downstream_before_redistribution",
            "none_observed_in_mit_license",
            today,
            git_head(RAW / "indian_medicine_dataset" / "Indian-Medicine-Dataset"),
            OPEN_CSV,
            raw_sha(raw_hashes, OPEN_CSV),
            "Complete CSV ingested as CANDIDATE_QUARANTINE; no local product truth promotion.",
        ),
        source_row(
            "NPPA_PHARMA_SAHI_DAAM",
            "NPPA Pharma Sahi Daam / IPDMS public indexes",
            SourceTier.A_OFFICIAL_INDIA,
            "public JSON endpoints",
            "Official Indian price/product index evidence",
            "official",
            "https://nppa.gov.in/en/pharma_sahi_daam",
            "public website endpoints: brandComboNew, formulationListNew, scheduledFormulationComboNew",
            "false",
            "government_public_website",
            "true",
            "not_stated",
            "true_public_interface_only",
            "not_assessed",
            "none_observed",
            today,
            "",
            NPPA_BRANDS,
            raw_sha(raw_hashes, NPPA_BRANDS),
            "Only source-wide public indexes parsed in v0; row-level product detail calls are deferred.",
        ),
        source_row(
            "PMBI_JAN_AUSHADHI",
            "PMBI Jan Aushadhi Product MRP List",
            SourceTier.A_OFFICIAL_INDIA,
            "public web app / manual CSV-PDF",
            "Official Jan Aushadhi product/MRP list evidence",
            "official",
            "https://janaushadhi.gov.in/product-portfolio/product-mrp-list",
            "public web page plus manual download when CSV/PDF endpoint is not stable",
            "guest_token_public_client",
            "government_public_website",
            "true",
            "manual_download_required",
            "false_v0",
            "not_assessed",
            "none_observed",
            today,
            "",
            PMBI_PROBE,
            raw_sha(raw_hashes, PMBI_PROBE),
            "Public POST requires pageIndex/pageSize and timed out with valid guest token; no scraper or admin endpoint used.",
        ),
        source_row(
            "CDSCO_APPROVED_NEW_DRUGS",
            "CDSCO approved new drugs index",
            SourceTier.A_OFFICIAL_INDIA,
            "HTML index of official documents",
            "Official approved new drug/FDC document metadata",
            "official",
            "https://cdsco.gov.in/opencms/opencms/en/Approval_new/Approved-New-Drugs/",
            "public HTML index",
            "false",
            "government_public_website",
            "true",
            "not_stated",
            "true_public_interface_only",
            "not_assessed",
            "none_observed",
            today,
            "",
            CDSCO_NEW,
            raw_sha(raw_hashes, CDSCO_NEW),
            "Document metadata parsed; PDF contents not interpreted in v0.",
        ),
        source_row(
            "CDSCO_FDC_MARKETING",
            "CDSCO FDC/New Drugs Marketing index",
            SourceTier.A_OFFICIAL_INDIA,
            "HTML index of official documents",
            "Official FDC/new drugs document metadata",
            "official",
            "https://cdsco.gov.in/opencms/opencms/en/Approval_new/FDC-New-Drugs-Marketing/",
            "public HTML index",
            "false",
            "government_public_website",
            "true",
            "not_stated",
            "true_public_interface_only",
            "not_assessed",
            "none_observed",
            today,
            "",
            CDSCO_FDC,
            raw_sha(raw_hashes, CDSCO_FDC),
            "Document metadata parsed; PDF contents not interpreted in v0.",
        ),
        source_row(
            "NLEM_2022",
            "National List of Essential Medicines 2022",
            SourceTier.A_OFFICIAL_INDIA,
            "official PDF",
            "Essential generic/formulation evidence; not brand identity",
            "official",
            "https://nppa.gov.in/en/nlem2022",
            "public PDF download",
            "false",
            "government_public_website",
            "true",
            "true_pdf",
            "true_public_interface_only",
            "not_assessed",
            "none_observed",
            today,
            "2022",
            NLEM_PDF,
            raw_sha(raw_hashes, NLEM_PDF),
            "Text extracted with pdftotext when available; no OCR.",
        ),
        source_row(
            "BODHI_M",
            "BODHI-M clinical concept-drug-lab KG",
            SourceTier.D_CONTEXT_ONLY,
            "JSONL triples",
            "Contextual treatment/lab graph only; not Indian brand/product/manufacturer truth",
            "secondary",
            "https://huggingface.co/datasets/ekacare/BODHI-M",
            "git clone from Hugging Face dataset",
            "false",
            "CC BY-NC 4.0",
            "true",
            "true",
            "false_context_only",
            "non_commercial_only",
            "non_commercial_license",
            today,
            git_head(RAW / "bodhi_m" / "ekacare_BODHI-M"),
            BODHI_TRIPLES,
            raw_sha(raw_hashes, BODHI_TRIPLES),
            "Loaded only into knowledge/context; excluded from brand identity establishment.",
        ),
        source_row(
            "PUBCHEM",
            "PubChem PUG REST",
            SourceTier.E_GLOBAL_ENRICHMENT,
            "cached public API client",
            "Ingredient enrichment only",
            "official",
            "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            "rate-limited API client scaffold",
            "false",
            "public_domain_government_data",
            "pending_per_record",
            "api_terms_apply",
            "true_rate_limited",
            "not_assessed",
            "none_observed",
            today,
            "",
            RAW / "pubchem",
            "",
            "No PubChem queries executed in v0; client rejects broad terms/brand identity.",
        ),
        source_row(
            "RXNORM_RXNAV",
            "RxNorm/RxNav/RxClass",
            SourceTier.E_GLOBAL_ENRICHMENT,
            "public API enrichment",
            "Global ontology crosswalk enrichment only",
            "official",
            "https://lhncbc.nlm.nih.gov/RxNav/",
            "not_ingested_v0",
            "false",
            "UMLS/RxNorm terms apply",
            "pending",
            "api_terms_apply",
            "true_rate_limited",
            "not_assessed",
            "not_required_for_local_validity",
            today,
            "",
            RAW / "rxnorm",
            "",
            "RxNorm not required for local Indian product validity.",
        ),
        source_row(
            "DRUGSETU",
            "DrugSetu Indian medicine database API",
            SourceTier.B_AUTHENTICATED_API,
            "authenticated API",
            "Authenticated Indian medicine product evidence when key is available",
            "secondary_authenticated",
            "https://drugsetu.com/",
            "X-API-Key to https://api.drugsetu.in/v1/medicines/search",
            "true",
            "contract_required",
            "false",
            "api_contract_required",
            "false_without_key",
            "contract_required",
            "contract_required",
            today,
            "",
            RAW / "drugsetu",
            "",
            "AUTH_REQUIRED; environment variable DRUGSETU_API_KEY only, no key stored.",
        ),
        source_row(
            "EKA_INDIAN_BRANDED_DRUGS",
            "Eka Indian Branded Drugs API",
            SourceTier.B_AUTHENTICATED_API,
            "authenticated API or MCP",
            "Authenticated Indian branded drug evidence when access is granted",
            "secondary_authenticated",
            "https://developer.eka.care/eka-medai/indian_branded_drugs",
            "access pending",
            "true",
            "contract_required",
            "false",
            "api_contract_required",
            "false_without_access",
            "contract_required",
            "contract_required",
            today,
            "",
            RAW,
            "",
            "EKA_ACCESS_PENDING; no scraping.",
        ),
        source_row(
            "MIMS_CIMS_INDIA",
            "MIMS/CIMS India",
            SourceTier.PROHIBITED,
            "commercial website/API",
            "Potential manual validation only after permission",
            "commercial",
            "https://www.mims.com/india",
            "PERMISSION_REQUIRED; DO_NOT_AUTOMATE",
            "true_or_permission_required",
            "commercial_restricted",
            "true_terms_observed",
            "false_without_contract",
            "false",
            "false_without_contract",
            "permission_required",
            today,
            "",
            RAW,
            "",
            "Use of automated systems is prohibited by MIMS terms; no scraper exists.",
        ),
    ]
    pd.DataFrame(rows, columns=SOURCE_REGISTRY_COLUMNS).to_csv(PROVENANCE / "source_registry.csv", index=False)


def source_row(
    source_id: str,
    source_name: str,
    tier: SourceTier,
    source_type: str,
    authority_scope: str,
    official_or_secondary: str,
    url: str,
    access_method: str,
    requires_auth: str,
    license_value: str,
    license_verified: str,
    bulk_download_allowed: str,
    automated_query_allowed: str,
    redistribution_allowed: str,
    ai_use_restriction: str,
    retrieval_date: str,
    source_version: str,
    raw_snapshot_path: Path,
    raw_sha256: str,
    notes: str,
) -> dict[str, str]:
    return {
        "source_id": source_id,
        "source_name": source_name,
        "source_tier": str(tier),
        "source_type": source_type,
        "authority_scope": authority_scope,
        "official_or_secondary": official_or_secondary,
        "url": url,
        "access_method": access_method,
        "requires_auth": requires_auth,
        "license": license_value,
        "license_verified": license_verified,
        "bulk_download_allowed": bulk_download_allowed,
        "automated_query_allowed": automated_query_allowed,
        "redistribution_allowed": redistribution_allowed,
        "ai_use_restriction": ai_use_restriction,
        "retrieval_date": retrieval_date,
        "source_version": source_version,
        "raw_snapshot_path": rel(raw_snapshot_path),
        "raw_sha256": raw_sha256,
        "parser_version": PARSER_VERSION,
        "notes": notes,
    }


def git_head(path: Path) -> str:
    if not (path / ".git").exists():
        return ""
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def load_json_tolerant(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_bytes().decode("utf-8", errors="replace"))


def parse_nppa_indexes(raw_hashes: dict[str, str]) -> dict[str, Any]:
    brands = load_json_tolerant(NPPA_BRANDS)
    formulations = load_json_tolerant(NPPA_FORMULATIONS)
    scheduled = load_json_tolerant(NPPA_SCHEDULED)
    brand_rows = []
    brand_family_norms = set()
    brand_name_norms = set()
    for row in brands:
        raw_brand = display_text(row.get("brandName"))
        family_display, family_norm = extract_brand_family(raw_brand)
        brand_rows.append(
            {
                "source_id": "NPPA_PHARMA_SAHI_DAAM",
                "source_brand_id": row.get("brandId", ""),
                "raw_brand_name": raw_brand,
                "normalized_brand_name": normalize_text(raw_brand),
                "brand_family_display": family_display,
                "normalized_brand_family": family_norm,
            }
        )
        if family_norm:
            brand_family_norms.add(family_norm)
        if raw_brand:
            brand_name_norms.add(normalize_text(raw_brand))
    formulation_rows = [
        {
            "source_id": "NPPA_PHARMA_SAHI_DAAM",
            "source_formulation_id": row.get("formulationId", ""),
            "raw_formulation_name": display_text(row.get("formulationName")),
            "normalized_formulation_name": normalize_text(row.get("formulationName")),
        }
        for row in formulations
    ]
    scheduled_rows = [
        {
            "source_id": "NPPA_PHARMA_SAHI_DAAM",
            "source_scheduled_formulation_id": row.get("schFormulationId", ""),
            "raw_scheduled_formulation_name": display_text(row.get("schFormulationName")),
            "normalized_scheduled_formulation_name": normalize_text(row.get("schFormulationName")),
        }
        for row in scheduled
    ]
    pd.DataFrame(brand_rows).to_csv(STAGING / "nppa_brand_index.csv", index=False)
    pd.DataFrame(formulation_rows).to_csv(STAGING / "nppa_formulation_index.csv", index=False)
    pd.DataFrame(scheduled_rows).to_csv(STAGING / "nppa_scheduled_formulation_index.csv", index=False)
    return {
        "brand_family_norms": brand_family_norms,
        "brand_name_norms": brand_name_norms,
        "counts": {
            "brand_index_rows": len(brand_rows),
            "formulation_index_rows": len(formulation_rows),
            "scheduled_formulation_index_rows": len(scheduled_rows),
            "unique_brand_families": len(brand_family_norms),
        },
    }


def parse_cdsco_indexes(raw_hashes: dict[str, str]) -> dict[str, int]:
    counts = {}
    for source_id, path, out_name in [
        ("CDSCO_APPROVED_NEW_DRUGS", CDSCO_NEW, "cdsco_approved_new_drugs_index.csv"),
        ("CDSCO_FDC_MARKETING", CDSCO_FDC, "cdsco_fdc_marketing_index.csv"),
    ]:
        rows: list[dict[str, Any]]
        try:
            table = pd.read_html(path)[0]
            rows = table.to_dict("records")
        except Exception:
            rows = []
        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    "source_id": source_id,
                    "source_document_title": display_text(row.get("Title")),
                    "release_date": display_text(row.get("Release Date")),
                    "pdf_size": display_text(row.get("Pdf Size")),
                    "raw_snapshot_path": rel(path),
                    "raw_sha256": raw_sha(raw_hashes, path),
                }
            )
        pd.DataFrame(normalized_rows).to_csv(STAGING / out_name, index=False)
        counts[out_name.replace(".csv", "_rows")] = len(normalized_rows)
    return counts


def parse_nlem(raw_hashes: dict[str, str]) -> dict[str, int]:
    if NLEM_PDF.exists() and shutil.which("pdftotext") and not NLEM_TEXT.exists():
        subprocess.run(["pdftotext", "-layout", str(NLEM_PDF), str(NLEM_TEXT)], check=False)
    lines = []
    if NLEM_TEXT.exists():
        for i, line in enumerate(NLEM_TEXT.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            text = display_text(line)
            if not text:
                continue
            normalized = normalize_text(text)
            if len(normalized) < 3:
                continue
            if re.match(r"^\d+(\.\d+)*\s+", normalized) or re.match(r"^[a-z][a-z -]{2,}\s+[pst, ]+\s+", normalized):
                lines.append(
                    {
                        "source_id": "NLEM_2022",
                        "line_number": i,
                        "raw_text": text,
                        "normalized_text": normalized,
                        "raw_snapshot_path": rel(NLEM_TEXT),
                        "raw_sha256": raw_sha(raw_hashes, NLEM_PDF),
                    }
                )
    pd.DataFrame(lines).to_csv(STAGING / "nlem2022_candidate_lines.csv", index=False)
    return {"candidate_lines": len(lines), "pdf_bytes": NLEM_PDF.stat().st_size if NLEM_PDF.exists() else 0}


def parse_bodhi_context() -> dict[str, int]:
    nodes: dict[tuple[str, str], dict[str, str]] = {}
    edges: list[dict[str, str]] = []
    if BODHI_TRIPLES.exists():
        with BODHI_TRIPLES.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                obj = json.loads(line)
                head_id = str(obj.get("head", ""))
                tail_id = str(obj.get("tail", ""))
                head_type = str(obj.get("head_type", ""))
                tail_type = str(obj.get("tail_type", ""))
                nodes[(head_id, head_type)] = {"node_id": head_id, "node_type": head_type, "source_id": "BODHI_M"}
                nodes[(tail_id, tail_type)] = {"node_id": tail_id, "node_type": tail_type, "source_id": "BODHI_M"}
                edges.append(
                    {
                        "edge_id": stable_prefixed_id("BODHIE", idx, head_id, obj.get("relation", ""), tail_id),
                        "source_id": "BODHI_M",
                        "head_id": head_id,
                        "head_type": head_type,
                        "relation": str(obj.get("relation", "")),
                        "tail_id": tail_id,
                        "tail_type": tail_type,
                        "properties_json": json.dumps(obj.get("properties", {}), sort_keys=True),
                        "kg_state": KGState.CANDIDATE_QUARANTINE.value,
                        "authority": Authority.CONTEXT_ONLY.value,
                    }
                )
    pd.DataFrame(nodes.values()).to_csv(CONTEXT / "bodhi_nodes.csv", index=False)
    pd.DataFrame(edges).to_csv(CONTEXT / "bodhi_edges.csv", index=False)
    return {"nodes": len(nodes), "edges": len(edges)}


def build_from_open_dataset(
    raw_hashes: dict[str, str],
    nppa_brand_family_norms: set[str],
    nppa_brand_name_norms: set[str],
) -> dict[str, int]:
    ingredients: dict[str, Ingredient] = {}
    brand_families: dict[str, BrandFamily] = {}
    formulations: dict[str, ClinicalFormulation] = {}
    formulation_components: dict[str, FormulationComponent] = {}
    brand_products: dict[str, BrandProduct] = {}
    package_skus: dict[str, PackageSKU] = {}
    price_observations: dict[str, PriceObservation] = {}
    companies: dict[str, CompanyEntity] = {}
    company_relationships: dict[str, CompanyRelationship] = {}
    aliases: dict[str, Alias] = {}
    alias_evidence_links: dict[str, AliasEvidenceLink] = {}
    evidence: dict[str, SourceEvidence] = {}
    crosswalks: dict[str, dict[str, str]] = {}
    retrieval_docs: list[dict[str, str]] = []
    quarantine_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, str]] = []
    staging_rows: list[dict[str, Any]] = []
    fdc_products = 0
    matched_l1 = 0
    source_sha = raw_sha(raw_hashes, OPEN_CSV)

    for chunk in pd.read_csv(OPEN_CSV, chunksize=25000):
        for row in chunk.to_dict("records"):
            source_product_id = display_text(row.get("id"))
            raw_brand_name = display_text(row.get("name"))
            normalized_brand_name = normalize_text(raw_brand_name)
            family_display, family_norm = extract_brand_family(raw_brand_name)
            brand_family_id = stable_prefixed_id("BFAM", "open", family_norm)
            components = parse_composition(row.get("short_composition1"), row.get("short_composition2"))
            dosage_form = infer_dosage_form(raw_brand_name, row.get("pack_size_label"))
            signature = formulation_signature(components, dosage_form)
            formulation_id = stable_prefixed_id("FORM", signature or "unknown")
            brand_product_id = stable_prefixed_id("BPROD", "INDIAN_MEDICINE_DATASET", source_product_id)
            package_sku_id = stable_prefixed_id("SKU", brand_product_id, display_text(row.get("pack_size_label")))
            evidence_id = stable_prefixed_id("EVID", "open-record", source_product_id)
            raw_record = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)

            brand_families.setdefault(
                brand_family_id,
                BrandFamily(
                    brand_family_id,
                    family_display,
                    family_norm,
                    KGState.CANDIDATE_QUARANTINE.value,
                    Authority.OPEN_DERIVATIVE.value,
                ),
            )
            formulations.setdefault(
                formulation_id,
                ClinicalFormulation(
                    formulation_id,
                    dosage_form,
                    "",
                    "",
                    signature,
                    KGState.CANDIDATE_QUARANTINE.value,
                    Authority.OPEN_DERIVATIVE.value,
                ),
            )
            brand_products[brand_product_id] = BrandProduct(
                brand_product_id,
                brand_family_id,
                formulation_id,
                source_product_id,
                raw_brand_name,
                normalized_brand_name,
                display_text(row.get("type")),
                str(row.get("Is_discontinued")),
                KGState.CANDIDATE_QUARANTINE.value,
                Authority.OPEN_DERIVATIVE.value,
            )
            package_skus[package_sku_id] = PackageSKU(
                package_sku_id,
                brand_product_id,
                display_text(row.get("pack_size_label")),
                KGState.CANDIDATE_QUARANTINE.value,
                Authority.OPEN_DERIVATIVE.value,
            )
            price_raw = display_text(row.get("price(₹)"))
            if price_raw:
                price_id = stable_prefixed_id("PRICE", package_sku_id, price_raw)
                price_observations[price_id] = PriceObservation(
                    price_id,
                    package_sku_id,
                    price_raw,
                    "INR",
                    "INDIAN_MEDICINE_DATASET",
                    KGState.CANDIDATE_QUARANTINE.value,
                    Authority.OPEN_DERIVATIVE.value,
                )

            company_norm = normalize_text(row.get("manufacturer_name"))
            company_id = stable_prefixed_id("COMP", company_norm)
            if company_norm:
                companies.setdefault(
                    company_id,
                    CompanyEntity(
                        company_id,
                        display_text(row.get("manufacturer_name")),
                        company_norm,
                        KGState.CANDIDATE_QUARANTINE.value,
                        Authority.OPEN_DERIVATIVE.value,
                    ),
                )
                relationship_id = stable_prefixed_id("CREL", company_id, brand_product_id, "SOURCE_UNSPECIFIED_COMPANY")
                company_relationships[relationship_id] = CompanyRelationship(
                    relationship_id,
                    company_id,
                    "BrandProduct",
                    brand_product_id,
                    "SOURCE_UNSPECIFIED_COMPANY",
                    KGState.CANDIDATE_QUARANTINE.value,
                    Authority.OPEN_DERIVATIVE.value,
                )

            evidence[evidence_id] = SourceEvidence(
                evidence_id,
                "INDIAN_MEDICINE_DATASET",
                "BrandProduct",
                brand_product_id,
                "source_record",
                raw_record,
                rel(OPEN_CSV),
                source_sha,
                PARSER_VERSION,
                KGState.CANDIDATE_QUARANTINE.value,
                Authority.OPEN_DERIVATIVE.value,
            )

            alias_id = stable_prefixed_id("ALIAS", "brand", normalized_brand_name)
            aliases.setdefault(
                alias_id,
                Alias(
                    alias_id,
                    raw_brand_name,
                    normalized_brand_name,
                    "BRAND_PRODUCT_NAME",
                    KGState.CANDIDATE_QUARANTINE.value,
                    Authority.OPEN_DERIVATIVE.value,
                ),
            )
            alias_link_id = stable_prefixed_id("ALINK", alias_id, evidence_id, brand_product_id)
            alias_evidence_links[alias_link_id] = AliasEvidenceLink(
                alias_link_id,
                alias_id,
                evidence_id,
                "BrandProduct",
                brand_product_id,
            )

            component_payload = []
            if len(components) > 1:
                fdc_products += 1
            for component_order, component in enumerate(components, start=1):
                ingredient_id = stable_prefixed_id("ING", component.normalized_ingredient)
                ingredients.setdefault(
                    ingredient_id,
                    Ingredient(
                        ingredient_id,
                        component.ingredient_name,
                        component.normalized_ingredient,
                        KGState.CANDIDATE_QUARANTINE.value,
                        Authority.OPEN_DERIVATIVE.value,
                    ),
                )
                component_id = stable_prefixed_id("FCOMP", formulation_id, ingredient_id, component_order)
                formulation_components[component_id] = FormulationComponent(
                    component_id,
                    formulation_id,
                    ingredient_id,
                    component_order,
                    component.raw_component_text,
                    component.strength_text,
                    component.normalized_strength,
                )
                ingredient_alias_id = stable_prefixed_id("ALIAS", "ingredient", component.normalized_ingredient)
                aliases.setdefault(
                    ingredient_alias_id,
                    Alias(
                        ingredient_alias_id,
                        component.ingredient_name,
                        component.normalized_ingredient,
                        "INGREDIENT_NAME",
                        KGState.CANDIDATE_QUARANTINE.value,
                        Authority.OPEN_DERIVATIVE.value,
                    ),
                )
                ingredient_link_id = stable_prefixed_id("ALINK", ingredient_alias_id, evidence_id, ingredient_id)
                alias_evidence_links[ingredient_link_id] = AliasEvidenceLink(
                    ingredient_link_id,
                    ingredient_alias_id,
                    evidence_id,
                    "Ingredient",
                    ingredient_id,
                )
                component_payload.append(f"{component.ingredient_name} {component.strength_text}".strip())

            l1_match = family_norm in nppa_brand_family_norms or normalized_brand_name in nppa_brand_name_norms
            if l1_match:
                matched_l1 += 1
            validation_rows.append(
                {
                    "source_product_id": source_product_id,
                    "raw_brand_name": raw_brand_name,
                    "normalized_brand_family": family_norm,
                    "l1_brand_family_overlap": "MATCH" if l1_match else "NOT_COMPARABLE",
                    "l2_exact_brand_product": "MATCH" if normalized_brand_name in nppa_brand_name_norms else "NOT_COMPARABLE",
                    "l3_composition": "NOT_COMPARABLE",
                    "l4_pack_or_sku": "NOT_COMPARABLE",
                    "l5_price": "NOT_COMPARABLE",
                    "promotion_decision": "NO_PROMOTION_SOURCE_INDEX_ONLY",
                    "notes": "NPPA v0 index lacks parsed source-wide composition/package detail.",
                }
            )
            quarantine_row = {
                "source_product_id": source_product_id,
                "brand_product_id": brand_product_id,
                "brand_family_id": brand_family_id,
                "formulation_id": formulation_id,
                "package_sku_id": package_sku_id,
                "raw_brand_name": raw_brand_name,
                "normalized_brand_name": normalized_brand_name,
                "raw_manufacturer": display_text(row.get("manufacturer_name")),
                "normalized_manufacturer": company_norm,
                "pack_size": display_text(row.get("pack_size_label")),
                "raw_composition_1": display_text(row.get("short_composition1")),
                "raw_composition_2": display_text(row.get("short_composition2")),
                "parsed_components": " | ".join(component_payload),
                "price": price_raw,
                "discontinued": str(row.get("Is_discontinued")),
                "medicine_type": display_text(row.get("type")),
                "source_id": "INDIAN_MEDICINE_DATASET",
                "kg_state": KGState.CANDIDATE_QUARANTINE.value,
                "authority": Authority.OPEN_DERIVATIVE.value,
            }
            quarantine_rows.append(quarantine_row)
            staging_rows.append(quarantine_row)
            retrieval_docs.append(
                {
                    "brand_product_id": brand_product_id,
                    "kg_state": KGState.CANDIDATE_QUARANTINE.value,
                    "brand_text": raw_brand_name,
                    "ingredient_text": " ".join(component_payload),
                    "formulation_text": signature,
                    "alias_text": raw_brand_name,
                    "manufacturer_text": display_text(row.get("manufacturer_name")),
                    "search_document": " ".join(
                        part
                        for part in [
                            raw_brand_name,
                            family_display,
                            " ".join(component_payload),
                            display_text(row.get("manufacturer_name")),
                            display_text(row.get("pack_size_label")),
                        ]
                        if part
                    ),
                }
            )

    write_csv(QUARANTINE / "open_indian_products.csv", quarantine_rows)
    write_csv(STAGING / "open_indian_products_staging.csv", staging_rows)
    write_csv(CROSSWALKS / "open_vs_nppa_validation.csv", validation_rows)
    write_csv(REPORTS / "open_vs_nppa_validation.csv", validation_rows)
    write_csv(CANONICAL / "ingredients.csv", [row_dict(x) for x in ingredients.values()])
    write_csv(CANONICAL / "brand_families.csv", [row_dict(x) for x in brand_families.values()])
    write_csv(CANONICAL / "clinical_formulations.csv", [row_dict(x) for x in formulations.values()])
    write_csv(CANONICAL / "formulation_components.csv", [row_dict(x) for x in formulation_components.values()])
    write_csv(CANONICAL / "brand_products.csv", [row_dict(x) for x in brand_products.values()])
    write_csv(CANONICAL / "package_skus.csv", [row_dict(x) for x in package_skus.values()])
    write_csv(CANONICAL / "company_entities.csv", [row_dict(x) for x in companies.values()])
    write_csv(CANONICAL / "company_relationships.csv", [row_dict(x) for x in company_relationships.values()])
    write_csv(CANONICAL / "aliases.csv", [row_dict(x) for x in aliases.values()])
    write_csv(CANONICAL / "alias_evidence_links.csv", [row_dict(x) for x in alias_evidence_links.values()])
    write_csv(CANONICAL / "source_evidence.csv", [row_dict(x) for x in evidence.values()])
    if crosswalks:
        write_csv(CANONICAL / "ontology_crosswalks.csv", list(crosswalks.values()))
    else:
        pd.DataFrame(
            columns=[
                "crosswalk_id",
                "local_entity_type",
                "local_entity_id",
                "external_system",
                "external_id",
                "match_status",
                "kg_state",
                "authority",
                "evidence_id",
            ]
        ).to_csv(CANONICAL / "ontology_crosswalks.csv", index=False)
    write_csv(CANONICAL / "price_observations.csv", [row_dict(x) for x in price_observations.values()])
    write_csv(CANONICAL / "retrieval_documents.csv", retrieval_docs)
    write_sqlite()
    return {
        "open_dataset_rows": len(brand_products),
        "open_dataset_fdc_products": fdc_products,
        "open_vs_nppa_l1_matches": matched_l1,
        "ingredients": len(ingredients),
        "brand_families": len(brand_families),
        "clinical_formulations": len(formulations),
        "formulation_components": len(formulation_components),
        "brand_products": len(brand_products),
        "package_skus": len(package_skus),
        "price_observations": len(price_observations),
        "company_entities": len(companies),
        "company_relationships": len(company_relationships),
        "aliases": len(aliases),
        "alias_evidence_links": len(alias_evidence_links),
        "source_evidence": len(evidence),
        "supported_products": 0,
        "authoritative_products": 0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def write_sqlite() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    try:
        for csv_path in sorted(CANONICAL.glob("*.csv")) + sorted(QUARANTINE.glob("*.csv")) + sorted(CONTEXT.glob("*.csv")):
            if csv_path.stat().st_size == 0:
                continue
            table_name = csv_path.stem
            if csv_path.parent == QUARANTINE and table_name == "open_indian_products":
                table_name = "quarantine_open_indian_products"
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        create_views(conn)
    finally:
        conn.close()


def create_views(conn: sqlite3.Connection) -> None:
    view_sql = {
        "v_brand_search": """
            SELECT bp.brand_product_id, bp.raw_brand_name, bp.normalized_brand_name, bf.canonical_name AS brand_family,
                   bp.kg_state, bp.authority
            FROM brand_products bp
            LEFT JOIN brand_families bf ON bp.brand_family_id = bf.brand_family_id
        """,
        "v_ingredient_search": """
            SELECT ingredient_id, canonical_name, normalized_name, kg_state, authority
            FROM ingredients
        """,
        "v_formulation_search": """
            SELECT formulation_id, dosage_form, normalized_component_signature, kg_state, authority
            FROM clinical_formulations
        """,
        "v_fdc_search": """
            SELECT fc.formulation_id, COUNT(*) AS component_count,
                   GROUP_CONCAT(i.canonical_name || ' ' || fc.strength_text, ' + ') AS components
            FROM formulation_components fc
            LEFT JOIN ingredients i ON fc.ingredient_id = i.ingredient_id
            GROUP BY fc.formulation_id
            HAVING COUNT(*) > 1
        """,
        "v_source_evidence": """
            SELECT evidence_id, source_id, entity_type, entity_id, field_name, raw_snapshot_path, raw_sha256, kg_state, authority
            FROM source_evidence
        """,
        "v_candidate_quarantine": """
            SELECT * FROM quarantine_open_indian_products
        """,
        "v_context_drug_links": """
            SELECT edge_id, head_id, head_type, relation, tail_id, tail_type
            FROM bodhi_edges
            WHERE head_type = 'Drug' OR tail_type = 'Drug'
        """,
    }
    for name, sql in view_sql.items():
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(f"CREATE VIEW {name} AS {sql}")
    conn.commit()


def write_duckdb_if_available() -> None:
    try:
        import duckdb  # type: ignore
    except Exception:
        return
    duckdb_path = KNOWLEDGE / "medication_knowledge.duckdb"
    if duckdb_path.exists():
        duckdb_path.unlink()
    conn = duckdb.connect(str(duckdb_path))
    try:
        for csv_path in sorted(CANONICAL.glob("*.csv")) + sorted(QUARANTINE.glob("*.csv")) + sorted(CONTEXT.glob("*.csv")):
            if csv_path.stat().st_size == 0:
                continue
            table_name = csv_path.stem
            if csv_path.parent == QUARANTINE and table_name == "open_indian_products":
                table_name = "quarantine_open_indian_products"
            frame = pd.read_csv(csv_path, dtype=str).fillna("")
            conn.register("stage1b_frame", frame)
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM stage1b_frame")
            conn.unregister("stage1b_frame")
    finally:
        conn.close()


def write_access_reports(raw_hashes: dict[str, str], counts: dict[str, Any]) -> None:
    (REPORTS / "PMBI_MANUAL_DOWNLOAD_REQUIRED.md").write_text(
        """# PMBI Manual Download Required

The PMBI Jan Aushadhi Product MRP List page was fetched from the official public URL, and the shipped JavaScript exposes a website product endpoint. A direct unauthenticated POST returned an HTTP 500 response; a public guest-token path exists, but a valid guest-token request with pageIndex/pageSize timed out during the Stage 1B probe.

No admin endpoints, OCR, CAPTCHA bypass, credentialed calls, or scraping were used. To ingest PMBI later, manually download the official CSV/PDF from the PMBI page and place it in `knowledge/manual_inputs/pmbi/` with the retrieval date in the filename.
""",
        encoding="utf-8",
    )
    (REPORTS / "DRUGSETU_ACCESS_REQUIREMENTS.md").write_text(
        """# DrugSetu Access Requirements

Status: `AUTH_REQUIRED`

Expected endpoint: `https://api.drugsetu.in/v1/medicines/search`

Required credential handling: set `DRUGSETU_API_KEY` in the runtime environment and send it as `X-API-Key`. Do not store the key in source files, reports, raw snapshots, or SQLite exports.

No DrugSetu request was executed in Stage 1B because no API key was available.
""",
        encoding="utf-8",
    )
    (REPORTS / "SOURCE_ACCESS_AND_LICENSE_AUDIT.md").write_text(
        f"""# Source Access And License Audit

Retrieval date: {date.today().isoformat()}

## Allowed And Parsed

- `INDIAN_MEDICINE_DATASET`: MIT-licensed open CSV, ingested as `CANDIDATE_QUARANTINE`; raw SHA `{raw_sha(raw_hashes, OPEN_CSV)}`.
- `NPPA_PHARMA_SAHI_DAAM`: official public indexes parsed: {counts.get('nppa_brand_index_rows', 0)} brand rows, {counts.get('nppa_formulation_index_rows', 0)} formulation rows, {counts.get('nppa_scheduled_formulation_index_rows', 0)} scheduled formulation rows.
- `CDSCO_APPROVED_NEW_DRUGS` and `CDSCO_FDC_MARKETING`: official HTML document indexes parsed as metadata only.
- `NLEM_2022`: official PDF downloaded and text-extracted with `pdftotext`; used for essential generic/formulation context, not brand identity.
- `BODHI_M`: CC BY-NC 4.0 contextual KG loaded into `knowledge/context` only.

## Restricted Or Pending

- `PMBI_JAN_AUSHADHI`: official page observed, but stable bulk endpoint was not established in v0. Manual download instructions are in `PMBI_MANUAL_DOWNLOAD_REQUIRED.md`.
- `DRUGSETU`: `AUTH_REQUIRED`; use only `DRUGSETU_API_KEY` in environment.
- `EKA_INDIAN_BRANDED_DRUGS`: `EKA_ACCESS_PENDING`; no scraping.
- `MIMS_CIMS_INDIA`: `PERMISSION_REQUIRED`, `DO_NOT_AUTOMATE`; automated collection is prohibited by MIMS terms.

## Guardrails

No prescription-derived medication strings, MIMS/CIMS scraping, CAPTCHA bypass, stored API keys, OCR, LLM promotion, or lexical-only promotion were used.
""",
        encoding="utf-8",
    )


def write_stage_report(raw_hashes: dict[str, str], counts: dict[str, Any]) -> None:
    source_registry = pd.read_csv(PROVENANCE / "source_registry.csv")
    source_count = len(source_registry)
    source_wide_paths = [
        "knowledge/raw/indian_medicine_dataset/Indian-Medicine-Dataset/DATA/indian_medicine_data.csv",
        "knowledge/raw/nppa/brandComboNew.json",
        "knowledge/raw/nppa/formulationListNew.json",
        "knowledge/raw/nppa/scheduledFormulationComboNew.json",
        "knowledge/raw/cdsco/cdsco_new_drugs.html",
        "knowledge/raw/cdsco/fdc_marketing.html",
        "knowledge/raw/nlem/nlem-2022.pdf",
        "knowledge/raw/bodhi_m/ekacare_BODHI-M/data/triples.jsonl",
    ]
    missing = [path for path in source_wide_paths if path not in raw_hashes]
    report = f"""# Stage 1B India-Aware Medication KB Report

## 1. Build Scope

Stage 1B built a source-wide India-aware medication knowledge base v0 independently of prescription-specific medication strings. The build inputs manifest contains only `knowledge/raw/...` files.

## 2. Source Registry

`knowledge/provenance/source_registry.csv` contains {source_count} entries across official India, authenticated-pending, open high-recall, context-only, global enrichment, and prohibited/manual-validation sources.

## 3. Raw Snapshot Hashing

{len(raw_hashes)} raw files were hashed. Missing expected source-wide paths: {missing or 'none'}.

## 4. Open Dataset Ingest

The junioralive CSV produced {counts.get('open_dataset_rows', 0)} quarantined product candidates from SHA `{raw_sha(raw_hashes, OPEN_CSV)}`. The updated CSV was preserved but not selected as the v0 primary ingest table.

## 5. Quarantine Policy

All open-derived rows have `kg_state=CANDIDATE_QUARANTINE` and `authority=OPEN_DERIVATIVE`. No lexical match promoted a candidate to authoritative or supported canonical status.

## 6. NPPA Coverage

NPPA public indexes parsed: {counts.get('nppa_brand_index_rows', 0)} brand rows, {counts.get('nppa_formulation_index_rows', 0)} formulation rows, and {counts.get('nppa_scheduled_formulation_index_rows', 0)} scheduled formulation rows.

## 7. Open Versus NPPA

`knowledge/reports/open_vs_nppa_validation.csv` contains {counts.get('open_dataset_rows', 0)} rows. L1 brand/family overlaps: {counts.get('open_vs_nppa_l1_matches', 0)}. L3-L5 are `NOT_COMPARABLE` in v0 because source-wide NPPA product-detail composition/package/price extraction was deferred.

## 8. PMBI Status

PMBI is `PMBI_MANUAL_DOWNLOAD_REQUIRED` for v0. Public page and JavaScript were preserved; no admin endpoint, OCR, credentialed call, or scraper was used.

## 9. CDSCO Status

CDSCO metadata rows parsed: {counts.get('cdsco_cdsco_approved_new_drugs_index_rows', 0)} approved-new-drug documents and {counts.get('cdsco_cdsco_fdc_marketing_index_rows', 0)} FDC/new-drug marketing documents. PDF content interpretation is deferred.

## 10. NLEM Status

NLEM 2022 PDF bytes: {counts.get('nlem_pdf_bytes', 0)}. Candidate text lines exported: {counts.get('nlem_candidate_lines', 0)}. NLEM is generic/formulation context only, not brand identity.

## 11. BODHI-M Status

BODHI-M context exports contain {counts.get('bodhi_nodes', 0)} nodes and {counts.get('bodhi_edges', 0)} edges. BODHI-M cannot establish Indian brand/product/manufacturer truth.

## 12. Authenticated APIs

DrugSetu is `AUTH_REQUIRED`; Eka Indian Branded Drugs is `EKA_ACCESS_PENDING`. No scraping or substitute sources were used.

## 13. Prohibited Sources

MIMS/CIMS India is registered as `PERMISSION_REQUIRED` and `DO_NOT_AUTOMATE`. No scraper exists or is enabled.

## 14. Entity Counts

Ingredients: {counts.get('ingredients', 0)}; brand families: {counts.get('brand_families', 0)}; brand products: {counts.get('brand_products', 0)}; formulations: {counts.get('clinical_formulations', 0)}; formulation components: {counts.get('formulation_components', 0)}; package SKUs: {counts.get('package_skus', 0)}.

## 15. FDC Preservation

Open-derived FDC product candidates with more than one parsed component: {counts.get('open_dataset_fdc_products', 0)}. Components are first-class rows in `formulation_components.csv`.

## 16. Company Roles

Company entities: {counts.get('company_entities', 0)}. Open dataset company relationships use `SOURCE_UNSPECIFIED_COMPANY`, preserving the distinction from verified manufacturer/marketer roles.

## 17. Alias Evidence

Aliases: {counts.get('aliases', 0)}; alias evidence links: {counts.get('alias_evidence_links', 0)}. Alias links point to source evidence rows.

## 18. Source Evidence

Source evidence rows: {counts.get('source_evidence', 0)}. Every open-derived product row has a source-record evidence link to the hashed CSV snapshot.

## 19. Price Observations

Price observations: {counts.get('price_observations', 0)}. Prices from the open dataset remain quarantined observations, not official MRPs.

## 20. Retrieval Documents

`knowledge/canonical/retrieval_documents.csv` provides retrieval-ready text fields while keeping canonical entity tables separate.

## 21. SQLite

SQLite database: `knowledge/medication_knowledge.sqlite`. Required search views were created: `v_brand_search`, `v_ingredient_search`, `v_formulation_search`, `v_fdc_search`, `v_source_evidence`, `v_candidate_quarantine`, `v_context_drug_links`.

## 22. DuckDB

DuckDB export is created only when the `duckdb` Python package is available.

## 23. PubChem

`src/knowledge/pubchem_client.py` provides a <=5 requests/second cached scaffold for narrow ingredient enrichment only. No PubChem query was executed in v0.

## 24. RxNorm

RxNorm/RxNav is registered as optional global enrichment. RxNorm is not required for local Indian product validity.

## 25. Tri-State Comparison

Cross-validation uses `MATCH`, `CONFLICT`, and `NOT_COMPARABLE`; missing source fields are not conflicts.

## 26. Promotion Readiness

Supported products: {counts.get('supported_products', 0)}. Authoritative products: {counts.get('authoritative_products', 0)}. This is intentional until official/authenticated source evidence supports product-level claims.

## 27. Manual Validation Queue

MIMS/CIMS manual validation can be added later using internal IDs only. No MIMS/CIMS content was collected.

## 28. Determinism

IDs are stable hashes over source IDs, source product IDs, normalized names, formulation signatures, and package text. Parsers do not call LLMs or external APIs.

## 29. Non-Use Of Prescription Strings

The Stage 1B acquisition/build code reads only `knowledge/raw` sources and does not run prescription normalization.

## 30. Next Stage Boundary

STOP: prescription normalization remains out of scope for Stage 1B.
"""
    (REPORTS / "STAGE1B_INDIA_AWARE_KB_REPORT.md").write_text(report, encoding="utf-8")
    (REPORTS / "SOURCE_QUALITY_AND_COVERAGE_REPORT.md").write_text(
        f"""# Source Quality And Coverage Report

## Coverage Summary

- High-recall candidate products: {counts.get('open_dataset_rows', 0)}
- NPPA official brand index rows: {counts.get('nppa_brand_index_rows', 0)}
- NPPA official formulation rows: {counts.get('nppa_formulation_index_rows', 0)}
- NPPA official scheduled formulation rows: {counts.get('nppa_scheduled_formulation_index_rows', 0)}
- CDSCO official document index rows: {counts.get('cdsco_cdsco_approved_new_drugs_index_rows', 0) + counts.get('cdsco_cdsco_fdc_marketing_index_rows', 0)}
- NLEM candidate text lines: {counts.get('nlem_candidate_lines', 0)}
- BODHI contextual edges: {counts.get('bodhi_edges', 0)}

## Authority Boundaries

Official India sources can support official claims only at the field granularity actually parsed. Open derivative rows are useful for recall but remain quarantined. BODHI-M, PubChem, and RxNorm cannot establish Indian brand identity.

## Known Gaps

NPPA detailed product rows, PMBI bulk CSV/PDF ingestion, CDSCO PDF parsing, DrugSetu, and Eka authenticated APIs are deferred until access or stable public bulk retrieval is available.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
