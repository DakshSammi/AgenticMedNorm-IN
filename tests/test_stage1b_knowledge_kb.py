from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.knowledge.models import (
    Authority,
    ComparisonStatus,
    KGState,
    can_promote,
    compare_field,
    source_can_establish_brand_identity,
)
from src.knowledge.parsers import extract_brand_family, formulation_signature, parse_composition
from src.knowledge.pubchem_client import PubChemClient


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_source_registry_has_required_columns_and_hashes():
    registry = pd.read_csv(KNOWLEDGE / "provenance/source_registry.csv", dtype=str).fillna("")
    required = [
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
    assert registry.columns.tolist() == required
    open_row = registry.loc[registry["source_id"] == "INDIAN_MEDICINE_DATASET"].iloc[0]
    nppa_row = registry.loc[registry["source_id"] == "NPPA_PHARMA_SAHI_DAAM"].iloc[0]
    assert len(open_row["raw_sha256"]) == 64
    assert len(nppa_row["raw_sha256"]) == 64


def test_open_dataset_is_complete_and_quarantined():
    products = pd.read_csv(KNOWLEDGE / "quarantine/open_indian_products.csv", dtype=str).fillna("")
    assert len(products) == 253973
    assert set(products["kg_state"]) == {KGState.CANDIDATE_QUARANTINE.value}
    assert set(products["authority"]) == {Authority.OPEN_DERIVATIVE.value}


def test_no_duplicate_canonical_ids():
    for filename, key in [
        ("ingredients.csv", "ingredient_id"),
        ("brand_families.csv", "brand_family_id"),
        ("brand_products.csv", "brand_product_id"),
        ("clinical_formulations.csv", "formulation_id"),
        ("package_skus.csv", "package_sku_id"),
        ("company_entities.csv", "company_id"),
        ("aliases.csv", "alias_id"),
        ("source_evidence.csv", "evidence_id"),
    ]:
        df = pd.read_csv(KNOWLEDGE / "canonical" / filename, dtype=str).fillna("")
        assert df[key].is_unique, filename


def test_alias_foreign_keys_are_valid():
    aliases = set(pd.read_csv(KNOWLEDGE / "canonical/aliases.csv", dtype=str)["alias_id"])
    evidence = set(pd.read_csv(KNOWLEDGE / "canonical/source_evidence.csv", dtype=str)["evidence_id"])
    links = pd.read_csv(KNOWLEDGE / "canonical/alias_evidence_links.csv", dtype=str).fillna("")
    assert set(links["alias_id"]) <= aliases
    assert set(links["evidence_id"]) <= evidence


def test_brand_family_product_formulation_and_package_are_separate():
    products = pd.read_csv(KNOWLEDGE / "canonical/brand_products.csv", dtype=str).fillna("")
    skus = pd.read_csv(KNOWLEDGE / "canonical/package_skus.csv", dtype=str).fillna("")
    formulations = pd.read_csv(KNOWLEDGE / "canonical/clinical_formulations.csv", dtype=str).fillna("")
    assert products["brand_family_id"].ne(products["brand_product_id"]).all()
    assert products["formulation_id"].isin(set(formulations["formulation_id"])).all()
    assert skus["brand_product_id"].isin(set(products["brand_product_id"])).all()


def test_fdc_components_are_first_class_and_preserved():
    components = parse_composition("Amoxycillin  (500mg)", "Clavulanic Acid (125mg)")
    assert len(components) == 2
    assert "amoxycillin:500mg" in formulation_signature(components, "tablet")
    component_rows = pd.read_csv(KNOWLEDGE / "canonical/formulation_components.csv", dtype=str).fillna("")
    fdc = component_rows.groupby("formulation_id").size()
    assert (fdc > 1).any()


def test_company_relationship_role_is_not_overclaimed():
    relationships = pd.read_csv(KNOWLEDGE / "canonical/company_relationships.csv", dtype=str).fillna("")
    assert "SOURCE_UNSPECIFIED_COMPANY" in set(relationships["relationship_role"])
    assert "MANUFACTURER" not in set(relationships["relationship_role"])


def test_missing_field_is_not_conflict_and_quarantine_promotion_is_blocked():
    assert compare_field("", "amoxicillin") == ComparisonStatus.NOT_COMPARABLE
    assert compare_field(None, "amoxicillin") == ComparisonStatus.NOT_COMPARABLE
    assert compare_field("a", "b") == ComparisonStatus.CONFLICT
    assert not can_promote(KGState.CANDIDATE_QUARANTINE, {Authority.OFFICIAL_INDIA}, {"L1"})


def test_context_and_global_sources_cannot_establish_brand_identity():
    assert not source_can_establish_brand_identity(Authority.CONTEXT_ONLY)
    assert not source_can_establish_brand_identity(Authority.GLOBAL_ENRICHMENT)
    assert source_can_establish_brand_identity(Authority.OFFICIAL_INDIA)


def test_pubchem_client_rejects_broad_terms_and_brand_identity(tmp_path: Path):
    client = PubChemClient(tmp_path)
    with pytest.raises(ValueError):
        client.get_compound_by_name("common pain fever tablet brand")


def test_source_provenance_required_for_open_claims():
    evidence = pd.read_csv(KNOWLEDGE / "canonical/source_evidence.csv", dtype=str).fillna("")
    products = pd.read_csv(KNOWLEDGE / "canonical/brand_products.csv", dtype=str).fillna("")
    assert len(evidence) == len(products)
    assert set(products["brand_product_id"]) <= set(evidence["entity_id"])
    assert evidence["raw_sha256"].str.len().eq(64).all()


def test_no_prescription_inputs_used_for_stage1b():
    manifest = pd.read_csv(KNOWLEDGE / "provenance/build_inputs_manifest.csv", dtype=str).fillna("")
    assert manifest["input_path"].str.startswith("knowledge/raw/").all()
    forbidden = ["ground_truths_json", "layer_a_medication_mentions", "prescription_pipeline_jbhi_ieee/data"]
    assert not manifest["input_path"].str.contains("|".join(forbidden), regex=True).any()


def test_mims_scraper_absent_and_disabled():
    registry = pd.read_csv(KNOWLEDGE / "provenance/source_registry.csv", dtype=str).fillna("")
    mims = registry.loc[registry["source_id"] == "MIMS_CIMS_INDIA"].iloc[0]
    assert mims["access_method"] == "PERMISSION_REQUIRED; DO_NOT_AUTOMATE"
    assert mims["automated_query_allowed"] == "false"
    assert not list(ROOT.rglob("*mims*scraper*"))


def test_sqlite_views_exist_and_are_queryable():
    conn = sqlite3.connect(KNOWLEDGE / "medication_knowledge.sqlite")
    try:
        views = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view' AND name LIKE 'v_%'"
            ).fetchall()
        }
        assert {
            "v_brand_search",
            "v_ingredient_search",
            "v_formulation_search",
            "v_fdc_search",
            "v_source_evidence",
            "v_candidate_quarantine",
            "v_context_drug_links",
        } <= views
        assert conn.execute("SELECT COUNT(*) FROM v_candidate_quarantine").fetchone()[0] == 253973
    finally:
        conn.close()


def test_parser_is_deterministic_for_brand_family_and_composition():
    assert extract_brand_family("Augmentin 625 Duo Tablet") == extract_brand_family("Augmentin 625 Duo Tablet")
    first = parse_composition("Ambroxol (30mg/5ml)", "Levosalbutamol (1mg/5ml)")
    second = parse_composition("Ambroxol (30mg/5ml)", "Levosalbutamol (1mg/5ml)")
    assert first == second

