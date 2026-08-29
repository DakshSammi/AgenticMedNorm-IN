from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class KGState(StrEnum):
    AUTHORITATIVE_CANONICAL = "AUTHORITATIVE_CANONICAL"
    SUPPORTED_CANONICAL = "SUPPORTED_CANONICAL"
    CANDIDATE_QUARANTINE = "CANDIDATE_QUARANTINE"


class Authority(StrEnum):
    OFFICIAL_INDIA = "OFFICIAL_INDIA"
    AUTHENTICATED_API = "AUTHENTICATED_API"
    OPEN_DERIVATIVE = "OPEN_DERIVATIVE"
    CONTEXT_ONLY = "CONTEXT_ONLY"
    GLOBAL_ENRICHMENT = "GLOBAL_ENRICHMENT"


class ComparisonStatus(StrEnum):
    MATCH = "MATCH"
    CONFLICT = "CONFLICT"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class SourceTier(StrEnum):
    A_OFFICIAL_INDIA = "A_OFFICIAL_INDIA"
    B_AUTHENTICATED_API = "B_AUTHENTICATED_API"
    C_OPEN_HIGH_RECALL = "C_OPEN_HIGH_RECALL"
    D_CONTEXT_ONLY = "D_CONTEXT_ONLY"
    E_GLOBAL_ENRICHMENT = "E_GLOBAL_ENRICHMENT"
    PROHIBITED = "PROHIBITED"


@dataclass(frozen=True)
class SourceEvidence:
    evidence_id: str
    source_id: str
    entity_type: str
    entity_id: str
    field_name: str
    raw_value: str
    raw_snapshot_path: str
    raw_sha256: str
    parser_version: str
    kg_state: str
    authority: str
    notes: str = ""


@dataclass(frozen=True)
class Ingredient:
    ingredient_id: str
    canonical_name: str
    normalized_name: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class BrandFamily:
    brand_family_id: str
    canonical_name: str
    normalized_name: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class ClinicalFormulation:
    formulation_id: str
    dosage_form: str
    route: str
    release_modifier: str
    normalized_component_signature: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class FormulationComponent:
    formulation_component_id: str
    formulation_id: str
    ingredient_id: str
    component_order: int
    raw_component_text: str
    strength_text: str
    normalized_strength: str


@dataclass(frozen=True)
class BrandProduct:
    brand_product_id: str
    brand_family_id: str
    formulation_id: str
    source_product_id: str
    raw_brand_name: str
    normalized_brand_name: str
    medicine_type: str
    discontinued: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class PackageSKU:
    package_sku_id: str
    brand_product_id: str
    pack_size: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class PriceObservation:
    price_observation_id: str
    package_sku_id: str
    price: str
    currency: str
    source_id: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class CompanyEntity:
    company_id: str
    canonical_name: str
    normalized_name: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class CompanyRelationship:
    company_relationship_id: str
    company_id: str
    related_entity_type: str
    related_entity_id: str
    relationship_role: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class Alias:
    alias_id: str
    alias_text: str
    normalized_alias: str
    alias_type: str
    kg_state: str
    authority: str


@dataclass(frozen=True)
class AliasEvidenceLink:
    alias_evidence_link_id: str
    alias_id: str
    evidence_id: str
    linked_entity_type: str
    linked_entity_id: str


@dataclass(frozen=True)
class OntologyCrosswalk:
    crosswalk_id: str
    local_entity_type: str
    local_entity_id: str
    external_system: str
    external_id: str
    match_status: str
    kg_state: str
    authority: str
    evidence_id: str


def row_dict(model: Any) -> dict[str, Any]:
    return asdict(model)


def compare_field(left: str | None, right: str | None) -> ComparisonStatus:
    if left in (None, "") or right in (None, ""):
        return ComparisonStatus.NOT_COMPARABLE
    return ComparisonStatus.MATCH if left == right else ComparisonStatus.CONFLICT


def can_promote(
    current_state: KGState,
    supporting_authorities: set[Authority],
    evidence_levels: set[str],
) -> bool:
    if current_state != KGState.CANDIDATE_QUARANTINE:
        return True
    official = Authority.OFFICIAL_INDIA in supporting_authorities
    authenticated = Authority.AUTHENTICATED_API in supporting_authorities
    return official and authenticated and {"L1", "L2", "L3"} <= evidence_levels


def source_can_establish_brand_identity(authority: Authority) -> bool:
    return authority in {Authority.OFFICIAL_INDIA, Authority.AUTHENTICATED_API}

