from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.utils.stable_ids import stable_hash


PARSER_VERSION = "stage1b_v0.1"

DOSAGE_WORDS = {
    "tablet",
    "tab",
    "capsule",
    "cap",
    "syrup",
    "suspension",
    "injection",
    "infusion",
    "drop",
    "drops",
    "cream",
    "gel",
    "ointment",
    "solution",
    "powder",
    "lotion",
    "spray",
    "soap",
    "shampoo",
    "respules",
    "respule",
    "strip",
    "bottle",
    "vial",
    "ampoule",
    "sachet",
}


@dataclass(frozen=True)
class ParsedComponent:
    raw_component_text: str
    ingredient_name: str
    normalized_ingredient: str
    strength_text: str
    normalized_strength: str


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^a-z0-9%./+ -]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def display_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def extract_brand_family(raw_brand_name: object) -> tuple[str, str]:
    raw = display_text(raw_brand_name)
    normalized = normalize_text(raw)
    tokens = normalized.split()
    family_tokens: list[str] = []
    for token in tokens:
        stripped = re.sub(r"^[0-9.]+|[0-9.]+$", "", token)
        if not stripped:
            break
        if stripped in DOSAGE_WORDS:
            break
        if re.fullmatch(r"\d+(\.\d+)?(%|mg|mcg|g|ml|iu|kg|l).*", token):
            break
        family_tokens.append(token)
    family_norm = " ".join(family_tokens) or normalized
    family_display = " ".join(raw.split()[: len(family_tokens)]) if family_tokens else raw
    return family_display, family_norm


def infer_dosage_form(raw_brand_name: object, pack_size: object) -> str:
    text = f"{normalize_text(raw_brand_name)} {normalize_text(pack_size)}"
    for word in sorted(DOSAGE_WORDS):
        if re.search(rf"\b{re.escape(word)}s?\b", text):
            return word
    return ""


def parse_component(raw_component: object) -> ParsedComponent | None:
    raw = display_text(raw_component)
    if not raw:
        return None
    match = re.match(r"^(?P<name>.*?)(?:\s*\((?P<strength>[^()]*)\))?\s*$", raw)
    if not match:
        return ParsedComponent(raw, raw, normalize_text(raw), "", "")
    name = display_text(match.group("name"))
    strength = display_text(match.group("strength"))
    if not name:
        return None
    return ParsedComponent(raw, name, normalize_text(name), strength, normalize_strength(strength))


def parse_composition(*raw_components: object) -> list[ParsedComponent]:
    parsed: list[ParsedComponent] = []
    for raw_component in raw_components:
        raw = display_text(raw_component)
        if not raw:
            continue
        pieces = re.split(r"\s+\+\s+|;\s*", raw)
        for piece in pieces:
            component = parse_component(piece)
            if component:
                parsed.append(component)
    return parsed


def normalize_strength(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s+", "", text)
    return text


def formulation_signature(components: list[ParsedComponent], dosage_form: str = "") -> str:
    parts = [
        f"{component.normalized_ingredient}:{component.normalized_strength}"
        for component in sorted(components, key=lambda item: (item.normalized_ingredient, item.normalized_strength))
    ]
    return "|".join(parts + ([f"form:{dosage_form}"] if dosage_form else []))


def stable_prefixed_id(prefix: str, *parts: object, length: int = 20) -> str:
    return f"{prefix}_{stable_hash(*parts, length=length)}"

