from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def document_uid(collection_date: str, source_document_id: str, source_json_relpath: str, source_json_sha256: str) -> str:
    return "DOC_" + stable_hash(collection_date, source_document_id, source_json_relpath, source_json_sha256, length=20)


def page_uid(document_uid_value: str, page_number: int | None, source_image_identity: str | None) -> str:
    return "PAGE_" + stable_hash(document_uid_value, page_number or "", source_image_identity or "", length=20)


def mention_id(document_uid_value: str, source_json_path: str, source_object_index: int) -> str:
    return "MENT_" + stable_hash(document_uid_value, source_json_path, source_object_index, length=24)


def context_bundle_id(document_uid_value: str, page_number: int | None, context_scope: str = "document") -> str:
    return "CTX_" + stable_hash(document_uid_value, page_number or "", context_scope, length=20)


def task_id(stage: str, stable_input_id: str) -> str:
    return "TASK_" + stable_hash(stage, stable_input_id, length=24)


def lexical_surface(value: str | None) -> str | None:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if text else ""
