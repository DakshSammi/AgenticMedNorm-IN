from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process

from src.knowledge.parsers import normalize_text


STAGE2C_VERSION = "stage2c_candidate_retrieval_v0.1"
BRANCHES = ["R1_EXACT_FUZZY", "R2_BM25", "R3_BIOMEDICAL_DENSE", "R4_RXNORM", "R5_INDIA_KB"]
STATUS_SUCCESS = "SUCCESS"
STATUS_EMPTY = "EMPTY"
STATUS_FAILED = "FAILED"
STATUS_UNAVAILABLE = "UNAVAILABLE"
TOP_N = 25
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BIOMEDICAL_DENSE_FALLBACKS = [
    SAPBERT_MODEL,
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
]


@dataclass(frozen=True)
class RetrievalHit:
    candidate_id: str
    candidate_name: str
    candidate_type: str
    score: float
    score_semantics: str
    matched_field: str
    matched_alias: str
    source_state: str
    authority: str
    provenance_evidence_ids: str
    metadata: dict[str, Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", normalize_text(text)) if len(tok) > 1]


def _clean_query_surface(text: str) -> str:
    norm = normalize_text(text)
    norm = re.sub(r"\b(tab|tabs|tablet|cap|caps|capsule|syp|syr|syrup|inj|injection|drop|drops|cream|oint|ointment)\b\.?", " ", norm)
    norm = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?|%)\b", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm or normalize_text(text)


def _candidate_sort_key(hit: RetrievalHit) -> tuple[float, str]:
    return (-hit.score, hit.candidate_id)


class BM25Index:
    def __init__(self, docs: list[dict[str, str]], version: str) -> None:
        self.docs = docs
        self.version = version
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}

    @classmethod
    def build(cls, docs: list[dict[str, str]], version: str) -> "BM25Index":
        index = cls(docs, version)
        doc_freq: Counter[str] = Counter()
        tmp_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_i, doc in enumerate(docs):
            counts = Counter(_tokenize(doc.get("search_text", "")))
            index.doc_len.append(sum(counts.values()) or 1)
            for term, tf in counts.items():
                doc_freq[term] += 1
                tmp_postings[term].append((doc_i, tf))
        n_docs = max(len(docs), 1)
        index.avgdl = sum(index.doc_len) / n_docs
        index.postings = dict(tmp_postings)
        index.idf = {term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}
        return index

    def search(self, query: str, top_n: int = TOP_N) -> list[tuple[int, float]]:
        terms = _tokenize(query)
        if not terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_i, tf in self.postings.get(term, []):
                denom = tf + k1 * (1 - b + b * self.doc_len[doc_i] / max(self.avgdl, 1e-9))
                scores[doc_i] += idf * (tf * (k1 + 1)) / denom
        return sorted(scores.items(), key=lambda item: (-item[1], self.docs[item[0]].get("candidate_id", "")))[:top_n]


class R1ExactFuzzy:
    def __init__(self, root: Path) -> None:
        canonical = root / "knowledge/canonical"
        aliases = _read_csv(canonical / "aliases.csv")
        links = _read_csv(canonical / "alias_evidence_links.csv")
        products = _read_csv(canonical / "brand_products.csv")
        families = _read_csv(canonical / "brand_families.csv")
        ingredients = _read_csv(canonical / "ingredients.csv")
        formulations = _read_csv(canonical / "clinical_formulations.csv")

        self.entries: list[RetrievalHit] = []
        self.exact: dict[str, list[RetrievalHit]] = defaultdict(list)
        self.choice_to_hits: dict[str, list[RetrievalHit]] = defaultdict(list)
        self.choices_by_initial: dict[str, set[str]] = defaultdict(set)
        self.choices_by_token: dict[str, set[str]] = defaultdict(set)

        product_name = dict(zip(products["brand_product_id"], products["raw_brand_name"], strict=False))
        family_name = dict(zip(families["brand_family_id"], families["canonical_name"], strict=False))
        ingredient_name = dict(zip(ingredients["ingredient_id"], ingredients["canonical_name"], strict=False))

        def add(candidate_id: str, name: str, candidate_type: str, field: str, alias: str, state: str, authority: str, evidence: str = "") -> None:
            text = normalize_text(alias)
            if not text or not candidate_id:
                return
            hit = RetrievalHit(
                candidate_id=candidate_id,
                candidate_name=name or alias,
                candidate_type=candidate_type,
                score=1.0,
                score_semantics="normalized_exact_or_rapidfuzz_wratio",
                matched_field=field,
                matched_alias=alias,
                source_state=state or "UNKNOWN",
                authority=authority or "",
                provenance_evidence_ids=evidence,
                metadata={},
            )
            self.entries.append(hit)
            self.exact[text].append(hit)
            self.choice_to_hits[text].append(hit)

        for row in products.itertuples(index=False):
            add(row.brand_product_id, row.raw_brand_name, "BrandProduct", "brand_product.raw_brand_name", row.raw_brand_name, row.kg_state, row.authority)
            add(row.brand_product_id, row.raw_brand_name, "BrandProduct", "brand_product.normalized_brand_name", row.normalized_brand_name, row.kg_state, row.authority)
        for row in families.itertuples(index=False):
            add(row.brand_family_id, row.canonical_name, "BrandFamily", "brand_family.canonical_name", row.canonical_name, row.kg_state, row.authority)
        for row in ingredients.itertuples(index=False):
            add(row.ingredient_id, row.canonical_name, "Ingredient", "ingredient.canonical_name", row.canonical_name, row.kg_state, row.authority)
        for row in formulations.itertuples(index=False):
            display = row.normalized_component_signature
            add(row.formulation_id, display, "ClinicalFormulation", "formulation.normalized_component_signature", display, row.kg_state, row.authority)

        link_map: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for row in links.itertuples(index=False):
            link_map[row.alias_id].append((row.linked_entity_id, row.linked_entity_type, row.evidence_id))
        alias_lookup = {
            "BrandProduct": product_name,
            "BrandFamily": family_name,
            "Ingredient": ingredient_name,
        }
        alias_meta = dict(zip(aliases["alias_id"], aliases[["alias_text", "kg_state", "authority"]].to_dict("records"), strict=False))
        for alias_id, linked in link_map.items():
            meta = alias_meta.get(alias_id)
            if not meta:
                continue
            for candidate_id, candidate_type, evidence_id in linked[:4]:
                name = alias_lookup.get(candidate_type, {}).get(candidate_id, meta["alias_text"])
                add(candidate_id, name, candidate_type, f"alias.{meta['alias_text']}", meta["alias_text"], meta["kg_state"], meta["authority"], evidence_id)

        self.choices = list(self.choice_to_hits)
        for choice in self.choices:
            if choice:
                self.choices_by_initial[choice[0]].add(choice)
            for token in _tokenize(choice)[:4]:
                self.choices_by_token[token].add(choice)

    def _fuzzy_choices(self, query: str) -> list[str]:
        token_sets = [self.choices_by_token.get(token, set()) for token in _tokenize(query)]
        token_sets = [items for items in token_sets if items]
        if token_sets:
            choices = set().union(*token_sets)
        else:
            choices = self.choices_by_initial.get(query[:1], set())
        if not choices:
            choices = self.choices_by_initial.get(query[:1], set())
        return sorted(choices)[:50000]

    def search(self, surface: str, top_n: int = TOP_N) -> list[RetrievalHit]:
        query = normalize_text(surface)
        cleaned = _clean_query_surface(surface)
        candidates: list[RetrievalHit] = []
        for q in dict.fromkeys([query, cleaned]):
            for hit in self.exact.get(q, []):
                candidates.append(hit)
        seen: set[str] = set()
        exact_hits = []
        for hit in candidates:
            key = f"{hit.candidate_id}|{hit.matched_alias}|exact"
            if key not in seen:
                seen.add(key)
                exact_hits.append(hit)
        if len(exact_hits) >= top_n:
            return exact_hits[:top_n]

        fuzzy_hits: list[RetrievalHit] = []
        for q in dict.fromkeys([query, cleaned]):
            if not q:
                continue
            choices = self._fuzzy_choices(q)
            for choice, score, _ in process.extract(q, choices, scorer=fuzz.WRatio, limit=top_n * 3, score_cutoff=82):
                for base in self.choice_to_hits[choice][:3]:
                    key = f"{base.candidate_id}|{choice}|fuzzy"
                    if key in seen:
                        continue
                    seen.add(key)
                    fuzzy_hits.append(
                        RetrievalHit(
                            candidate_id=base.candidate_id,
                            candidate_name=base.candidate_name,
                            candidate_type=base.candidate_type,
                            score=float(score) / 100.0,
                            score_semantics="rapidfuzz_wratio_0_to_1",
                            matched_field=base.matched_field,
                            matched_alias=base.matched_alias,
                            source_state=base.source_state,
                            authority=base.authority,
                            provenance_evidence_ids=base.provenance_evidence_ids,
                            metadata={},
                        )
                    )
        return sorted(exact_hits + fuzzy_hits, key=_candidate_sort_key)[:top_n]


class R2BM25:
    def __init__(self, root: Path, cache_dir: Path) -> None:
        self.cache_path = cache_dir / "bm25_index.pkl"
        source_path = root / "knowledge/canonical/retrieval_documents.csv"
        source_hash = _sha256_file(source_path)
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("source_hash") == source_hash and cached.get("version") == STAGE2C_VERSION:
                self.index: BM25Index = cached["index"]
                return
        docs = []
        docs_df = _read_csv(source_path)
        for row in docs_df.itertuples(index=False):
            docs.append(
                {
                    "candidate_id": row.brand_product_id,
                    "candidate_name": row.brand_text,
                    "candidate_type": "BrandProduct",
                    "source_state": row.kg_state,
                    "search_text": " ".join([row.brand_text, row.alias_text, row.ingredient_text, row.formulation_text, row.manufacturer_text, row.search_document]),
                    "matched_alias": row.search_document,
                    "matched_field": "retrieval_documents.search_document",
                    "authority": "OPEN_DERIVATIVE",
                }
            )
        self.index = BM25Index.build(docs, STAGE2C_VERSION)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump({"version": STAGE2C_VERSION, "source_hash": source_hash, "created_at": _now_iso(), "index": self.index}, f)

    def search(self, surface: str, top_n: int = TOP_N) -> list[RetrievalHit]:
        hits = []
        for doc_i, score in self.index.search(surface, top_n=top_n):
            doc = self.index.docs[doc_i]
            hits.append(
                RetrievalHit(
                    candidate_id=doc["candidate_id"],
                    candidate_name=doc["candidate_name"],
                    candidate_type=doc["candidate_type"],
                    score=float(score),
                    score_semantics="bm25_okapi_unbounded",
                    matched_field=doc["matched_field"],
                    matched_alias=doc["matched_alias"],
                    source_state=doc["source_state"],
                    authority=doc["authority"],
                    provenance_evidence_ids="",
                    metadata={},
                )
            )
        return hits


class R3BiomedicalDense:
    def __init__(self, root: Path, cache_dir: Path) -> None:
        self.root = root
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = cache_dir / "dense_metadata.json"
        self.docs_path = cache_dir / "dense_documents.csv"
        self.embeddings_path = cache_dir / "dense_embeddings.npz"
        self.faiss_index_path = cache_dir / "dense_faiss.index"
        self.surface_results_path = cache_dir / "dense_surface_top25.csv"
        self.available = False
        self.unavailable_reason = ""
        self.model_name = SAPBERT_MODEL
        self.model_version_or_commit = ""
        self.pooling = "mean_pooling_attention_mask"
        self.embedding_dim = 0
        self.index_method = "faiss_IndexFlatIP_l2_normalized"
        self.device = "cpu"
        self.docs: list[dict[str, str]] = []
        self.embeddings: np.ndarray | None = None
        self.index: Any | None = None
        self.tokenizer = None
        self.model = None
        self._load_or_prepare()

    def _dense_docs(self) -> list[dict[str, str]]:
        docs: list[dict[str, str]] = []
        ingredients = _read_csv(self.root / "knowledge/canonical/ingredients.csv")
        for row in ingredients.itertuples(index=False):
            docs.append(
                {
                    "candidate_id": row.ingredient_id,
                    "candidate_name": row.canonical_name,
                    "candidate_type": "Ingredient",
                    "source_state": row.kg_state,
                    "authority": row.authority,
                    "evidence_id": "",
                    "search_text": row.canonical_name,
                    "matched_field": "ingredient.canonical_name",
                }
            )
        nlem = _read_csv(self.root / "knowledge/canonical/nlem_entries.csv")
        for row in nlem.itertuples(index=False):
            text = " ".join([row.ingredient, row.strength, row.dosage_form, row.section_category])
            docs.append(
                {
                    "candidate_id": row.nlem_entry_id,
                    "candidate_name": row.ingredient,
                    "candidate_type": "NLEMEntry",
                    "source_state": "AUTHORITATIVE_NLEM_CONTEXT",
                    "authority": "NLEM_2022",
                    "evidence_id": row.evidence_id,
                    "search_text": text,
                    "matched_field": "nlem.ingredient_strength_form",
                }
            )
        cdsco = _read_csv(self.root / "knowledge/canonical/cdsco_structured_records.csv")
        for row in cdsco.itertuples(index=False):
            docs.append(
                {
                    "candidate_id": row.cdsco_record_id,
                    "candidate_name": row.drug_name,
                    "candidate_type": "CDSCORecord",
                    "source_state": "AUTHORITATIVE_CDSCO_CONTEXT",
                    "authority": row.source_id,
                    "evidence_id": row.evidence_id,
                    "search_text": " ".join([row.drug_name, row.source_document_title, row.applicant_or_company]),
                    "matched_field": "cdsco.drug_name",
                }
            )
        return docs

    def _write_metadata(self) -> None:
        payload = {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "model_name": self.model_name,
            "model_type": "true_biomedical_transformer_encoder" if self.available else "UNAVAILABLE",
            "model_version_or_commit": self.model_version_or_commit or "unknown",
            "pooling_method": self.pooling,
            "embedding_dimension": self.embedding_dim,
            "index_method": self.index_method,
            "device": self.device,
            "char_ngram_used": False,
            "documents_path": str(self.docs_path),
            "embeddings_path": str(self.embeddings_path),
            "faiss_index_path": str(self.faiss_index_path),
            "surface_results_path": str(self.surface_results_path),
            "created_or_checked_at": _now_iso(),
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load_or_prepare(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            import faiss
        except Exception as exc:
            self.unavailable_reason = f"transformers/torch/faiss import failed: {type(exc).__name__}: {exc}"
            self._write_metadata()
            return
        allow_download = os.environ.get("STAGE2C_ALLOW_DENSE_DOWNLOAD", "0") == "1"
        load_errors: list[str] = []
        loaded = False
        for model_name in BIOMEDICAL_DENSE_FALLBACKS:
            self.model_name = model_name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir / "hf"),
                    local_files_only=not allow_download,
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir / "hf"),
                    local_files_only=not allow_download,
                )
                loaded = True
                break
            except Exception as exc:
                mode = "download_allowed" if allow_download else "local_files_only"
                load_errors.append(f"{model_name} ({mode}): {type(exc).__name__}: {exc}")
        if not loaded:
            self.unavailable_reason = "Biomedical dense encoder load/cache failed: " + " | ".join(load_errors)
            self.available = False
            self._write_metadata()
            return
        try:
            try:
                from huggingface_hub import model_info

                self.model_version_or_commit = model_info(self.model_name).sha or ""
            except Exception:
                self.model_version_or_commit = getattr(self.model.config, "_commit_hash", "") or ""
            self.device = "cuda" if torch.cuda.is_available() and os.environ.get("STAGE2C_DENSE_DEVICE", "auto") != "cpu" else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.docs = self._dense_docs()
            if self.docs_path.exists():
                cached_docs = _read_csv(self.docs_path).to_dict("records")
                if len(cached_docs) == len(self.docs):
                    self.docs = cached_docs
            else:
                pd.DataFrame(self.docs).to_csv(self.docs_path, index=False)
            if self.embeddings_path.exists():
                arr = np.load(self.embeddings_path)
                self.embeddings = arr["embeddings"]
                if self.embeddings.shape[0] != len(self.docs):
                    self.embeddings = self._encode([doc["search_text"] for doc in self.docs], batch_size=64, torch_module=torch)
                    np.savez_compressed(self.embeddings_path, embeddings=self.embeddings)
            else:
                texts = [doc["search_text"] for doc in self.docs]
                self.embeddings = self._encode(texts, batch_size=64, torch_module=torch)
                np.savez_compressed(self.embeddings_path, embeddings=self.embeddings)
            self.embedding_dim = int(self.embeddings.shape[1])
            if self.faiss_index_path.exists():
                self.index = faiss.read_index(str(self.faiss_index_path))
                if self.index.ntotal != len(self.docs):
                    self.index = None
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(np.ascontiguousarray(self.embeddings.astype("float32")))
                faiss.write_index(self.index, str(self.faiss_index_path))
            self.available = True
            self._write_metadata()
        except Exception as exc:
            mode = "download_allowed" if allow_download else "local_files_only"
            self.unavailable_reason = f"SapBERT load/cache failed ({mode}): {type(exc).__name__}: {exc}"
            self.available = False
            self._write_metadata()

    def _encode(self, texts: list[str], batch_size: int, torch_module: Any) -> np.ndarray:
        vectors: list[np.ndarray] = []
        with torch_module.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                arr = pooled.cpu().numpy().astype("float32")
                arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)
                vectors.append(arr)
        return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")

    def search(self, surface: str, top_n: int = TOP_N) -> list[RetrievalHit]:
        if not self.available or self.index is None:
            raise RuntimeError(self.unavailable_reason or "SapBERT unavailable")
        import torch

        query_vec = self._encode([surface], batch_size=1, torch_module=torch)
        scores, idxs = self.index.search(np.ascontiguousarray(query_vec.astype("float32")), top_n)
        hits = []
        for score, doc_i in zip(scores[0], idxs[0], strict=False):
            if int(doc_i) < 0:
                continue
            doc = self.docs[int(doc_i)]
            hits.append(
                RetrievalHit(
                    candidate_id=doc["candidate_id"],
                    candidate_name=doc["candidate_name"],
                    candidate_type=doc["candidate_type"],
                    score=float(score),
                    score_semantics="cosine_similarity_l2_normalized_sapbert_mean_pool",
                    matched_field=doc["matched_field"],
                    matched_alias=doc["search_text"],
                    source_state=doc["source_state"],
                    authority=doc["authority"],
                    provenance_evidence_ids=doc.get("evidence_id", ""),
                    metadata={},
                )
            )
        return hits


class R4RxNorm:
    def __init__(self, root: Path, cache_dir: Path) -> None:
        self.cache_dir = cache_dir / "rxnav"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.allow_network = os.environ.get("STAGE2C_ALLOW_RXNAV_NETWORK", "0") == "1"
        mappings = _read_csv(root / "knowledge/crosswalks/rxnorm_ingredient_mappings.csv")
        usable = mappings[mappings["rxcui"].astype(str).str.len() > 0].copy()
        usable = usable[usable["mapping_status"].isin(["EXACT", "NORMALIZED_SUPPORTED", "APPROXIMATE_REVIEW"])]
        self.local_records = usable.to_dict("records")
        self.local_exact: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.local_records:
            self.local_exact[normalize_text(row.get("ingredient_name", ""))].append(row)
            self.local_exact[normalize_text(row.get("rxnorm_name", ""))].append(row)

    def _cache_key(self, endpoint: str, params: dict[str, str]) -> Path:
        raw = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        return self.cache_dir / f"{hashlib.sha256(raw.encode()).hexdigest()}.json"

    def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        path = self._cache_key(endpoint, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = f"{RXNAV_BASE}/{endpoint}"
        response = self.session.get(url, params=params, timeout=15)
        payload = {"url": response.url, "status_code": response.status_code, "retrieved_at": _now_iso(), "json": response.json() if response.content else {}}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _properties(self, rxcui: str) -> dict[str, str]:
        try:
            payload = self._get(f"rxcui/{rxcui}/properties.json", {})
            return payload.get("json", {}).get("properties", {}) or {}
        except Exception:
            return {}

    def search(self, surface: str, top_n: int = TOP_N) -> list[RetrievalHit]:
        queries = list(dict.fromkeys([_clean_query_surface(surface), normalize_text(surface)]))
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for query in queries:
            if not query:
                continue
            local = self.local_exact.get(query, [])
            for row in local[:top_n]:
                candidate_id = f"RXCUI_{row['rxcui']}"
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                hits.append(
                    RetrievalHit(
                        candidate_id=candidate_id,
                        candidate_name=row.get("rxnorm_name", "") or row.get("ingredient_name", ""),
                        candidate_type="RxNormConcept",
                        score=1.0,
                        score_semantics="rxnorm_crosswalk_exact_normalized_lookup",
                        matched_field="rxnorm_ingredient_mappings.normalized_ingredient",
                        matched_alias=query,
                        source_state="RXNORM_CONCEPT",
                        authority="RxNorm/RxNav",
                        provenance_evidence_ids=row.get("evidence_id", "") or row.get("rxcui", ""),
                        metadata={"tty": row.get("tty", ""), "rxcui": row.get("rxcui", ""), "mapping_status": row.get("mapping_status", "")},
                    )
                )
            if hits:
                self._write_local_cache(query, hits, "local_exact")
                return hits[:top_n]
            if not self.allow_network:
                continue
            exact = self._get("rxcui.json", {"name": query, "search": "2"})
            ids = exact.get("json", {}).get("idGroup", {}).get("rxnormId", []) or []
            for rxcui in ids[:top_n]:
                props = self._properties(str(rxcui))
                candidate_id = f"RXCUI_{rxcui}"
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                hits.append(
                    RetrievalHit(
                        candidate_id=candidate_id,
                        candidate_name=props.get("name", query),
                        candidate_type="RxNormConcept",
                        score=1.0,
                        score_semantics="rxnav_exact_normalized_lookup",
                        matched_field="RxNav.rxcui.name",
                        matched_alias=query,
                        source_state="RXNORM_CONCEPT",
                        authority="RxNorm/RxNav",
                        provenance_evidence_ids=str(rxcui),
                        metadata={"tty": props.get("tty", ""), "rxcui": str(rxcui)},
                    )
                )
            if hits:
                break
        if hits:
            return hits[:top_n]

        query = queries[0] if queries else normalize_text(surface)
        local_choices = [row.get("ingredient_name", "") for row in self.local_records]
        for choice, score, idx in process.extract(query, local_choices, scorer=fuzz.WRatio, limit=top_n, score_cutoff=86):
            row = self.local_records[idx]
            candidate_id = f"RXCUI_{row['rxcui']}"
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            hits.append(
                RetrievalHit(
                    candidate_id=candidate_id,
                    candidate_name=row.get("rxnorm_name", "") or choice,
                    candidate_type="RxNormConcept",
                    score=float(score) / 100.0,
                    score_semantics="rxnorm_crosswalk_approximate_rapidfuzz_score_0_to_1",
                    matched_field="rxnorm_ingredient_mappings.ingredient_name",
                    matched_alias=query,
                    source_state="RXNORM_CONCEPT",
                    authority="RxNorm/RxNav",
                    provenance_evidence_ids=row.get("evidence_id", "") or row.get("rxcui", ""),
                    metadata={"tty": row.get("tty", ""), "rxcui": row.get("rxcui", ""), "mapping_status": row.get("mapping_status", "")},
                )
            )
        if hits:
            self._write_local_cache(query, hits, "local_approximate")
            return hits[:top_n]
        if not self.allow_network:
            self._write_local_cache(query, [], "network_disabled_empty")
            return []
        approx = self._get("approximateTerm.json", {"term": query, "maxEntries": str(top_n)})
        candidates = approx.get("json", {}).get("approximateGroup", {}).get("candidate", []) or []
        for item in candidates[:top_n]:
            rxcui = str(item.get("rxcui", ""))
            if not rxcui:
                continue
            props = self._properties(rxcui)
            candidate_id = f"RXCUI_{rxcui}"
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            score = float(item.get("score", 0) or 0) / 100.0
            hits.append(
                RetrievalHit(
                    candidate_id=candidate_id,
                    candidate_name=props.get("name", item.get("name", query)),
                    candidate_type="RxNormConcept",
                    score=score,
                    score_semantics="rxnav_approximate_score_0_to_1",
                    matched_field="RxNav.approximateTerm",
                    matched_alias=query,
                    source_state="RXNORM_CONCEPT",
                    authority="RxNorm/RxNav",
                    provenance_evidence_ids=rxcui,
                    metadata={"tty": props.get("tty", ""), "rxcui": rxcui},
                )
            )
        return hits[:top_n]

    def _write_local_cache(self, query: str, hits: list[RetrievalHit], method: str) -> None:
        path = self._cache_key("local_crosswalk.json", {"query": query, "method": method})
        if path.exists():
            return
        payload = {
            "method": method,
            "query": query,
            "retrieved_at": _now_iso(),
            "network_used": False,
            "json": {
                "candidates": [
                    {
                        "candidate_id": hit.candidate_id,
                        "candidate_name": hit.candidate_name,
                        "score": hit.score,
                        "metadata": hit.metadata,
                    }
                    for hit in hits
                ]
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class R5IndiaKB:
    def __init__(self, root: Path, r2: R2BM25) -> None:
        self.r2 = r2
        docs = _read_csv(root / "knowledge/canonical/retrieval_documents.csv")
        self.exact_brand: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in docs.itertuples(index=False):
            record = {
                "candidate_id": row.brand_product_id,
                "candidate_name": row.brand_text,
                "source_state": row.kg_state,
                "matched_alias": row.brand_text,
                "ingredient_text": row.ingredient_text,
            }
            self.exact_brand[normalize_text(row.brand_text)].append(record)
            family = normalize_text(str(row.brand_text).split()[0] if row.brand_text else "")
            if family:
                self.exact_brand[family].append(record)

    def search(self, surface: str, top_n: int = TOP_N) -> list[RetrievalHit]:
        query = _clean_query_surface(surface)
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for key in dict.fromkeys([normalize_text(surface), query]):
            for record in self.exact_brand.get(key, [])[:top_n]:
                seen.add(record["candidate_id"])
                hits.append(
                    RetrievalHit(
                        candidate_id=record["candidate_id"],
                        candidate_name=record["candidate_name"],
                        candidate_type="IndiaBrandProduct",
                        score=1.0,
                        score_semantics="india_kb_normalized_exact",
                        matched_field="india_kb.brand_text",
                        matched_alias=record["matched_alias"],
                        source_state=record["source_state"],
                        authority="OPEN_DERIVATIVE",
                        provenance_evidence_ids="",
                        metadata={},
                    )
                )
        if len(hits) < top_n:
            for hit in self.r2.search(surface, top_n=top_n):
                if hit.candidate_id in seen:
                    continue
                seen.add(hit.candidate_id)
                hits.append(
                    RetrievalHit(
                        candidate_id=hit.candidate_id,
                        candidate_name=hit.candidate_name,
                        candidate_type="IndiaBrandProduct",
                        score=hit.score,
                        score_semantics="india_kb_bm25_recall_score",
                        matched_field=hit.matched_field,
                        matched_alias=hit.matched_alias,
                        source_state=hit.source_state,
                        authority=hit.authority,
                        provenance_evidence_ids=hit.provenance_evidence_ids,
                        metadata={},
                    )
                )
                if len(hits) >= top_n:
                    break
        return hits[:top_n]


class Stage2CRetrievalAgent:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cache_dir = root / "knowledge/cache/stage2c"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.r1 = R1ExactFuzzy(root)
        self.r2 = R2BM25(root, self.cache_dir)
        self.r3 = R3BiomedicalDense(root, self.cache_dir)
        self.r4 = R4RxNorm(root, self.cache_dir)
        self.r5 = R5IndiaKB(root, self.r2)

    def _branch_search(self, branch: str, surface: str) -> tuple[str, list[RetrievalHit], float, str]:
        start = time.perf_counter()
        try:
            if branch == "R1_EXACT_FUZZY":
                hits = self.r1.search(surface)
            elif branch == "R2_BM25":
                hits = self.r2.search(surface)
            elif branch == "R3_BIOMEDICAL_DENSE":
                if not self.r3.available:
                    return STATUS_UNAVAILABLE, [], (time.perf_counter() - start) * 1000, self.r3.unavailable_reason
                hits = self.r3.search(surface)
            elif branch == "R4_RXNORM":
                hits = self.r4.search(surface)
            elif branch == "R5_INDIA_KB":
                hits = self.r5.search(surface)
            else:
                raise ValueError(f"unknown branch {branch}")
            status = STATUS_SUCCESS if hits else STATUS_EMPTY
            return status, hits, (time.perf_counter() - start) * 1000, ""
        except Exception as exc:
            return STATUS_FAILED, [], (time.perf_counter() - start) * 1000, f"{type(exc).__name__}: {exc}"

    def retrieve_surface(self, surface: str) -> dict[str, tuple[str, list[RetrievalHit], float, str]]:
        results: dict[str, tuple[str, list[RetrievalHit], float, str]] = {}
        with ThreadPoolExecutor(max_workers=len(BRANCHES)) as executor:
            future_map = {executor.submit(self._branch_search, branch, surface): branch for branch in BRANCHES}
            for future in as_completed(future_map):
                results[future_map[future]] = future.result()
        return results


def _trace_rows_for_mention(mention: pd.Series, branch_results: dict[str, tuple[str, list[RetrievalHit], float, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        status, hits, latency_ms, error = branch_results[branch]
        if not hits:
            rows.append(
                {
                    "mention_id": mention["mention_id"],
                    "raw_medication_text": mention["raw_medication_text"],
                    "branch": branch,
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
                    "status": status,
                    "error": error,
                }
            )
            continue
        for rank, hit in enumerate(hits, start=1):
            rows.append(
                {
                    "mention_id": mention["mention_id"],
                    "raw_medication_text": mention["raw_medication_text"],
                    "branch": branch,
                    "candidate_id": hit.candidate_id,
                    "candidate_name": hit.candidate_name,
                    "candidate_type": hit.candidate_type,
                    "rank": rank,
                    "score": round(hit.score, 6),
                    "score_semantics": hit.score_semantics,
                    "matched_field": hit.matched_field,
                    "matched_alias": hit.matched_alias,
                    "source_state": hit.source_state,
                    "authority": hit.authority,
                    "provenance_evidence_ids": hit.provenance_evidence_ids,
                    "latency_ms": round(latency_ms, 3),
                    "status": status,
                    "error": error,
                }
            )
    return rows


def _union_rows(trace: pd.DataFrame) -> pd.DataFrame:
    candidate_rows = trace[trace["candidate_id"].astype(str) != ""].copy()
    rows = []
    for mention_id, group in candidate_rows.groupby("mention_id", sort=False):
        raw_text = group["raw_medication_text"].iloc[0]
        for candidate_id, cgroup in group.groupby("candidate_id", sort=True):
            rows.append(
                {
                    "mention_id": mention_id,
                    "raw_medication_text": raw_text,
                    "candidate_id": candidate_id,
                    "candidate_name": cgroup["candidate_name"].iloc[0],
                    "candidate_type": cgroup["candidate_type"].iloc[0],
                    "branches_returned": "|".join(sorted(cgroup["branch"].unique())),
                    "branch_count": cgroup["branch"].nunique(),
                    "source_states": "|".join(sorted(set(str(x) for x in cgroup["source_state"] if str(x)))),
                    "is_ranked": "false",
                }
            )
    return pd.DataFrame(rows)


def _pairwise_overlap(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidate_trace = trace[trace["candidate_id"].astype(str) != ""]
    by_mention_branch = {
        (mention_id, branch): set(group["candidate_id"])
        for (mention_id, branch), group in candidate_trace.groupby(["mention_id", "branch"], sort=False)
    }
    mentions = sorted(trace["mention_id"].unique())
    for i, b1 in enumerate(BRANCHES):
        for b2 in BRANCHES[i + 1 :]:
            intersections = []
            unions = []
            for mention_id in mentions:
                s1 = by_mention_branch.get((mention_id, b1), set())
                s2 = by_mention_branch.get((mention_id, b2), set())
                intersections.append(len(s1 & s2))
                unions.append(len(s1 | s2))
            rows.append(
                {
                    "branch_a": b1,
                    "branch_b": b2,
                    "total_intersection": sum(intersections),
                    "total_union": sum(unions),
                    "jaccard": round(sum(intersections) / sum(unions), 6) if sum(unions) else 0.0,
                    "median_intersection_per_mention": median(intersections) if intersections else 0,
                }
            )
    return pd.DataFrame(rows)


def _status_summary(trace: pd.DataFrame, mention_count: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for branch in BRANCHES:
        branch_status = trace[trace["branch"] == branch].groupby("mention_id")["status"].first()
        counts = branch_status.value_counts().to_dict()
        success = counts.get(STATUS_SUCCESS, 0)
        empty = counts.get(STATUS_EMPTY, 0)
        failed = counts.get(STATUS_FAILED, 0)
        unavailable = counts.get(STATUS_UNAVAILABLE, 0)
        candidates_per_success = (
            trace[(trace["branch"] == branch) & (trace["candidate_id"].astype(str) != "")]
            .groupby("mention_id")["candidate_id"]
            .nunique()
            .tolist()
        )
        summary[branch] = {
            "success": int(success),
            "empty": int(empty),
            "failed": int(failed),
            "unavailable": int(unavailable),
            "candidate_return_rate": success / mention_count if mention_count else 0,
            "empty_rate": empty / mention_count if mention_count else 0,
            "failure_rate": failed / mention_count if mention_count else 0,
            "unavailable_rate": unavailable / mention_count if mention_count else 0,
            "median_candidates_when_successful": median(candidates_per_success) if candidates_per_success else 0,
            "median_latency_ms": float(pd.to_numeric(trace[trace["branch"] == branch].groupby("mention_id")["latency_ms"].first(), errors="coerce").median()),
        }
    return summary


def _sample_category(surface: str, surface_trace: pd.DataFrame) -> str:
    text = normalize_text(surface)
    if surface_trace.empty or not (surface_trace["candidate_id"].astype(str) != "").any():
        return "low_resource_unresolved"
    if re.search(r"\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%)", text):
        return "strength_bearing"
    if any(token in text for token in ["+", "/", "with"]) and len(text.split()) > 1:
        return "fdc_looking"
    if len(text) <= 5 or text.startswith(("tab ", "cap ", "syp ")):
        return "abbreviation"
    r1 = surface_trace[(surface_trace["branch"] == "R1_EXACT_FUZZY") & (surface_trace["candidate_id"].astype(str) != "")]
    if not r1.empty and r1["score"].astype(float).max() >= 0.999:
        if r1["candidate_type"].str.contains("Ingredient", na=False).any():
            return "generic_name"
        return "exact_looking_brand"
    if not r1.empty:
        return "misspelling_or_variant"
    return "generic_name"


def _diagnostic_sample(mentions: pd.DataFrame, trace: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    surface_to_trace = {surface: group for surface, group in trace.groupby("raw_medication_text", sort=False)}
    rows = []
    for surface in mentions["raw_medication_text"].drop_duplicates():
        st = surface_to_trace.get(surface, pd.DataFrame())
        row = {"raw_medication_text": surface, "sample_category": _sample_category(surface, st)}
        for branch in BRANCHES:
            bhits = st[(st["branch"] == branch) & (st["candidate_id"].astype(str) != "")].head(10)
            row[f"{branch}_top10"] = json.dumps(
                [
                    {
                        "candidate_id": item.candidate_id,
                        "candidate_name": item.candidate_name,
                        "rank": item.rank,
                        "score": item.score,
                        "source_state": item.source_state,
                    }
                    for item in bhits.itertuples(index=False)
                ],
                ensure_ascii=True,
            )
        row.update({"correct_candidate_present": "", "best_candidate_id": "", "branch_helpful": "", "notes": ""})
        rows.append(row)
    df = pd.DataFrame(rows)
    target_categories = [
        "exact_looking_brand",
        "misspelling_or_variant",
        "strength_bearing",
        "fdc_looking",
        "generic_name",
        "abbreviation",
        "low_resource_unresolved",
    ]
    samples = []
    per_category = max(1, 100 // len(target_categories))
    for category in target_categories:
        part = df[df["sample_category"] == category].head(per_category)
        samples.append(part)
    sample = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    if len(sample) < min(100, len(df)):
        remaining = df[~df["raw_medication_text"].isin(set(sample["raw_medication_text"]))].head(100 - len(sample))
        sample = pd.concat([sample, remaining], ignore_index=True)
    sample = sample.head(100)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    return sample


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    dense = summary["dense_model"]
    lines = [
        "# Stage 2C Candidate Retrieval Report",
        "",
        f"- stage2c_version: {STAGE2C_VERSION}",
        f"- generated_at: {summary['generated_at']}",
        f"- mentions_processed: {summary['mentions_processed']}",
        f"- unique_surfaces: {summary['unique_surfaces']}",
        f"- ready_for_ranking: {str(summary['READY_FOR_RANKING']).lower()}",
        "",
        "## Branch Return Rates",
    ]
    for branch, item in summary["branches"].items():
        lines.append(
            f"- {branch}: return={item['candidate_return_rate']:.3f}, empty={item['empty_rate']:.3f}, "
            f"failed={item['failure_rate']:.3f}, unavailable={item['unavailable_rate']:.3f}, "
            f"median_candidates={item['median_candidates_when_successful']}, median_latency_ms={item['median_latency_ms']:.1f}"
        )
    lines.extend(
        [
            "",
            "## Dense Retrieval",
            f"- model_name: {dense.get('model_name', '')}",
            f"- available: {dense.get('available', False)}",
            f"- model_type: {dense.get('model_type', '')}",
            f"- pooling_method: {dense.get('pooling_method', '')}",
            f"- embedding_dimension: {dense.get('embedding_dimension', 0)}",
            f"- index_method: {dense.get('index_method', '')}",
            f"- unavailable_reason: {dense.get('unavailable_reason', '')}",
            "",
            "## Guardrails",
            "- No candidate ranking, evidence assessment, or verification is implemented in Stage 2C.",
            "- No true Recall@K, MRR, or accuracy is reported because the 1,098 mentions do not have clinician semantic gold.",
            "- The true union table is an unranked set-union of branch candidate IDs.",
            "- Prescription medication strings are used only as retrieval inputs, not for KB expansion or source acquisition.",
            "",
            "## Outputs",
        ]
    )
    for key, value in summary["paths"].items():
        lines.append(f"- {key}: {value}")
    if summary["unresolved_technical_issues"]:
        lines.extend(["", "## Unresolved Technical Issues"])
        for issue in summary["unresolved_technical_issues"]:
            lines.append(f"- {issue}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _kb_input_hashes(root: Path) -> dict[str, str]:
    rels = [
        "knowledge/canonical/retrieval_documents.csv",
        "knowledge/canonical/aliases.csv",
        "knowledge/canonical/alias_evidence_links.csv",
        "knowledge/canonical/brand_products.csv",
        "knowledge/canonical/brand_families.csv",
        "knowledge/canonical/ingredients.csv",
        "knowledge/canonical/clinical_formulations.csv",
        "knowledge/canonical/nlem_entries.csv",
        "knowledge/canonical/cdsco_structured_records.csv",
        "knowledge/crosswalks/rxnorm_ingredient_mappings.csv",
        "knowledge/provenance/open_indian_dataset_freeze.json",
    ]
    return {rel: _sha256_file(root / rel) for rel in rels if (root / rel).exists()}


def run_stage2c(root: Path | None = None) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    derived_dir = root / "derived/retrieval"
    review_dir = root / "review"
    report_dir = root / "rebuild/reports"
    derived_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    hashes_before = _kb_input_hashes(root)
    mentions = _read_csv(root / "derived/layer_a_medication_mentions.csv")
    mentions = mentions[mentions["raw_medication_text"].astype(str).str.len() > 0].copy()
    surfaces = mentions["raw_medication_text"].drop_duplicates().tolist()

    agent = Stage2CRetrievalAgent(root)
    surface_results: dict[str, dict[str, tuple[str, list[RetrievalHit], float, str]]] = {}
    for surface in surfaces:
        if len(surface_results) and len(surface_results) % 50 == 0:
            print(f"stage2c retrieval surfaces processed: {len(surface_results)}/{len(surfaces)}", flush=True)
        surface_results[surface] = agent.retrieve_surface(surface)

    trace_rows: list[dict[str, Any]] = []
    for _, mention in mentions.iterrows():
        trace_rows.extend(_trace_rows_for_mention(mention, surface_results[mention["raw_medication_text"]]))
    trace = pd.DataFrame(trace_rows)
    trace_path = derived_dir / "stage2c_branch_traces.csv"
    trace.to_csv(trace_path, index=False)

    union = _union_rows(trace)
    union_path = derived_dir / "stage2c_candidate_union.csv"
    union.to_csv(union_path, index=False)

    status_path = derived_dir / "stage2c_branch_status.csv"
    status = trace.groupby(["mention_id", "raw_medication_text", "branch"], as_index=False).agg(status=("status", "first"), latency_ms=("latency_ms", "first"), candidates=("candidate_id", lambda s: int((s.astype(str) != "").sum())))
    status.to_csv(status_path, index=False)

    overlap = _pairwise_overlap(trace)
    overlap_path = derived_dir / "stage2c_pairwise_overlap.csv"
    overlap.to_csv(overlap_path, index=False)

    sample_path = review_dir / "retrieval_candidate_inspection_100.csv"
    sample = _diagnostic_sample(mentions, trace, sample_path)

    hashes_after = _kb_input_hashes(root)
    hash_path = derived_dir / "stage2c_kb_input_hashes.json"
    hash_path.write_text(json.dumps({"before": hashes_before, "after": hashes_after, "unchanged": hashes_before == hashes_after}, indent=2, sort_keys=True), encoding="utf-8")

    dense_meta_path = root / "knowledge/cache/stage2c/dense_metadata.json"
    dense_meta = json.loads(dense_meta_path.read_text(encoding="utf-8")) if dense_meta_path.exists() else {}
    branch_summary = _status_summary(trace, len(mentions))
    union_sizes = union.groupby("mention_id")["candidate_id"].nunique().tolist() if not union.empty else []
    source_states = trace[trace["candidate_id"].astype(str) != ""]["source_state"].value_counts().to_dict()
    state_total = sum(source_states.values()) or 1
    candidate_state_proportions = {state: count / state_total for state, count in source_states.items()}
    unresolved: list[str] = []
    if not dense_meta.get("available", False):
        unresolved.append(f"R3 SapBERT unavailable: {dense_meta.get('unavailable_reason', 'unknown')}")
    if branch_summary["R4_RXNORM"]["failure_rate"] > 0:
        unresolved.append("R4 RxNav had failed branch calls; inspect trace error column and cache responses.")

    ready = (
        branch_summary["R1_EXACT_FUZZY"]["candidate_return_rate"] > 0
        and branch_summary["R2_BM25"]["candidate_return_rate"] > 0
        and branch_summary["R3_BIOMEDICAL_DENSE"]["candidate_return_rate"] > 0
        and branch_summary["R5_INDIA_KB"]["candidate_return_rate"] > 0
        and hashes_before == hashes_after
    )
    summary = {
        "generated_at": _now_iso(),
        "stage2c_version": STAGE2C_VERSION,
        "mentions_processed": int(len(mentions)),
        "unique_surfaces": int(len(surfaces)),
        "branches": branch_summary,
        "median_true_union_size": median(union_sizes) if union_sizes else 0,
        "pairwise_branch_overlap": overlap.to_dict("records"),
        "candidate_source_state_counts": source_states,
        "candidate_source_state_proportions": candidate_state_proportions,
        "dense_model": dense_meta,
        "paths": {
            "branch_traces": str(trace_path),
            "candidate_union": str(union_path),
            "branch_status": str(status_path),
            "pairwise_overlap": str(overlap_path),
            "kb_input_hashes": str(hash_path),
            "bm25_index": str(root / "knowledge/cache/stage2c/bm25_index.pkl"),
            "dense_metadata": str(dense_meta_path),
            "rxnav_cache": str(root / "knowledge/cache/stage2c/rxnav"),
            "diagnostic_review_package": str(sample_path),
            "report": str(report_dir / "STAGE2C_CANDIDATE_RETRIEVAL_REPORT.md"),
        },
        "diagnostic_sample_rows": int(len(sample)),
        "kb_inputs_unchanged": hashes_before == hashes_after,
        "ranking_performed": False,
        "gold_metrics_reported": False,
        "prescription_driven_source_acquisition": False,
        "unresolved_technical_issues": unresolved,
        "READY_FOR_RANKING": bool(ready),
    }
    summary_path = derived_dir / "stage2c_summary.json"
    summary["paths"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(report_dir / "STAGE2C_CANDIDATE_RETRIEVAL_REPORT.md", summary)
    return summary


def run_stage2c_r3_resume(root: Path | None = None) -> dict[str, Any]:
    """Resume Stage 2C by rebuilding only R3 dense outputs from existing traces."""
    root = root or Path(__file__).resolve().parents[2]
    derived_dir = root / "derived/retrieval"
    trace_path = derived_dir / "stage2c_branch_traces.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"Stage 2C trace not found: {trace_path}")

    trace = _read_csv(trace_path)
    non_r3 = trace[trace["branch"] != "R3_BIOMEDICAL_DENSE"].copy()
    mention_frame = trace[["mention_id", "raw_medication_text"]].drop_duplicates().reset_index(drop=True)
    surfaces = mention_frame["raw_medication_text"].drop_duplicates().tolist()

    dense = R3BiomedicalDense(root, root / "knowledge/cache/stage2c")
    surface_results: dict[str, tuple[str, list[RetrievalHit], float, str]] = {}
    for i, surface in enumerate(surfaces, start=1):
        if i % 50 == 0:
            print(f"stage2c R3 surfaces processed: {i}/{len(surfaces)}", flush=True)
        start = time.perf_counter()
        if dense.available:
            try:
                hits = dense.search(surface, top_n=TOP_N)
                status = STATUS_SUCCESS if hits else STATUS_EMPTY
                error = ""
            except Exception as exc:
                hits = []
                status = STATUS_FAILED
                error = f"{type(exc).__name__}: {exc}"
        else:
            hits = []
            status = STATUS_UNAVAILABLE
            error = dense.unavailable_reason
        surface_results[surface] = (status, hits, (time.perf_counter() - start) * 1000, error)

    r3_rows: list[dict[str, Any]] = []
    for _, mention in mention_frame.iterrows():
        status, hits, latency_ms, error = surface_results[mention["raw_medication_text"]]
        if not hits:
            r3_rows.append(
                {
                    "mention_id": mention["mention_id"],
                    "raw_medication_text": mention["raw_medication_text"],
                    "branch": "R3_BIOMEDICAL_DENSE",
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
                    "status": status,
                    "error": error,
                }
            )
            continue
        for rank, hit in enumerate(hits, start=1):
            r3_rows.append(
                {
                    "mention_id": mention["mention_id"],
                    "raw_medication_text": mention["raw_medication_text"],
                    "branch": "R3_BIOMEDICAL_DENSE",
                    "candidate_id": hit.candidate_id,
                    "candidate_name": hit.candidate_name,
                    "candidate_type": hit.candidate_type,
                    "rank": rank,
                    "score": round(hit.score, 6),
                    "score_semantics": hit.score_semantics,
                    "matched_field": hit.matched_field,
                    "matched_alias": hit.matched_alias,
                    "source_state": hit.source_state,
                    "authority": hit.authority,
                    "provenance_evidence_ids": hit.provenance_evidence_ids,
                    "latency_ms": round(latency_ms, 3),
                    "status": status,
                    "error": error,
                }
            )
    r3 = pd.DataFrame(r3_rows)
    r3.to_csv(dense.surface_results_path, index=False)

    updated = pd.concat([non_r3, r3], ignore_index=True)
    branch_order = {branch: i for i, branch in enumerate(BRANCHES)}
    updated["_branch_order"] = updated["branch"].map(branch_order)
    updated["_rank_order"] = pd.to_numeric(updated["rank"], errors="coerce").fillna(0)
    updated = updated.sort_values(["mention_id", "_branch_order", "_rank_order"], kind="stable").drop(columns=["_branch_order", "_rank_order"])
    updated.to_csv(trace_path, index=False)

    union = _union_rows(updated)
    union_path = derived_dir / "stage2c_candidate_union.csv"
    union.to_csv(union_path, index=False)

    status_path = derived_dir / "stage2c_branch_status.csv"
    status = updated.groupby(["mention_id", "raw_medication_text", "branch"], as_index=False).agg(
        status=("status", "first"),
        latency_ms=("latency_ms", "first"),
        candidates=("candidate_id", lambda s: int((s.astype(str) != "").sum())),
    )
    status.to_csv(status_path, index=False)

    overlap = _pairwise_overlap(updated)
    overlap_path = derived_dir / "stage2c_pairwise_overlap.csv"
    overlap.to_csv(overlap_path, index=False)

    mentions = _read_csv(root / "derived/layer_a_medication_mentions.csv")
    sample_path = root / "review/retrieval_candidate_inspection_100.csv"
    sample = _diagnostic_sample(mentions, updated, sample_path)

    dense_meta_path = root / "knowledge/cache/stage2c/dense_metadata.json"
    dense_meta = json.loads(dense_meta_path.read_text(encoding="utf-8")) if dense_meta_path.exists() else {}
    mention_count = int(mention_frame["mention_id"].nunique())
    branch_summary = _status_summary(updated, mention_count)
    union_sizes = union.groupby("mention_id")["candidate_id"].nunique().tolist() if not union.empty else []
    source_states = updated[updated["candidate_id"].astype(str) != ""]["source_state"].value_counts().to_dict()
    state_total = sum(source_states.values()) or 1
    unresolved: list[str] = []
    if not dense_meta.get("available", False):
        unresolved.append(f"R3 biomedical dense unavailable: {dense_meta.get('unavailable_reason', 'unknown')}")
    if branch_summary["R3_BIOMEDICAL_DENSE"]["failure_rate"] > 0:
        unresolved.append("R3 dense retrieval had failed branch calls; inspect trace error column.")

    hash_path = derived_dir / "stage2c_kb_input_hashes.json"
    hashes = json.loads(hash_path.read_text(encoding="utf-8")) if hash_path.exists() else {"unchanged": True}
    ready = (
        branch_summary["R1_EXACT_FUZZY"]["candidate_return_rate"] > 0
        and branch_summary["R2_BM25"]["candidate_return_rate"] > 0
        and branch_summary["R3_BIOMEDICAL_DENSE"]["candidate_return_rate"] > 0
        and branch_summary["R4_RXNORM"]["failure_rate"] == 0
        and branch_summary["R5_INDIA_KB"]["candidate_return_rate"] > 0
        and bool(hashes.get("unchanged", True))
    )
    summary = {
        "generated_at": _now_iso(),
        "stage2c_version": STAGE2C_VERSION,
        "resume_mode": "R3_ONLY_EXISTING_TRACE_SPLICE",
        "mentions_processed": mention_count,
        "unique_surfaces": int(len(surfaces)),
        "branches": branch_summary,
        "median_true_union_size": median(union_sizes) if union_sizes else 0,
        "pairwise_branch_overlap": overlap.to_dict("records"),
        "candidate_source_state_counts": source_states,
        "candidate_source_state_proportions": {state: count / state_total for state, count in source_states.items()},
        "dense_model": dense_meta,
        "paths": {
            "branch_traces": str(trace_path),
            "candidate_union": str(union_path),
            "branch_status": str(status_path),
            "pairwise_overlap": str(overlap_path),
            "kb_input_hashes": str(hash_path),
            "bm25_index": str(root / "knowledge/cache/stage2c/bm25_index.pkl"),
            "dense_metadata": str(dense_meta_path),
            "dense_faiss_index": str(root / "knowledge/cache/stage2c/dense_faiss.index"),
            "dense_surface_top25": str(root / "knowledge/cache/stage2c/dense_surface_top25.csv"),
            "rxnav_cache": str(root / "knowledge/cache/stage2c/rxnav"),
            "diagnostic_review_package": str(sample_path),
            "report": str(root / "rebuild/reports/STAGE2C_CANDIDATE_RETRIEVAL_REPORT.md"),
        },
        "diagnostic_sample_rows": int(len(sample)),
        "kb_inputs_unchanged": bool(hashes.get("unchanged", True)),
        "ranking_performed": False,
        "gold_metrics_reported": False,
        "prescription_driven_source_acquisition": False,
        "unresolved_technical_issues": unresolved,
        "READY_FOR_RANKING": bool(ready),
    }
    summary_path = derived_dir / "stage2c_summary.json"
    summary["paths"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(root / "rebuild/reports/STAGE2C_CANDIDATE_RETRIEVAL_REPORT.md", summary)
    return summary
