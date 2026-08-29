from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from src.utils.stable_ids import stable_hash


class PubChemClient:
    """Small cached PUG REST client for ingredient enrichment only."""

    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, cache_dir: Path, requests_per_second: float = 5.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / requests_per_second
        self.last_request_at = 0.0
        self.log_path = self.cache_dir / "pubchem_requests.jsonl"

    def get_compound_by_name(self, ingredient_name: str) -> dict[str, Any]:
        broad_terms = {
            "brand",
            "tablet",
            "capsule",
            "syrup",
            "injection",
            "pain",
            "fever",
            "common",
            "medicine",
            "drug",
        }
        tokens = {token.lower() for token in ingredient_name.split()}
        if not ingredient_name or len(tokens) > 4 or tokens & broad_terms:
            raise ValueError("PubChem enrichment requires a narrow ingredient name, not a broad term or brand identity.")
        cache_key = stable_hash("compound_name", ingredient_name, length=32)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        self._rate_limit()
        url = f"{self.base_url}/compound/name/{requests.utils.quote(ingredient_name)}/JSON"
        response = requests.get(url, timeout=20)
        event = {
            "ingredient_name": ingredient_name,
            "url": url,
            "status_code": response.status_code,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_at = time.monotonic()
