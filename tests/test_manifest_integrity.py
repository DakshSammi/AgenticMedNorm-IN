from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_config_manifest_is_valid():
    config = json.loads((ROOT / "configs/examples/synthetic_pipeline_config.json").read_text(encoding="utf-8"))
    assert config["pipeline_version"] == "public_synthetic_smoke_v1"
    assert config["ranking"]["rrf_k"] == 60
    assert len(config["candidate_catalog"]) >= 3
    assert all(item["candidate_id"].startswith("ENTITY:") for item in config["candidate_catalog"])
