from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_synthetic_pipeline_smoke(tmp_path):
    out = tmp_path / "smoke"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--input",
            "data/examples",
            "--annotations-dir",
            "data/examples/annotations",
            "--config",
            "configs/examples/synthetic_pipeline_config.json",
            "--output",
            str(out),
            "--resume",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "evaluation_export.json").read_text(encoding="utf-8"))
    assert summary["mentions"] == 5
    assert summary["layer_b_rows"] == 5
    for name in ["layer_a_medication_mentions.csv", "candidate_union.csv", "ranked_candidates.csv", "evidence_assessments.csv", "verification_results.csv", "layer_b.csv"]:
        assert (out / name).exists()
