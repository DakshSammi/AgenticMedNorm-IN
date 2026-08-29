from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_secret_scan_passes_for_public_paths():
    result = subprocess.run([sys.executable, "scripts/release_scan.py", "--check", "secrets"], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout
