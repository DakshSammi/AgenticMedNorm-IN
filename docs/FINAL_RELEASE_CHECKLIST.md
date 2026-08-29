# Final Release Checklist

| Gate | Status | Evidence |
| --- | --- | --- |
| Compile | PASS | `python -m compileall -q src scripts/run_pipeline.py scripts/release_scan.py scripts/reproduce_paper_artifacts.py`. |
| Tests | PASS | `pytest -q`: 9 release tests passed. |
| Smoke test | PASS | Synthetic precomputed run produced 5 Layer-A mentions and 5 Layer-B rows. |
| Secret scan | PASS | `python scripts/release_scan.py --check all`: secrets PASS. |
| Private-data scan | PASS | `python scripts/release_scan.py --check all`: private-data PASS. |
| Hardcoded-path scan | PASS | `python scripts/release_scan.py --check all`: paths PASS for the release candidate allowlist. |
| README ready | PASS | README rewritten for six-agent public release. |
| Final manifest valid | PASS | `configs/frozen/evaluation_final_893_manifest.json` and `release_manifest.json`. |
| Paper artifacts | PASS | `python scripts/reproduce_paper_artifacts.py`: 16 figures, 13 tables. |
| Git repository present | PASS_WITH_NOTICE | Branch exists, but `HEAD` is unborn until the release commit is created. |
| Push allowed | PENDING | Allowed only after the release commit is created and remote fetch/push checks pass. |
