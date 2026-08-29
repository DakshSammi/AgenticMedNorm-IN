# Public Release Audit

Audit date: 2026-08-29

Release candidate policy:

- include code, schemas, synthetic examples, safe configuration, aggregate result tables, aggregate figure source data, rendered publication figures, frozen final893 manifests/hashes, and paper-writer handoff files
- exclude raw prescription images, de-identified prescription images, real row-level annotation JSONs, row-level reviewer packages, private runtime state, model caches, FAISS indexes, request dumps, and local server configuration

Verified release gates:

| Gate | Status | Evidence |
| --- | --- | --- |
| FINAL_893_PIPELINE_COMPLETE | PASS | `configs/frozen/evaluation_final_893_manifest.json` has `FINAL_893_PIPELINE_COMPLETE=true`. |
| FINAL_893_INTEGRITY_PASS | PASS | `results/frozen_aggregate/FINAL_893_INTEGRITY_AUDIT.md` and final manifest show all final893 artifacts complete. |
| Reproduction | PASS | `python scripts/reproduce_paper_artifacts.py` generated 16 figures and 13 tables. |
| Synthetic smoke | PASS | `python scripts/run_pipeline.py ...` produced 5 mentions and 5 Layer-B rows. |
| Tests | PASS | `python -m pytest -q` collected and passed 9 public/release tests. |
| Secret scan | PASS | `python scripts/release_scan.py --check all` reported `secrets: PASS`. |
| Private-data scan | PASS | `python scripts/release_scan.py --check all` reported `private-data: PASS`. |
| Hardcoded private path scan | PASS | `python scripts/release_scan.py --check all` reported `paths: PASS`. |
| Real prescription ignore rules | PASS | `.gitignore` excludes raw, anonymized, and ground-truth triads. |
| License notice | PASS_WITH_NOTICE | `LICENSE_REQUIRED` is present; no open-source license is invented. |

Important non-release local files remain in the workspace but are not part of the public release candidate. The public candidate file list is defined by `scripts/release_scan.py`.

Human expert adjudication was not completed for this submission. Public documentation must retain this as future work and must not claim human-validated accuracy or an expert reference standard.
