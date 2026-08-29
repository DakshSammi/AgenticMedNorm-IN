# Release Audit

## Public Code

- `src/`
- `scripts/run_pipeline.py`
- selected release utility scripts that do not require private data

## Public Config

- `configs/examples/`
- frozen configs that do not expose private paths or credentials

## Public Documentation

- `README.md`
- `docs/`
- `data/README.md`

## Public Aggregate Results

Aggregate-only frozen results may be published after claim-registry verification. Patient-level rows are excluded by default. The final public artifact set uses `paper_artifacts/final_metrics/final_metrics.json`, `paper_artifacts/accounting/PAPER_CLAIM_REGISTRY_FINAL.csv`, and the final table/figure manifests.

## Public Example Data

- `data/examples/` synthetic annotations only
- `data/examples_synthetic/` mirrored synthetic annotations for release packaging

## Private Data

- `prescription_pipeline_jbhi_ieee/raw/`
- `prescription_pipeline_jbhi_ieee/anonymized/`
- `prescription_pipeline_jbhi_ieee/ground_truths_json/`
- patient-level generated annotations

## Large Generated Data

- `derived/`
- `generated/`
- `outputs/`
- `knowledge/cache/`
- FAISS indexes and model caches

## Credentials

- `.env`
- key/token files
- credential helper outputs

## Internal Infrastructure

- server-specific configs under `configs/servers/`
- local service URLs
- machine-specific absolute paths

## Temporary Or Diagnostic Outputs

- `logs/`
- `review/`
- `needs_review/`
- pytest caches and Python bytecode

## Stale Results

Older `737` and `867` reports are historical unless a final claim registry explicitly references them.

## Paper-Only Artifacts

Paper tables and figures are released only as aggregate artifacts after final claim consistency checks. The release candidate allowlist is enforced by `scripts/release_scan.py`.
