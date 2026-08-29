# Paper Artifacts

This directory is reserved for publication-safe paper artifacts such as aggregate tables, figure data, and claim registries.

Do not place patient-level rows, raw prescription images, deidentified prescription images, or reviewer-identifying files here.

Final manuscript-facing artifacts are:

- `paper_artifacts/accounting/PAPER_CLAIM_REGISTRY_FINAL.csv`
- `paper_artifacts/accounting/PAPER_NUMBER_DICTIONARY_FINAL.json`
- `paper_artifacts/final_metrics/final_metrics.json`
- `paper_artifacts/tables/TABLE_MANIFEST_FINAL.csv`
- `paper_artifacts/figures/FIGURE_MANIFEST_FINAL.csv`
- `paper_artifacts/PAPER_WRITER_HANDOFF_FINAL.md`
- `paper_artifacts/paper_writer_bundle_manifest.json`

Regenerate the release-safe tables, figures, metrics, and handoff bundle with:

```bash
python scripts/reproduce_paper_artifacts.py
```
