# Dataset

The working server currently contains a private clinical corpus with `893` p-numbered raw prescription images, matching anonymized images, and matching ground-truth JSONs.

These patient-level artifacts are not public-release assets:

- `prescription_pipeline_jbhi_ieee/raw/`
- `prescription_pipeline_jbhi_ieee/anonymized/`
- `prescription_pipeline_jbhi_ieee/ground_truths_json/`
- reviewer or validation files containing patient-level rows

Public examples in `data/examples/` are synthetic and non-sensitive. They are intended only to verify code execution and output contracts.

Historical note: `EVALUATION_V1_1_737` and later `867`-record checkpoints were development snapshots. Final documentation should not mix those denominators with the current `893` corpus unless the text explicitly labels them as historical.
