# Evaluation

Final public evaluation claims are sourced from frozen aggregate artifacts only:

- `configs/frozen/evaluation_final_893_manifest.json`
- `paper_artifacts/final_metrics/final_metrics.json`
- `paper_artifacts/accounting/PAPER_CLAIM_REGISTRY_FINAL.csv`
- `paper_artifacts/accounting/PAPER_NUMBER_DICTIONARY_FINAL.json`

The final end-to-end pipeline study evaluates `893` prescriptions, including prescriptions with no medication mentions. The public release reports corpus accounting, pipeline disposition, resolution levels, semantic identifier coverage, knowledge-resource usage, and consistency checks. It does not report human-reference semantic accuracy.

The semantic audit is a separate automated evaluation study. Qwen V2 independently assessed `762` medication mentions from a stratified `150`-prescription cohort and produced SUPPORT/CONTRADICTION/INSUFFICIENT_EVIDENCE and routing-agreement labels. These labels are inter-model audit evidence, not clinical ground truth.

The public synthetic smoke test verifies execution and output contracts only. It is not a scientific evaluation.
