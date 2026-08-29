# Semantic Audit

The semantic audit is Study C in the release package. It is independent of the six-agent production pipeline and should not be described as a seventh pipeline agent.

The frozen audit cohort contains `150` prescriptions and `762` medication mentions. Qwen V2 was the primary automated semantic auditor. The final aggregate results are:

- `SUPPORTED`: 558 / 762
- `CONTRADICTED`: 109 / 762
- `INSUFFICIENT_EVIDENCE`: 95 / 762
- `AGREE`: 562 / 762
- `DISAGREE`: 200 / 762
- `UNCERTAIN`: 0 / 762

Secondary auditor comparisons are reported only as inter-model concordance on overlapping valid subsets. Terra and GPT-OSS comparisons are not human-reference accuracy estimates.

Use the tables under `paper_artifacts/tables/` and `results/tables/` for manuscript numbers. Do not call `SUPPORTED` correct or `CONTRADICTED` incorrect unless future human expert adjudication establishes a reference standard.
