# Annotation Model Benchmark

The annotation-model selection study is described as an engineering benchmark, not as human clinical validation.

The intended final benchmark compared a 125-unit stratified set across:

- conventional OCR systems
- open-weight systems
- proprietary visual language systems
- hybrid pipelines

The release task requested use of final audited artifacts:

- `paper_artifacts/benchmarking/ANNOTATION_MODEL_SELECTION_STUDY.md`
- `rebuild/reports/ANNOTATION_BENCHMARK_FAIRNESS_AUDIT.md`

Those files were not present in this working tree during release preparation, so this document does not reproduce a leaderboard table. README language is therefore limited to the audited design claim and the final architecture choice: the private Annotation Creation Agent uses GPT-5.5, while public reproduction supports precomputed annotations without paid API execution.
