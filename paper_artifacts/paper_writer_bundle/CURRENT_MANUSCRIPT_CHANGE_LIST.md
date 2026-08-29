# Current Manuscript Change List

| Section | Old claim | Why stale | Verified replacement fact | Source artifact |
| --- | --- | --- | --- | --- |
| Abstract/Intro | more than 1,000 prescriptions | final verified corpus is 893 | 893 prescriptions processed | configs/frozen/evaluation_final_893_manifest.json |
| Evaluation | expert-reviewed subset calibrates LLM judge | no completed expert reference exists | stratified automated Qwen semantic audit | paper_artifacts/tables/table_qwen_semantic_audit.csv |
| Results | missing/stale quantitative results | final pipeline rerun completed | use generated final tables | paper_artifacts/tables/TABLE_MANIFEST_FINAL.csv |
| Discussion | Reliability, Provenance, and Human Validation | human validation deferred | Reliability, Provenance, and Semantic Auditing | paper_artifacts/final_metrics/final_metrics.json |
| Conclusion/C1 | independent expert validation | not completed | reproducible semantic audit and future expert adjudication | paper_artifacts/PAPER_WRITER_HANDOFF_FINAL.md |
| Benchmark | proprietary models outperform open models | unsupported generalization | GPT-5.5 led this 125-document benchmark under this protocol | paper_artifacts/tables/table_annotation_model_benchmark_main.csv |
