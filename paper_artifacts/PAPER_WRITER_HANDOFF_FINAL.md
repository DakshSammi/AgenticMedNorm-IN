# Paper Writer Handoff Final

## Verified Final Numbers

Study A: benchmark125 is `LEGACY_MODEL_SELECTION_BENCHMARK`; GPT-5.5 best primary score is 0.9134.
Study B: 893 prescriptions, 782 medication-bearing, 111 zero-medication, 3027 mentions, 1276 unique normalized surfaces.
Study C: Qwen 762 audit: {'SUPPORTED': 558, 'CONTRADICTED': 109, 'INSUFFICIENT_EVIDENCE': 95}; routing {'AGREE': 562, 'DISAGREE': 200}.

## Exact Methods Implementation

- Six-agent pipeline: de-identification, annotation, retrieval, ranking, evidence assessment, verification.
- Retrieval: R1 exact/fuzzy; R2 BM25; R3 SapBERT biomedical dense; R4 RxNorm/RxNav; R5 India-specific resources.
- Ranking: true candidate union, unweighted RRF, `k=60`, top-20 to evidence.
- Evidence and verification are deterministic and source/provenance-preserving.
- Qwen audit is evaluation-only and not a seventh pipeline agent.

## Current Manuscript Corrections

- Replace `more than 1,000 prescriptions` with verified 893-prescription language.
- Remove claims of independent clinical-expert review, expert-validated subset, expert primary semantic correctness estimate, and expert adjudication metrics.
- Replace LLM-judge-calibrated-against-experts language with stratified automated LLM semantic audit language.
- Replace empty/stale Results with the generated final tables.
- Retitle expert-focused reliability sections toward semantic auditing and provenance.
- Add future work: expert adjudication, multi-reviewer agreement, expert calibration, expert-adjudicated public benchmark.

## Benchmark Wording

Preferred: On the stratified 125-document model-selection benchmark, GPT-5.5 achieved the highest score among the evaluated systems under the study protocol. Do not claim general proprietary superiority.

## Results Outline

A. Corpus and Processing Coverage; B. Semantic Normalization and Verification Outcomes; C. Identifier Coverage; D. Annotation-Model Selection Benchmark; E. Stratified LLM-Based Semantic Audit; F. Inter-Model Concordance.

## Discussion Outline

A. From transcription to semantic medication representation; B. India-specific knowledge and formulation preservation; C. Selective verification and automated semantic auditing; D. Model-selection limits; E. Limitations/future expert validation.

## Safe To Claim
- 893 prescriptions were processed by the final pipeline.
- The final pipeline produced 3,027 medication mentions and explicit ACCEPT/HUMAN_REVIEW/NIL states.
- The pipeline preserves local Indian semantic identifiers and RxNorm/ATC identifiers where supported.
- Qwen independently classified 558/109/95 mappings as SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE on the 762-mention audit cohort.
- On the 125-document model-selection benchmark, GPT-5.5 achieved the highest score among evaluated systems under the study protocol.

## Do Not Claim
- human-validated accuracy
- clinical correctness
- expert gold-standard performance
- general superiority of proprietary models
- patient-outcome benefit
- Recall@K/MRR without independent reference
- all mappings are correct
- RxNorm coverage means exact Indian product equivalence

## Data Availability

Code, configuration, aggregate artifacts, synthetic examples, and public-source KB build scripts may be released. Real de-identified prescription images and row-level annotations are deferred pending expert adjudication and governance approval.

## Limitations
- single institution
- General Medicine OPD
- automated annotation
- no completed human semantic adjudication
- LLM-as-judge is not ground truth
- benchmark model-family generalization not supported
- India-specific KB coverage incomplete
- RxNorm is U.S.-oriented
- ATC mapping partial
- open high-recall candidate layer non-authoritative
- external validation absent
- HUMAN_REVIEW queue not fully adjudicated

## Main Figure Recommendations
- `paper_artifacts/figures/fig01_six_agent_architecture/figure.pdf`
- `paper_artifacts/figures/fig02_study_cohort_flow/figure.pdf`
- `paper_artifacts/figures/fig06_semantic_identifier_coverage/figure.pdf`
- `paper_artifacts/figures/fig07_qwen_semantic_audit/figure.pdf`

## Main Table Recommendations
- `paper_artifacts/tables/table_final_corpus_accounting.csv`
- `paper_artifacts/tables/table_pipeline_disposition.csv`
- `paper_artifacts/tables/table_resolution_levels.csv`
- `paper_artifacts/tables/table_qwen_semantic_audit.csv`
