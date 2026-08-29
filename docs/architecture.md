# Architecture

AgenticMedNorm-IN is organized as a six-agent medication normalization pipeline.

1. De-identification Agent removes or masks patient- and clinician-identifying regions before annotation.
2. Annotation Creation Agent transcribes visible medication and clinical context into structured JSON. The private final pipeline uses GPT-5.5; public reproduction can use precomputed annotations through `--annotations-dir`.
3. Candidate Retrieval Agent produces candidates from five branches: exact/fuzzy lookup, BM25, biomedical dense retrieval/SapBERT, RxNorm/RxNav, and India-specific structured resources.
4. Candidate Ranking Agent builds the true candidate union and applies unweighted reciprocal rank fusion with `k=60`.
5. Evidence Assessment Agent compares lexical, formulation, provenance, terminology, and contextual evidence without introducing new candidates.
6. Verification Agent emits accepted, human-review, or NIL Layer-B decisions.

The LLM judge is evaluation-only and is not part of the operational six-agent core.
