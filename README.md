<div align="center">

# AgenticMedNorm-IN

### Knowledge-Guided Semantic Medication Normalization from Handwritten Indian Prescriptions

[Paper](#citation) · [Dataset](#public-llm-audited-dataset) · [Quick Start](#quick-start) · [Results](#pipeline-outcomes) · [Documentation](docs/)

Submitted to the IEEE Journal of Biomedical and Health Informatics, 2026.

</div>

---

## Why AgenticMedNorm-IN?

Handwritten Indian prescriptions contain local brand names, fixed-dose combinations, shorthand dosing, and spelling variation that defeat standard OCR pipelines. AgenticMedNorm-IN separates transcription from normalization: the system records what is visible, then resolves each medication mention to a supported local product, brand family, ingredient, or international terminology—explicitly routing ambiguous cases to human review rather than hallucinating a mapping.

---

## At a Glance

| | |
|---|---|
| Full corpus | 893 processed prescriptions, 3,027 medication mentions |
| Medication-bearing | 782 prescriptions (111 zero-medication) |
| Unique surfaces | 1,276 |
| Pipeline agents | 6 bounded agents |
| Retrieval branches | 5 (exact/fuzzy, BM25, dense, RxNorm, India-KB) |
| Public validation set | 150 prescriptions, 762 medication mentions |
| Semantic audit | Qwen3-30B on 762 mentions |

---

## Architecture

```
Handwritten Prescription
        │
   ┌────▼────┐
   │  A1     │  De-identification
   └────┬────┘
        │
   ┌────▼────┐
   │  A2     │  Annotation Creation (structured extraction)
   └────┬────┘
        │
   ┌────▼────────────────────────────────────┐
   │  A3  Candidate Retrieval                │
   │  R1: exact/fuzzy surface match          │
   │  R2: BM25 lexical retrieval             │
   │  R3: SapBERT biomedical dense retrieval │
   │  R4: RxNorm/RxNav terminology lookup    │
   │  R5: India-specific KB (NPPA, CDSCO)    │
   └────┬────────────────────────────────────┘
        │
   ┌────▼────┐
   │  A4     │  Candidate Ranking (RRF, k=60)
   └────┬────┘
        │
   ┌────▼────┐
   │  A5     │  Evidence Assessment
   └────┬────┘
        │
   ┌────▼────┐
   │  A6     │  Verification
   └────┬────┘
        │
   ┌────▼──────────────────────────┐
   │  Layer B Normalized Output    │
   │  ACCEPT / HUMAN_REVIEW / NIL  │
   └───────────────────────────────┘
```

---

## Six Agents

| Agent | Role |
|-------|------|
| **A1: De-identification** | Removes protected health information from prescription images |
| **A2: Annotation Creation** | Structured visual extraction of medication fields |
| **A3: Candidate Retrieval** | Multi-branch evidence gathering across five retrieval sources |
| **A4: Candidate Ranking** | Score fusion via unweighted reciprocal rank fusion (k=60) |
| **A5: Evidence Assessment** | Supporting and contradicting evidence scoring |
| **A6: Verification** | Final routing: ACCEPT, HUMAN_REVIEW, or NIL with reason codes |

---

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Smoke Test (synthetic examples, no API keys needed)

```bash
python scripts/run_pipeline.py \
  --input data/examples \
  --annotations-dir data/examples/annotations \
  --config configs/examples/synthetic_pipeline_config.json \
  --output outputs/example_run \
  --resume
```

### Dry Run

```bash
python scripts/run_pipeline.py \
  --input data/examples \
  --annotations-dir data/examples/annotations \
  --config configs/examples/synthetic_pipeline_config.json \
  --output outputs/example_run \
  --dry-run
```

---

## Public LLM-Audited Dataset

The `dataset/llm_audited_150/` directory contains the public validation cohort:

| Component | Records | Description |
|-----------|---------|-------------|
| `images/` | 150 | De-identified prescription images (AMNIN_RX_0001–0150) |
| `annotations.json` | 762 | Medication mention annotations |
| `normalization.json` | 762 | Pipeline normalization outputs |
| `llm_audit.json` | 762 | Qwen3-30B semantic audit results |
| `checksums.sha256` | — | SHA-256 integrity verification |

All public IDs use the `AMNIN_RX_XXXX` format. Internal identifiers are not included.

**Semantic auditor:** `dengcao/Qwen3-30B-A3B-Instruct-2507:latest` (Q4_K_M, 30.5B parameters, Ollama backend, temperature=0).

**Audit results (N=762):**

| Assessment | N | % |
|------------|---|---|
| SUPPORTED | 558 | 73.2% |
| CONTRADICTED | 109 | 14.3% |
| INSUFFICIENT_EVIDENCE | 95 | 12.5% |
| AGREE (pipeline routing) | 562 | 73.8% |
| DISAGREE (pipeline routing) | 200 | 26.2% |

---

## Pipeline Outcomes

### Corpus and Verification (full 893 corpus)

| Metric | N | % |
|--------|---|---|
| Processed prescriptions | 893 | — |
| Medication-bearing | 782 | 87.6% |
| Zero-medication | 111 | 12.4% |
| Medication mentions | 3,027 | — |
| Unique surface forms | 1,276 | — |

| Verification Decision | N | % |
|-----------------------|---|---|
| ACCEPT | 2,803 | 92.6% |
| HUMAN_REVIEW | 223 | 7.4% |
| NIL | 1 | <0.1% |

> **Note:** HUMAN_REVIEW denotes a pipeline routing state and does not imply completed expert adjudication.

### Semantic Identifier Coverage (full 893 corpus)

| Identifier Type | N | % |
|-----------------|---|---|
| RxNorm RxCUI | 424 | 14.0% |
| ATC therapeutic class | 106 | 3.5% |
| Local brand-family ID | 2,623 | 86.7% |
| Any local semantic ID | 3,017 | 99.7% |

---

## Knowledge Resources

| Resource | Purpose |
|----------|---------|
| NPPA Pharma Sahi Daam | Indian drug pricing and brand index |
| CDSCO | Approved drugs and fixed-dose combinations |
| NLEM 2022 | National List of Essential Medicines |
| RxNorm / RxNav | US drug terminology (ingredient and term mapping) |
| ATC / RxClass | WHO therapeutic classification |
| Open Indian Medicine Dataset | Indian brand name coverage |

Full documentation: [docs/knowledge_resources.md](docs/knowledge_resources.md)

---

## Annotation Model Selection

A separate frozen 125-document benchmark was used to select the annotation configuration. The best primary system (GPT-5.5 direct structured annotation) achieved token-level set-based F1 of 0.523 [95% CI: 0.475–0.569] on that benchmark. Full benchmark results are in `paper_artifacts/tables/`.

---

## Reproducibility

### Full Pipeline

```bash
python scripts/run_pipeline.py \
  --input prescription_pipeline_jbhi_ieee/raw \
  --annotations-dir prescription_pipeline_jbhi_ieee/annotations_json \
  --config configs/frozen/evaluation_final_893_manifest.json \
  --output outputs/full_893 \
  --resume
```

### Benchmark Reconstruction

```bash
python scripts/reproduce_paper_artifacts.py
```

### Dataset Verification

```bash
cd dataset/llm_audited_150
sha256sum -c checksums.sha256
```

See [docs/reproducibility.md](docs/reproducibility.md) for complete instructions.

---

## Repository Structure

```
AgenticMedNorm-IN/
├── src/                    Six-agent pipeline core
│   ├── adapters/           Backend adapters (VLM, OCR)
│   ├── annotation/         Agent A2: annotation creation
│   ├── benchmark/          Evaluation metrics and runner
│   ├── deidentification/   Agent A1: PHI removal
│   ├── evidence/           Agent A5: evidence assessment
│   ├── knowledge/          Knowledge base construction
│   ├── pipeline/           Orchestrator
│   ├── ranking/            Agent A4: candidate ranking
│   ├── retrieval/          Agent A3: multi-branch retrieval
│   ├── schemas/            Pydantic data models
│   └── verification/       Agent A6: routing decisions
├── scripts/                Orchestration and utility scripts
├── configs/                Frozen pipeline configurations
├── tests/                  Test suite (9 active tests)
├── dataset/
│   └── llm_audited_150/    Public validation cohort
├── paper_artifacts/        Paper tables, figures, metrics
├── docs/                   Architecture, evaluation, reproducibility
└── knowledge/
    └── reports/            Implementation matrix
```

---

## Limitations

- The corpus is limited to Indian handwritten prescriptions in English.
- Not all local brands or fixed-dose combinations can be resolved automatically. Verification returns HUMAN_REVIEW or NIL when evidence is missing or conflicting.
- RxNorm/RxNav and dense retrieval (SapBERT/FAISS) may require separately downloaded resources.
- The public LLM audit is a semantic audit, not expert human validation.

---

## Citation

```bibtex
@software{agenticmednormin2026,
  title       = {AgenticMedNorm-IN},
  author      = {Daksh Sammi and Sanju Tiwari and Mayank Kejriwal and Ashok Kumar and Deepak Sharma},
  year        = {2026},
  url         = {https://github.com/DakshSammi/AgenticMedNorm-IN},
  status      = {submitted}
}
```

**Manuscript citation:** Sammi, D., Tiwari, S., Kejriwal, M., Kumar, A., & Sharma, D. (2026). AgenticMedNorm-IN: Knowledge-Guided Semantic Medication Normalization from Handwritten Indian Prescriptions. *Submitted to IEEE Journal of Biomedical and Health Informatics.* Bibliographic details will be updated upon publication.

Citation metadata: [CITATION.cff](CITATION.cff)

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

---

## Data Availability and Privacy

**Full 893 corpus:** De-identified prescription images, raw images, and ground-truth JSONs containing patient-level clinical content are not publicly released as a complete image dataset. Access requires a separate governance and data-use process.

**Public stratified subset:** The `dataset/llm_audited_150/` directory contains 150 de-identified prescription images with publication-safe annotations, normalization outputs, and LLM audit results. All internal identifiers have been replaced with public AMNIN_RX identifiers.
