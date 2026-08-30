# AgenticMedNorm-IN

> A reproducible six-agent pipeline for medication normalization in Indian handwritten prescriptions.
> Submitted to the IEEE Journal of Biomedical and Health Informatics, 2026.

---

## At a Glance

| Property | Value |
|----------|-------|
| Corpus | 893 processed prescriptions, 3027 medication mentions |
| Public dataset | 150 prescriptions, 762 mentions (LLM-audited) |
| Pipeline agents | 6 (De-identification, Annotation, Retrieval, Ranking, Evidence, Verification) |
| Retrieval branches | 5 (R1 exact/fuzzy, R2 BM25, R3 dense, R4 RxNorm, R5 India-KB) |
| Ranking | Reciprocal Rank Fusion (k=60) |
| Verification routing | ACCEPT=2803, HUMAN_REVIEW=223, NIL=1 |
| Best benchmark | GPT-5.5 direct annotation: token_f1=0.5230 [0.4752, 0.5687] |
| Structured extraction | GPT-5.5 0.3598 vs Gemini 0.3601 (no statistical difference) |
| LLM semantic audit | Qwen3-30B: 73.2% SUPPORTED, 14.3% CONTRADICTED, 12.5% INSUFFICIENT_EVIDENCE |

---

## Architecture

```
Prescription Image
       |
  [A1] De-identification Agent
       |
  [A2] Annotation Creation Agent
       |
  [A3] Candidate Retrieval Agent
       |--- R1: Exact/Fuzzy surface match
       |--- R2: BM25 lexical retrieval
       |--- R3: Biomedical dense retrieval (SapBERT)
       |--- R4: RxNorm/RxNav terminology lookup
       |--- R5: India-specific KB (NPPA, CDSCO, NLEM)
       |
  [A4] Candidate Ranking Agent (RRF k=60)
       |
  [A5] Evidence Assessment Agent
       |
  [A6] Verification Agent
       |
  Normalized Output (resolution_level, semantic IDs, evidence)
```

The pipeline separates transcription from normalization: transcription records what is visible in a prescription, while normalization resolves medication mentions to supported local products, brand families, ingredients, formulations, RxNorm concepts, and ATC mappings where available.

---

## Agents

| Agent | Role | Key Capability |
|-------|------|----------------|
| A1: De-identification | Removes PHI from images | Redaction, face blur, date masking |
| A2: Annotation Creation | Structured visual extraction | GPT-5.5 structured output, field-level parsing |
| A3: Candidate Retrieval | Multi-branch evidence gathering | 5 retrieval branches, true candidate union |
| A4: Candidate Ranking | Score fusion | RRF with k=60, Top-K selection |
| A5: Evidence Assessment | Supporting/contradicting evidence | NPPA, CDSCO, RxNorm, ATC evidence scoring |
| A6: Verification | Final routing decision | ACCEPT / HUMAN_REVIEW / NIL with reason codes |

---

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` only if you need optional online services. The public smoke test does not require API keys.

### Run Smoke Test

```bash
python scripts/run_pipeline.py \
  --input data/examples \
  --annotations-dir data/examples/annotations \
  --config configs/examples/synthetic_pipeline_config.json \
  --output outputs/example_run \
  --resume
```

### Dry Run (no outputs written)

```bash
python scripts/run_pipeline.py \
  --input data/examples \
  --annotations-dir data/examples/annotations \
  --config configs/examples/synthetic_pipeline_config.json \
  --output outputs/example_run \
  --dry-run
```

---

## Public Dataset

The `dataset/llm_audited_150/` directory contains the public validation cohort:

| File | Records | Description |
|------|---------|-------------|
| `images/` | 150 | Anonymized prescription images (AMNIN_RX_0001-0150) |
| `annotations.json` | 762 | Medication mention annotations |
| `normalization.json` | 762 | Pipeline normalization outputs |
| `llm_audit.json` | 762 | Qwen3-30B semantic audit results |
| `checksums.sha256` | - | SHA-256 integrity verification |

All public IDs use the `AMNIN_RX_XXXX` format. Internal p-IDs and mention IDs are not included.

**ID remapping:** `AMNIN_RX_0001` through `AMNIN_RX_0150` map to internal prescription IDs. Each prescription's mentions are suffixed `_M001`, `_M002`, etc. The internal mapping is excluded from the repository.

**Audit coverage:** 711 of 762 mentions were processed by the Qwen semantic auditor. The remaining 51 are marked `NOT_AUDITED`.

---

## Benchmark

All benchmark results use token-level set-based F1 (macro-averaged over 125 GPT-5.5 annotated prescriptions).

### Table 5: Annotation Model Benchmark

| System | Track | Token F1 | 95% CI |
|--------|-------|-----------|--------|
| GPT-5.5 | DIRECT_STRUCTURED_VLM | 0.5230 | [0.4752, 0.5687] |
| Gemini 2.5 Flash | DIRECT_STRUCTURED_VLM | 0.5185 | [0.4760, 0.5603] |
| Qwen3-VL-235B | DIRECT_STRUCTURED_VLM | 0.4958 | [0.4563, 0.5339] |
| HF Qwen2.5-VL-72B | DIRECT_STRUCTURED_VLM | 0.4859 | [0.4461, 0.5235] |
| DocTR + Qwen3 | HYBRID | 0.2780 | [0.2421, 0.3156] |
| DocTR | RAW_OCR | 0.2712 | [0.2365, 0.3062] |
| TrOCR + Qwen3 | HYBRID | 0.0215 | [0.0131, 0.0313] |
| TrOCR | RAW_OCR | 0.0146 | [0.0086, 0.0227] |

**Statistical comparison:** GPT-5.5 vs Gemini 2.5 Flash — bootstrap 95% CI for difference includes zero; no statistically significant superiority.

### Structured Extraction (entity-level)

| System | Metric |
|--------|--------|
| GPT-5.5 | 0.3598 |
| Gemini 2.5 Flash | 0.3601 |

Virtually tied; neither system is statistically superior.

---

## Results

### Corpus Accounting (Table 1)

| Metric | Count |
|--------|-------|
| Raw prescriptions | 893 |
| Medication-bearing | 782 |
| Zero-medication | 111 |
| Total medication mentions | 3027 |
| Unique lexical surfaces | 1276 |

### Verification Routing (Table 2)

| Decision | Count | Rate |
|----------|-------|------|
| ACCEPT | 2803 | 92.6% |
| HUMAN_REVIEW | 223 | 7.4% |
| NIL | 1 | <0.1% |

### LLM Semantic Audit (Table 8, N=762)

| Assessment | Count | Rate |
|------------|-------|------|
| SUPPORTED | 558 | 73.2% |
| CONTRADICTED | 109 | 14.3% |
| INSUFFICIENT_EVIDENCE | 95 | 12.5% |

| Pipeline Decision | Count | Rate |
|-------------------|-------|------|
| AGREE | 562 | 73.8% |
| DISAGREE | 200 | 26.2% |

---

## Knowledge Resources

Operational knowledge sources used in the pipeline:

| Resource | Purpose | Coverage |
|----------|---------|----------|
| NPPA Pharma Sahi Daam | Indian drug pricing/index | Brand family matching |
| CDSCO | Approved drugs/FDCs | Formulation validation |
| NLEM 2022 | National List of Essential Medicines | Ingredient canonicalization |
| RxNorm/RxNav | US drug terminology | Ingredient/term mapping |
| ATC/RxClass | WHO classification | Therapeutic category |
| Open Indian Medicine Dataset | Indian brand names | Local brand families |

Full documentation: [docs/knowledge_resources.md](docs/knowledge_resources.md)

---

## Reproducibility

### Full Pipeline Reproduction

```bash
python scripts/run_pipeline.py \
  --input prescription_pipeline_jbhi_ieee/raw \
  --annotations-dir prescription_pipeline_jbhi_ieee/annotations_json \
  --config configs/frozen/evaluation_final_893_manifest.json \
  --output outputs/full_893 \
  --resume
```

### Benchmark Table Reconstruction

```bash
python scripts/reproduce_paper_artifacts.py
```

### Public Dataset Verification

```bash
cd dataset/llm_audited_150
sha256sum -c checksums.sha256
```

See [docs/reproducibility.md](docs/reproducibility.md) for complete instructions covering resource construction, pipeline execution, figure-data generation, and expert-validation metric generation.

---

## Repository Structure

```
AgenticMedNorm-IN/
  src/                    Six-agent pipeline core
    adapters/             Backend adapters (VLM, OCR, FLORENCE, etc.)
    annotation/           Agent A2: annotation creation
    benchmark/            Evaluation metrics and runner
    deidentification/     Agent A1: PHI removal
    evidence/             Agent A5: evidence assessment
    knowledge/            Knowledge base construction
    ocr_benchmark/        OCR engine evaluation suite
    pipeline/             Orchestrator
    ranking/              Agent A4: candidate ranking (RRF)
    retrieval/            Agent A3: multi-branch retrieval
    schemas/              Pydantic data models
    semantic_pipeline/    NER, normalization, ontology mapping
    state/                Pipeline state management
    utils/                Rate limiting, stable IDs
    verification/         Agent A6: routing decisions
  scripts/                Orchestration and utility scripts
  configs/                Frozen pipeline configurations
    frozen/               Frozen evaluation manifests
    evaluation/           Output schema definitions
  tests/                  Test suite (9 active tests in pytest.ini)
  dataset/
    llm_audited_150/      Public validation cohort
  paper_artifacts/        Paper tables, figures, metrics
    accounting/           Paper claim registry
    figures/              Generated figures (fig01-fig16)
    final_metrics/        Verified benchmark scores
    tables/               CSV + LaTeX tables
  docs/                   Architecture, evaluation, reproducibility docs
  knowledge/
    reports/              Implementation matrix
  supplementary/          Supplementary materials
```

---

## Limitations

- The corpus is limited to Indian handwritten prescriptions in English. Generalizability to other scripts or languages is not claimed.
- Not all local brands or fixed-dose combinations can be resolved automatically. Verification returns HUMAN_REVIEW or NIL when evidence is missing, conflicting, or only source-record level.
- RxNorm/RxNav and dense retrieval (SapBERT/FAISS) may require separately downloaded resources.
- The public LLM audit is a semantic audit by Qwen3-30B, not expert human validation. Ground-truth adjudication is pending governance approval.
- Semantic ID coverage (RxCUI 14.0%, ATC 3.5%) reflects the current state of available mappings for Indian products, not a deficiency in the pipeline.

---

## Citation

```bibtex
@article{agenticmednormin2026,
  title={AgenticMedNorm-IN: A Six-Agent Pipeline for Medication Normalization in Indian Handwritten Prescriptions},
  author={},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  status={submitted}
}
```

Citation metadata: [CITATION.cff](CITATION.cff)

---

## License

No license file is currently included. Add the appropriate license before treating this repository as reusable by third parties.

---

## Privacy and Data Availability

Real prescription images, de-identified prescription images, and ground-truth JSONs containing patient-level clinical content are excluded from the public release. The `dataset/llm_audited_150/` directory contains only de-identified prescription images with public AMNIN_RX IDs and associated LLM audit results. Access to the full clinical corpus requires a separate governance and data-use process.
