# AgenticMedNorm-IN

AgenticMedNorm-IN is a reproducible six-agent pipeline for medication normalization in Indian handwritten prescription workflows. It separates transcription from normalization: transcription records what is visible in a prescription, while normalization resolves medication mentions to supported local products, brand families, ingredients, formulations, RxNorm concepts, ATC mappings where available, and explicit review or NIL outcomes when evidence is insufficient.

## Why This Exists

Medication normalization is not the same problem as optical character recognition. A recognizer may read a surface string such as `Tab A TO Z` or `Syp Mucaine`, but the normalization system still has to decide whether that string can be safely connected to a medicine identity, what evidence supports the connection, and when ambiguity should remain unresolved. Indian prescriptions add extra difficulty through local brand names, fixed-dose combinations, shorthand dosing, spelling variation, and source coverage gaps in global terminologies.

## Architecture

The public release documents the final six-agent core:

1. De-identification Agent
2. Annotation Creation Agent
3. Candidate Retrieval Agent
4. Candidate Ranking Agent
5. Evidence Assessment Agent
6. Verification Agent

Candidate retrieval uses five branches:

- `R1_EXACT_FUZZY`: exact and fuzzy local surface matching
- `R2_BM25`: lexical retrieval over canonical resource text
- `R3_BIOMEDICAL_DENSE`: biomedical dense retrieval, using SapBERT where cached/available
- `R4_RXNORM`: RxNorm/RxNav ingredient and terminology lookup
- `R5_INDIA_KB`: India-specific structured resources

Ranking uses the true candidate union followed by unweighted reciprocal rank fusion with `k=60`, returning Top-K candidates for evidence assessment.

The LLM judge used in evaluation studies is not part of the six-agent production core.

## Knowledge Resources

Operational/frozen sources are documented in [knowledge_resources.md](docs/knowledge_resources.md). The intended resource set includes NPPA Pharma Sahi Daam evidence, CDSCO approved-drug/FDC resources, NLEM 2022, the open Indian medicine dataset, RxNorm/RxNav, and supported ATC/RxClass mappings where available. Experimental or non-operational resources are documented separately and must not be described as implemented in the paper unless supported by release manifests.

## Annotation Model

The Annotation Creation Agent supports GPT-5.5 structured visual annotation for the private full pipeline. The public repository is also reproducible without paid API execution by using precomputed annotation JSONs through `--annotations-dir`. API credentials are never required for the synthetic smoke test.

See [annotation_benchmark.md](docs/annotation_benchmark.md) for the audited model-selection rationale.

## Dataset And Evaluation

The private clinical corpus currently present on the server contains `893` p-numbered raw prescription images with matching ground-truth JSONs and anonymized images. These patient-level files are excluded from the public release by default. Earlier `737` and `867` development snapshots are historical and should not be mixed with the current `893` corpus in final paper claims.

Public examples under `data/examples/` are synthetic and non-sensitive. They are intended for smoke testing only, not for reporting scientific performance.

## Installation

Use Python 3.11 or newer. GPU is optional for the public smoke test; full dense retrieval with SapBERT/FAISS may benefit from GPU acceleration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` only if you need optional online services. The public smoke test does not need API keys.

## Quick Start

Run the synthetic end-to-end pipeline in precomputed-annotation mode:

```bash
python scripts/run_pipeline.py \
  --input data/examples \
  --annotations-dir data/examples/annotations \
  --config configs/examples/synthetic_pipeline_config.json \
  --output outputs/example_run \
  --resume
```

Preview planned stages without writing outputs:

```bash
python scripts/run_pipeline.py \
  --input data/examples \
  --annotations-dir data/examples/annotations \
  --config configs/examples/synthetic_pipeline_config.json \
  --output outputs/example_run \
  --dry-run
```

## Full Reproduction

See [reproducibility.md](docs/reproducibility.md) for commands covering resource construction, pipeline execution, benchmark table reconstruction, figure-data generation, and expert-validation metric generation.

## Repository Structure

- `src/`: pipeline agents, schemas, adapters, retrieval, ranking, evidence, verification, and evaluation code
- `scripts/`: release-safe orchestration and utility entry points
- `configs/`: frozen and example configuration
- `docs/`: architecture, dataset, benchmark, evaluation, release, and reproducibility documentation
- `data/examples/`: synthetic public examples
- `rebuild/reports/`: internal execution reports, some of which may be historical
- `derived/`, `generated/`, `outputs/`, `state/`, `logs/`: runtime or private/generated artifacts, excluded from public release where appropriate

## Expected Outputs

The public smoke runner writes:

- `layer_a_medication_mentions.csv`
- `candidate_union.csv`
- `ranked_candidates.csv`
- `evidence_assessments.csv`
- `verification_results.csv`
- `layer_b.csv`
- `evaluation_export.json`

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). No DOI is claimed in this repository.

## License

`LICENSE_REQUIRED`: no license file was present in the working tree during release preparation, so no public open-source license has been invented. Add the correct license before treating the GitHub repository as reusable by third parties.

## Privacy And Data Availability

Real prescription images, de-identified prescription images, and ground-truth JSONs containing patient-level clinical content are private by default and are ignored for public release. The public repository includes synthetic examples only. Access to the full clinical corpus requires a separate governance and data-use process.

## Limitations

This repository does not claim that every local brand or FDC can be resolved automatically. Verification may return `HUMAN_REVIEW` or `NIL` when evidence is missing, conflicting, or only source-record level. Dense retrieval, RxNorm/RxNav, and optional local-judge components may require separately downloaded resources.
