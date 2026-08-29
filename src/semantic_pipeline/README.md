# Semantic Enrichment & Ontology Mapping Pipeline

This module implements the second stage of the prescription understanding pipeline. It takes raw OCR outputs and transforms them into semantically enriched, standardized data.

## Architecture

The pipeline is organized into several modular layers:

1.  **Normalization Layer (`normalization/`)**: 
    - Cleans OCR noise and expands medical abbreviations (e.g., "TDS" -> "three times daily").
    - Rule-based logic specialized for Indian medical prescriptions.
2.  **Biomedical NER Layer (`ner/`)**: 
    - Extracts clinical entities using **scispaCy** (default) or **Transformers** (PubMedBERT).
    - Pluggable backend allows for easy model swaps.
3.  **Ontology Mapping Layer (`ontology_mapping/`)**: 
    - Maps extracted terms to **SNOMED CT**, **RxNorm**, and **ICD-10** using the **AberOWL API**.
    - Supports fuzzy matching for local dictionaries.
4.  **Benchmarking Layer (`benchmarking/`)**: 
    - Compares enriched outputs against ground truth JSONs.
    - Computes Precision, Recall, and F1-Score for semantic entities.
5.  **Visualization Layer (`visualization/`)**: 
    - Generates performance charts and precision-recall plots.

## Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Install scispaCy model:
    ```bash
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
    ```

## Usage

Run the full semantic pipeline:
```bash
python -m src.semantic_pipeline.run_semantic_pipeline
```

The outputs will be saved in:
- `data/normalized/`: Normalized JSONs (rule-based cleaning).
- `data/enhanced_outputs/`: Full semantically enriched JSONs.
- `data/semantic_outputs/benchmarks/`: Evaluation reports and charts.

## Configuration

Settings can be adjusted in `src/semantic_pipeline/config.py` or via environment variables in `.env`.

- `SCISPACY_MODEL`: Choose a different scispaCy model (e.g., `en_core_sci_lg`).
- `USE_LLM_NORMALIZATION`: Enable LLM-based refinement (requires additional setup).
- `TRANSFORMER_BIOMEDICAL_MODEL`: Specify a HuggingFace model for NER.

## Future Plans

- **FHIR Conversion**: Mapping enriched entities to FHIR resources.
- **Knowledge Graph Integration**: Loading entities into a GraphDB.
- **Temporal Reasoning**: Normalizing durations and review dates into absolute timestamps.
