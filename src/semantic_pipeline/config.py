"""
Configuration for the semantic enrichment pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# API Credentials (read from env â€” NEVER hardcode)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN", ""))
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY", "")

# Directory Constants
DATA_DIR = _PROJECT_ROOT / "data"
OCR_OUTPUT_DIR = DATA_DIR / "outputs"
SEMANTIC_OUTPUT_DIR = DATA_DIR / "semantic_outputs"
NORMALIZED_DIR = DATA_DIR / "normalized"
ENHANCED_OUTPUT_DIR = DATA_DIR / "enhanced_outputs"
SEMANTIC_EVAL_DIR = DATA_DIR / "semantic_evaluation"
GROUND_TRUTH_DIR = DATA_DIR / "raw_ground_truths"

# Create directories
for d in [SEMANTIC_OUTPUT_DIR, NORMALIZED_DIR, ENHANCED_OUTPUT_DIR, SEMANTIC_EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# NER Settings
# Default model for scispaCy
SCISPACY_MODEL = os.getenv("SCISPACY_MODEL", "en_core_sci_sm")
# PubMedBERT / SciBERT
TRANSFORMER_BIOMEDICAL_MODEL = os.getenv("TRANSFORMER_BIOMEDICAL_MODEL", "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# Ontology Mapping Settings
ABEROWL_API_URL = "http://aber-owl.net/api/"
BIOPORTAL_API_URL = "http://data.bioontology.org"
BODHI_RESOURCES_DIR = DATA_DIR / "ontologies" / "bodhi"
BODHI_DICTIONARY_PATH = BODHI_RESOURCES_DIR / "indian_medical_terms.json"

# Ensure Bodhi directory exists
BODHI_RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

# LLM Normalization (Optional)
USE_LLM_NORMALIZATION = os.getenv("USE_LLM_NORMALIZATION", "false").lower() == "true"
MEDGEMMA_MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-2b")
