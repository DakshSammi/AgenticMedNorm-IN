"""
Central configuration for the OCR benchmarking pipeline.

All paths, engine toggles, and preprocessing flags are defined here.
Credentials are read from environment variables (never hardcoded).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
BASE_DIR: Path = _PROJECT_ROOT
DATA_DIR: Path = BASE_DIR / "data"
PRESCRIPTIONS_DIR: Path = DATA_DIR / "prescriptions"
GROUND_TRUTH_DIR: Path = DATA_DIR / "raw_ground_truths"
OUTPUT_DIR: Path = DATA_DIR / "outputs"
LOG_DIR: Path = BASE_DIR / "logs"
BASELINE_EVAL_DIR: Path = DATA_DIR / "baseline_evaluation"

# Ensure directories exist
for d in [OUTPUT_DIR, LOG_DIR, BASELINE_EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ML Kit pre-exported JSON directory (populated by the Android app)
MLKIT_EXPORT_DIR: Path = Path(
    os.getenv("GOOGLE_MLKIT_EXPORT_DIR", str(DATA_DIR / "mlkit_exports"))
)

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------
# All engines that the pipeline will attempt to run (in order).
# Engines whose credentials / packages are unavailable will be skipped
# gracefully at runtime rather than crashing the whole pipeline.
ALL_ENGINES = [
    "aws_textract",
    "google_mlkit",
    "paddleocr",
    "doctr",
    "trocr",
    "surya",
    "easyocr",
    "tesseract",
    "nougat",
    "donut",
    "llama_cpp",
    "gemini",
    "ollama",
    "qwen_vl",
    "deepseek_vl",
    "florence",
    "moondream",
]

# Override via env: OCR_ENGINES=paddleocr,doctr
_engines_env = os.getenv("OCR_ENGINES", "").strip()
ENABLED_ENGINES: list[str] = (
    [e.strip() for e in _engines_env.split(",") if e.strip()]
    if _engines_env
    else ALL_ENGINES
)

# ---------------------------------------------------------------------------
# Preprocessing flags
# ---------------------------------------------------------------------------
_PREPROCESSING_ENABLED = os.getenv("OCR_PREPROCESSING_ENABLED", "true").lower() == "true"

PREPROCESSING: dict[str, bool] = {
    "enabled": _PREPROCESSING_ENABLED,
    "grayscale": True,
    "denoise": True,
    "adaptive_threshold": False,   # can over-sharpen; off by default
    "contrast_enhancement": True,  # CLAHE
    "deskew": True,
}

# ---------------------------------------------------------------------------
# API Credentials (read from env — NEVER hardcode)
# ---------------------------------------------------------------------------
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Model identifiers & Paths
# ---------------------------------------------------------------------------
TROCR_MODEL_ID: str = os.getenv("TROCR_MODEL_ID", "microsoft/trocr-large-handwritten")
NOUGAT_MODEL_ID: str = os.getenv("NOUGAT_MODEL_ID", "facebook/nougat-base")
DONUT_MODEL_ID: str = os.getenv(
    "DONUT_MODEL_ID", "naver-clova-ix/donut-base-finetuned-cord-v2"
)

# Llama.cpp VLM settings (GGUF)
LLAMA_CPP_MODEL_PATH: str = os.getenv("LLAMA_CPP_MODEL_PATH", "")
LLAMA_CPP_CLIP_PATH: str = os.getenv("LLAMA_CPP_CLIP_PATH", "")

# Ollama settings
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_ID: str = os.getenv("OLLAMA_MODEL_ID", "llama3-vision")

# Qwen-VL (HuggingFace)
QWEN_VL_MODEL_ID: str = os.getenv("QWEN_VL_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")

# DeepSeek-VL (HuggingFace)
DEEPSEEK_VL_MODEL_ID: str = os.getenv("DEEPSEEK_VL_MODEL_ID", "deepseek-ai/deepseek-vl-1.3b-chat")

# Florence-2 (HuggingFace - Ultra fast)
FLORENCE_MODEL_ID: str = os.getenv("FLORENCE_MODEL_ID", "microsoft/Florence-2-base")

# Moondream (HuggingFace - Lightweight)
MOONDREAM_MODEL_ID: str = os.getenv("MOONDREAM_MODEL_ID", "vikhyatk/moondream2")

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE: Path = LOG_DIR / "pipeline.log"
LOG_LEVEL: str = os.getenv("OCR_LOG_LEVEL", "INFO")
