"""
OCR engine sub-package.

Each engine is a standalone module that:
    1. Imports only when actually used (heavy model loading is deferred).
    2. Exposes `is_available()` to allow graceful skipping.
    3. Returns a standard dict understood by the pipeline runner.

Import map:
    aws_textract    → AWSTextractEngine
    google_mlkit    → MLKitIngestorEngine
    paddleocr       → PaddleOCREngine
    doctr           → DocTREngine
    trocr           → TrOCREngine
    surya           → SuryaEngine
    nougat          → NougatEngine
    donut           → DonutEngine
"""
