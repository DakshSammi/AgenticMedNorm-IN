"""
Entry point for the OCR benchmarking pipeline.

Usage:
    python src/run_pipeline.py [options]

    Options:
        --engines paddleocr,doctr   Run specific engines only
        --patient p1,p25            Process specific patients only
        --no-preprocessing          Skip image preprocessing
        --log-level DEBUG           Set logging verbosity

    Run `python src/run_pipeline.py --help` for full usage.
"""

import sys
from pathlib import Path

# Ensure src/ is on the Python path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from ocr_benchmark.runner import main

if __name__ == "__main__":
    main()
