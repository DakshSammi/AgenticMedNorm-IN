"""
OCR Benchmarking Pipeline for Handwritten Indian Medical Prescriptions.

This package provides modular wrappers for multiple OCR engines,
a heuristic field extractor, image preprocessing utilities, and a
pipeline runner that produces standardised raw-extraction JSONs
ready for future evaluation against annotated ground truths.

Engines supported:
    - AWS Textract
    - Google ML Kit (via pre-exported JSONs from Android module)
    - PaddleOCR
    - docTR
    - TrOCR (Microsoft)
    - Surya OCR
    - Nougat (Facebook)
    - Donut (NAVER Clova)
"""

__version__ = "1.0.0"
__author__ = "OCR Benchmark Pipeline"
