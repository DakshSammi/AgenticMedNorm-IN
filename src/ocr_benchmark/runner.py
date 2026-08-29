"""
Main OCR benchmarking pipeline runner.

Entry point: python src/run_pipeline.py
             python src/ocr_benchmark/runner.py

Pipeline flow for each prescription:
    1. Discover all images (single-page files + multi-page folders)
    2. Apply preprocessing (if enabled)
    3. For each enabled engine:
        a. Check engine availability
        b. Run OCR → EngineResult
        c. Extract structured fields from raw text
        d. Build OCROutput schema object
        e. Save JSON to data/outputs/<engine>/<patient_id>.json
        f. Log result

CLI:
    python src/run_pipeline.py [--engines paddleocr,doctr] [--patient p1,p25]
                               [--no-preprocessing] [--log-level DEBUG]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .config import (
    ENABLED_ENGINES,
    PRESCRIPTIONS_DIR,
    PREPROCESSING,
    OUTPUT_DIR,
    BASELINE_EVAL_DIR,
)
from .engines.base_engine import EngineResult
from .parsers.field_extractor import extract_all_fields, extract_hospital_header, extract_sections_detected
from .preprocessing import preprocess_images
from .schema import (
    BoundingBox,
    DocumentLayout,
    DocumentMetadata,
    EngineMetadata,
    EngineMetadata,
    LineCoordinate,
    OCRCoordinates,
    OCROutput,
    RawEntities,
    RawText,
    WordCoordinate,
)
from .utils import (
    PrescriptionInput,
    discover_prescriptions,
    save_output_json,
    setup_logging,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine registry — maps name → import factory
# ---------------------------------------------------------------------------

def _load_engine(name: str):
    """Lazily import and instantiate the requested engine."""
    if name == "aws_textract":
        from .engines.aws_textract import AWSTextractEngine
        return AWSTextractEngine()
    if name == "google_mlkit":
        from .engines.mlkit_ingestor import MLKitIngestorEngine
        return MLKitIngestorEngine()
    if name == "paddleocr":
        from .engines.paddleocr_engine import PaddleOCREngine
        return PaddleOCREngine()
    if name == "doctr":
        from .engines.doctr_engine import DocTREngine
        return DocTREngine()
    if name == "trocr":
        from .engines.trocr_engine import TrOCREngine
        return TrOCREngine()
    if name == "surya":
        from .engines.surya_engine import SuryaEngine
        return SuryaEngine()
    if name == "easyocr":
        from .engines.easyocr_engine import EasyOCREngine
        return EasyOCREngine()
    if name == "tesseract":
        from .engines.tesseract_engine import TesseractEngine
        return TesseractEngine()
    if name == "nougat":
        from .engines.nougat_engine import NougatEngine
        return NougatEngine()
    if name == "donut":
        from .engines.donut_engine import DonutEngine
        return DonutEngine()
    if name == "llama_cpp":
        from .engines.llama_cpp_engine import LlamaCppEngine
        return LlamaCppEngine()
    if name == "gemini":
        from .engines.gemini_engine import GeminiEngine
        return GeminiEngine()
    if name == "ollama":
        from .engines.ollama_engine import OllamaEngine
        return OllamaEngine()
    if name == "qwen_vl":
        from .engines.transformers_vlm_engine import TransformersVLMEngine
        return TransformersVLMEngine()
    if name == "deepseek_vl":
        from .engines.deepseek_engine import DeepSeekVLEngine
        return DeepSeekVLEngine()
    if name == "florence":
        from .engines.florence_engine import FlorenceEngine
        return FlorenceEngine()
    if name == "moondream":
        from .engines.moondream_engine import MoondreamEngine
        return MoondreamEngine()
    raise ValueError(f"Unknown engine: {name}")


# ---------------------------------------------------------------------------
# Result → OCROutput schema builder
# ---------------------------------------------------------------------------

def _build_ocr_output(
    prescription: PrescriptionInput,
    engine_name: str,
    engine_result: EngineResult,
    preprocessing_applied: list[str],
    processing_time: float,
    supports_bb: bool,
) -> OCROutput:
    """Assemble a fully-populated OCROutput from an EngineResult."""

    raw_text = engine_result.full_text or ""

    # --- Document metadata ---
    meta = DocumentMetadata(
        document_id=prescription.patient_id,
        source_type="prescription",
        language=[],
        source_image=prescription.primary_image_name,
        page_number=prescription.page_numbers,
        total_pages=prescription.total_pages,
        ocr_engine=engine_name,
    )

    # --- Layout ---
    layout = DocumentLayout(
        hospital_header=extract_hospital_header(raw_text),
        sections_detected=extract_sections_detected(raw_text),
    )

    # --- Raw text ---
    raw = RawText(
        full_text=raw_text,
        pages=engine_result.pages,
    )

    # --- Structured entities ---
    extracted = extract_all_fields(raw_text)
    entities = RawEntities(**extracted) if extracted else RawEntities()

    # --- Bounding boxes ---
    words_coords = [
        WordCoordinate(
            text=w.text,
            confidence=w.confidence,
            bounding_box=BoundingBox(
                left=w.left,
                top=w.top,
                width=w.width,
                height=w.height,
                polygon=w.polygon,
            ),
            page=w.page,
        )
        for w in engine_result.words
    ]
    lines_coords = [
        LineCoordinate(
            text=ln.text,
            confidence=ln.confidence,
            bounding_box=BoundingBox(
                left=ln.left,
                top=ln.top,
                width=ln.width,
                height=ln.height,
                polygon=ln.polygon,
            ),
            page=ln.page,
        )
        for ln in engine_result.lines
    ]
    coords = OCRCoordinates(words=words_coords, lines=lines_coords)

    # --- Engine metadata ---
    eng_meta = EngineMetadata(
        engine_name=engine_name,
        engine_version=engine_result.engine_version,
        model_id=engine_result.model_id,
        processing_time_seconds=round(processing_time, 3),
        overall_confidence=engine_result.overall_confidence,
        supports_bounding_boxes=supports_bb,
        preprocessing_applied=preprocessing_applied,
        errors=engine_result.errors,
        warnings=engine_result.warnings,
    )

    return OCROutput(
        document_metadata=meta,
        document_layout=layout,
        raw_text=raw,
        raw_entities=entities,
        ocr_coordinates=coords,
        ocr_engine_metadata=eng_meta,
    )


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    engines_to_run: list[str] | None = None,
    patient_filter: list[str] | None = None,
    preprocessing_config: dict | None = None,
    prescriptions_dir: Path = PRESCRIPTIONS_DIR,
) -> dict:
    """
    Run the full OCR benchmarking pipeline.

    Args:
        engines_to_run    : List of engine names to run (default: ENABLED_ENGINES).
        patient_filter    : List of patient IDs to process (default: all).
        preprocessing_config: Preprocessing flags (default: config.PREPROCESSING).
        prescriptions_dir : Override prescriptions directory.

    Returns:
        Summary dict with counts of successes and failures per engine.
    """
    engines_to_run = engines_to_run or ENABLED_ENGINES
    preprocessing_config = preprocessing_config or PREPROCESSING

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover prescriptions
    prescriptions = discover_prescriptions(prescriptions_dir)
    if patient_filter:
        prescriptions = [p for p in prescriptions if p.patient_id in patient_filter]

    if not prescriptions:
        logger.warning("No prescriptions found to process.")
        return {}

    logger.info("=" * 60)
    logger.info("OCR Benchmark Pipeline starting")
    logger.info("Prescriptions: %d", len(prescriptions))
    logger.info("Engines: %s", engines_to_run)
    logger.info("Preprocessing: %s", preprocessing_config)
    logger.info("=" * 60)

    # Build summary tracker
    summary: dict[str, dict] = {
        eng: {"success": 0, "failed": 0, "skipped": 0}
        for eng in engines_to_run
    }

    # Instantiate all engines upfront (checks is_available())
    active_engines = []
    for engine_name in engines_to_run:
        try:
            engine = _load_engine(engine_name)
        except ValueError as exc:
            logger.error("Cannot load engine '%s': %s", engine_name, exc)
            summary[engine_name]["skipped"] += len(prescriptions)
            continue

        if not engine.is_available():
            logger.warning(
                "Engine '%s' is not available — skipping all prescriptions.", engine_name
            )
            summary[engine_name]["skipped"] += len(prescriptions)
            continue

        active_engines.append(engine)
        logger.info("Engine '%s' is available and ready.", engine_name)

    if not active_engines:
        logger.error("No engines available. Exiting.")
        return summary

    # Process each prescription
    for prescription in prescriptions:
        pid = prescription.patient_id
        logger.info("-" * 50)
        logger.info(
            "Processing prescription: %s (%d page(s))",
            pid, prescription.total_pages
        )

        # Apply preprocessing once for all engines
        preprocessing_applied: list[str] = []
        preprocessed_arrays = None

        if preprocessing_config.get("enabled", True):
            try:
                preprocessed = preprocess_images(
                    prescription.image_paths, preprocessing_config
                )
                preprocessed_arrays = [p.numpy_array for p in preprocessed]
                if preprocessed:
                    preprocessing_applied = preprocessed[0].applied_steps
                logger.debug(
                    "Preprocessing applied: %s", preprocessing_applied
                )
            except Exception as exc:
                logger.warning(
                    "Preprocessing failed for %s: %s — using originals.", pid, exc
                )

        # Run each engine
        for engine in active_engines:
            ename = engine.name
            logger.info("  [%s] Starting OCR on %s", ename, pid)
            t_start = time.perf_counter()

            engine_result: EngineResult = engine.safe_run(
                prescription.image_paths, preprocessed_arrays
            )
            elapsed = time.perf_counter() - t_start

            if engine_result.errors and not engine_result.full_text:
                logger.error(
                    "  [%s] FAILED for %s in %.2fs: %s",
                    ename, pid, elapsed, engine_result.errors
                )
                summary[ename]["failed"] += 1
                # Still save a partial output with error info
            else:
                logger.info(
                    "  [%s] Completed %s in %.2fs — %d chars",
                    ename, pid, elapsed, len(engine_result.full_text)
                )
                summary[ename]["success"] += 1

            # Build and save output JSON
            try:
                output = _build_ocr_output(
                    prescription=prescription,
                    engine_name=ename,
                    engine_result=engine_result,
                    preprocessing_applied=preprocessing_applied,
                    processing_time=elapsed,
                    supports_bb=engine.supports_bounding_boxes,
                )
                out_path = save_output_json(output.to_dict(), ename, pid)
                logger.info("  [%s] Saved → %s", ename, out_path)
            except Exception as exc:
                logger.exception(
                    "  [%s] Failed to save output for %s: %s", ename, pid, exc
                )
                summary[ename]["failed"] += 1

    # Final summary
    logger.info("=" * 60)
    logger.info("Pipeline complete. Summary:")
    for eng, counts in summary.items():
        logger.info(
            "  %-20s success=%-4d failed=%-4d skipped=%d",
            eng, counts["success"], counts["failed"], counts["skipped"],
        )
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR Benchmarking Pipeline for Indian Medical Prescriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all engines on all prescriptions
  python src/run_pipeline.py

  # Run only PaddleOCR and docTR
  python src/run_pipeline.py --engines paddleocr,doctr

  # Run on specific patients only
  python src/run_pipeline.py --patient p1,p25

  # Run without preprocessing
  python src/run_pipeline.py --no-preprocessing

  # Debug logging
  python src/run_pipeline.py --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="",
        help="Comma-separated engine names to run (default: all enabled in config)",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default="",
        help="Comma-separated patient IDs to process (default: all)",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable all image preprocessing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run benchmarking against ground truths after processing",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Setup logging first
    setup_logging(level=args.log_level)

    engines = (
        [e.strip() for e in args.engines.split(",") if e.strip()]
        if args.engines
        else None
    )
    patients = (
        [p.strip() for p in args.patient.split(",") if p.strip()]
        if args.patient
        else None
    )

    preprocessing_cfg = dict(PREPROCESSING)
    if args.no_preprocessing:
        preprocessing_cfg["enabled"] = False

    # 1. Run inference
    if not args.evaluate or engines:
        run_pipeline(
            engines_to_run=engines,
            patient_filter=patients,
            preprocessing_config=preprocessing_cfg,
        )

    # 2. Run evaluation
    if args.evaluate:
        from .evaluator import BenchmarkEvaluator
        from .config import ALL_ENGINES, OUTPUT_DIR
        
        logger.info("Starting benchmarking evaluation...")
        evaluator = BenchmarkEvaluator(use_llm_judge=True)
        
        # Determine which engines to evaluate
        engines_to_eval = engines if engines else [e for e in ALL_ENGINES if (OUTPUT_DIR / e).exists()]
        
        report = evaluator.run_all(engines_to_eval)
        evaluator.save_markdown_report(BASELINE_EVAL_DIR / "benchmark_report.md")
        
        with open(BASELINE_EVAL_DIR / "benchmark_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Benchmarking complete. Report saved to {BASELINE_EVAL_DIR}/benchmark_report.md")


if __name__ == "__main__":
    main()
