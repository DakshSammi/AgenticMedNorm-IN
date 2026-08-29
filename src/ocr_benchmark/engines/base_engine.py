"""
Abstract base class for all OCR engine wrappers.

Every engine must implement:
    name        : str   — unique engine identifier (matches output folder name)
    run()       : method that accepts a list of image Paths and returns an
                  EngineResult dict
    is_available(): method that checks if the engine's dependencies and
                    credentials are present
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WordBox:
    """Word-level OCR result with optional bounding box."""
    text: str
    confidence: Optional[float] = None
    left: Optional[float] = None
    top: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    polygon: Optional[list[list[float]]] = None
    page: int = 1


@dataclass
class LineBox:
    """Line-level OCR result with optional bounding box."""
    text: str
    confidence: Optional[float] = None
    left: Optional[float] = None
    top: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    polygon: Optional[list[list[float]]] = None
    page: int = 1


@dataclass
class EngineResult:
    """
    Standardised return value from every engine's run() method.

    Attributes:
        full_text           : Concatenation of all pages' text.
        pages               : Per-page raw text strings.
        words               : Word-level boxes (if supported).
        lines               : Line-level boxes (if supported).
        engine_version      : Optional version string.
        overall_confidence  : Optional mean confidence [0.0-1.0].
        errors              : List of non-fatal error messages.
        warnings            : List of non-fatal warnings.
        model_id            : HuggingFace model ID or API endpoint used.
        processing_time_seconds: Total time taken for OCR in seconds.
    """
    full_text: str = ""
    pages: list[str] = field(default_factory=list)
    words: list[WordBox] = field(default_factory=list)
    lines: list[LineBox] = field(default_factory=list)
    engine_version: str = ""
    overall_confidence: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_id: str = ""
    processing_time_seconds: Optional[float] = None


class BaseOCREngine(ABC):
    """
    Abstract base for all OCR engine wrappers.

    Subclasses must set `name` as a class attribute and implement
    `run()` and `is_available()`.
    """

    #: Unique identifier — used as the output subfolder name.
    name: str = "base"

    #: Whether the engine supports returning bounding boxes.
    supports_bounding_boxes: bool = False

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"ocr.{self.name}")

    @abstractmethod
    def is_available(self) -> bool:
        """
        Return True if the engine's dependencies and credentials are
        available. Called before run() — if False, the pipeline skips
        this engine with a warning log.
        """

    @abstractmethod
    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        """
        Run OCR on a list of image paths (one per page).

        Args:
            image_paths        : Ordered list of page image Paths.
            preprocessed_arrays: Optional list of numpy arrays if
                                 preprocessing was already applied.
                                 If None, engines load from image_paths.

        Returns:
            EngineResult with at minimum `full_text` and `pages` populated.
        """

    def safe_run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        """
        Wrapper around run() that catches all exceptions and returns a
        partial EngineResult with the error recorded — never raises.
        """
        try:
            self.logger.info(
                "Starting OCR on %d image(s): %s",
                len(image_paths),
                [p.name for p in image_paths],
            )
            result = self.run(image_paths, preprocessed_arrays)
            self.logger.info("OCR complete — extracted %d chars", len(result.full_text))
            return result
        except Exception as exc:  # noqa: BLE001
            error_msg = f"{type(exc).__name__}: {exc}"
            self.logger.exception("Engine %s failed: %s", self.name, error_msg)
            return EngineResult(errors=[error_msg])
