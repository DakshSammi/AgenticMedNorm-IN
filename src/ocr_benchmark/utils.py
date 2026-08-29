"""
Pipeline utilities: image discovery, output I/O, logging setup.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from PIL import Image

from .config import IMAGE_EXTENSIONS, LOG_FILE, LOG_LEVEL, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Prescription input abstraction
# ---------------------------------------------------------------------------

@dataclass
class PrescriptionInput:
    """
    Represents one patient prescription — either a single image or a
    multi-page folder containing multiple images.
    """
    patient_id: str               # e.g. "p1", "p25"
    image_paths: list[Path]       # ordered list of page images
    is_multi_page: bool = False
    source_dir: Path = field(default_factory=Path)

    @property
    def primary_image_name(self) -> Union[str, list[str]]:
        if self.is_multi_page:
            return [p.name for p in self.image_paths]
        return self.image_paths[0].name if self.image_paths else ""

    @property
    def page_numbers(self) -> Union[int, list[int]]:
        if self.is_multi_page:
            return list(range(1, len(self.image_paths) + 1))
        return 1

    @property
    def total_pages(self) -> int:
        return len(self.image_paths)


# ---------------------------------------------------------------------------
# Prescription discovery
# ---------------------------------------------------------------------------

def discover_prescriptions(prescriptions_dir: Path) -> list[PrescriptionInput]:
    """
    Scan the prescriptions directory and return a list of PrescriptionInput
    objects, handling both:
        - Single-image files  (e.g. p1.jpeg → PrescriptionInput("p1", [...]))
        - Multi-page folders  (e.g. p25/   → PrescriptionInput("p25", [...]))

    Folders are detected by containing image files directly (not nested).
    Images are sorted by name for consistent page ordering.
    """
    prescriptions: list[PrescriptionInput] = []

    if not prescriptions_dir.exists():
        logging.getLogger(__name__).error(
            "Prescriptions directory not found: %s", prescriptions_dir
        )
        return prescriptions

    for entry in sorted(prescriptions_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            # Single-page prescription
            patient_id = entry.stem  # "p1" from "p1.jpeg"
            prescriptions.append(
                PrescriptionInput(
                    patient_id=patient_id,
                    image_paths=[entry],
                    is_multi_page=False,
                    source_dir=prescriptions_dir,
                )
            )

        elif entry.is_dir():
            # Multi-page prescription folder
            images = sorted(
                [p for p in entry.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            )
            if images:
                patient_id = entry.name  # "p25" from folder name
                prescriptions.append(
                    PrescriptionInput(
                        patient_id=patient_id,
                        image_paths=images,
                        is_multi_page=True,
                        source_dir=entry,
                    )
                )
            else:
                logging.getLogger(__name__).warning(
                    "Folder %s contains no image files — skipping.", entry
                )

    logging.getLogger(__name__).info(
        "Discovered %d prescriptions in %s", len(prescriptions), prescriptions_dir
    )
    return prescriptions


# ---------------------------------------------------------------------------
# Output I/O
# ---------------------------------------------------------------------------

def get_output_path(engine_name: str, patient_id: str) -> Path:
    """Return the output JSON path for a given engine + patient."""
    output_dir = OUTPUT_DIR / engine_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{patient_id}.json"


def save_output_json(output_dict: dict, engine_name: str, patient_id: str) -> Path:
    """
    Serialise and save an OCROutput dict to the standard output path.

    Args:
        output_dict: The .to_dict() result from an OCROutput instance.
        engine_name: Name of the OCR engine (becomes the sub-folder name).
        patient_id: Patient identifier (becomes the JSON file name).

    Returns:
        Path to the saved JSON file.
    """
    out_path = get_output_path(engine_name, patient_id)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    return out_path


def load_output_json(engine_name: str, patient_id: str) -> dict | None:
    """Load a previously saved output JSON, or None if not found."""
    path = get_output_path(engine_name, patient_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_pil_image(image_path: Path) -> Image.Image:
    """Load an image as PIL RGB, raising IOError on failure."""
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as exc:
        raise IOError(f"Cannot open image: {image_path}") from exc


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: Path = LOG_FILE, level: str = LOG_LEVEL) -> None:
    """
    Configure root logger with:
        - Rotating file handler → logs/pipeline.log
        - Stream handler → stdout

    Call once at pipeline startup.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler (10 MB, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    # Quieten noisy third-party loggers
    for noisy in ("PIL", "urllib3", "boto3", "botocore", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
