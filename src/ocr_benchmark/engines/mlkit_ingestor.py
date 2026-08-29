"""
Google ML Kit OCR — Python-side ingestor.

ML Kit runs natively on Android. This module reads pre-exported JSON files
produced by the companion Android app (android_mlkit/).

Workflow:
    1. Build and install the Android app (see android_mlkit/).
    2. Load prescription images on the device and run OCR.
    3. Export results to device Downloads/mlkit_exports/<patient_id>.json
    4. Copy those JSON files to GOOGLE_MLKIT_EXPORT_DIR (env var).
    5. Run the pipeline — this ingestor picks them up automatically.

Expected ML Kit export JSON format:
{
    "source_image": "p1.jpeg",
    "text": "<full extracted text>",
    "blocks": [
        {
            "text": "block text",
            "bounding_box": {"top": 0, "left": 0, "width": 100, "height": 50},
            "lines": [
                {
                    "text": "line text",
                    "bounding_box": {"top": 5, "left": 10, "width": 90, "height": 20},
                    "elements": [
                        {"text": "word", "bounding_box": {...}}
                    ]
                }
            ]
        }
    ],
    "image_width": 1080,
    "image_height": 1920
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ..config import MLKIT_EXPORT_DIR
from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)


def _normalise_box(bb: dict, img_w, img_h) -> dict:
    if img_w and img_h and img_w > 0 and img_h > 0:
        return {
            "left": bb.get("left", 0) / img_w,
            "top": bb.get("top", 0) / img_h,
            "width": bb.get("width", 0) / img_w,
            "height": bb.get("height", 0) / img_h,
        }
    return {k: bb.get(k) for k in ("left", "top", "width", "height")}


class MLKitIngestorEngine(BaseOCREngine):
    """Python-side ingestor for Google ML Kit OCR exports."""

    name = "google_mlkit"
    supports_bounding_boxes = True

    def __init__(self, export_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.export_dir = export_dir or MLKIT_EXPORT_DIR

    def is_available(self) -> bool:
        if not self.export_dir.exists():
            self.logger.warning(
                "ML Kit export directory not found: %s. "
                "Run the Android app and copy exports there.",
                self.export_dir,
            )
            return False
        if not list(self.export_dir.glob("*.json")):
            self.logger.warning("No ML Kit export JSONs found in %s.", self.export_dir)
            return False
        return True

    def _find_export_files(self, image_paths: list[Path]) -> list[Optional[Path]]:
        results: list[Optional[Path]] = []
        if len(image_paths) == 1:
            pid = image_paths[0].stem
            c = self.export_dir / f"{pid}.json"
            results.append(c if c.exists() else None)
        else:
            for i, img_path in enumerate(image_paths, start=1):
                pid = img_path.parent.name
                candidates = [
                    self.export_dir / f"{pid}_page{i}.json",
                    self.export_dir / f"{img_path.stem}.json",
                ]
                found = next((c for c in candidates if c.exists()), None)
                results.append(found)
        return results

    def _parse_export(self, export_path: Path, page_num: int):
        with open(export_path, encoding="utf-8") as fh:
            data = json.load(fh)
        full_text = data.get("text", "")
        img_w = data.get("image_width")
        img_h = data.get("image_height")
        words: list[WordBox] = []
        lines: list[LineBox] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                bb = _normalise_box(line.get("bounding_box", {}), img_w, img_h)
                lines.append(LineBox(
                    text=line.get("text", ""), page=page_num, **bb
                ))
                for elem in line.get("elements", []):
                    ebb = _normalise_box(elem.get("bounding_box", {}), img_w, img_h)
                    words.append(WordBox(
                        text=elem.get("text", ""), page=page_num, **ebb
                    ))
        return full_text, words, lines

    def run(self, image_paths: list[Path], preprocessed_arrays=None) -> EngineResult:
        export_files = self._find_export_files(image_paths)
        all_words: list[WordBox] = []
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        for i, (img_path, export_path) in enumerate(zip(image_paths, export_files), 1):
            if export_path is None:
                msg = f"No ML Kit export for page {i} ({img_path.name})"
                warnings.append(msg)
                self.logger.warning(msg)
                page_texts.append("")
                continue
            try:
                pt, ws, ls = self._parse_export(export_path, i)
                page_texts.append(pt)
                all_words.extend(ws)
                all_lines.extend(ls)
            except Exception as exc:
                err = f"Page {i}: {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error(err)
                page_texts.append("")

        return EngineResult(
            full_text="\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t),
            pages=page_texts,
            words=all_words,
            lines=all_lines,
            engine_version="android_mlkit_ingestor_1.0",
            errors=errors,
            warnings=warnings,
            model_id="com.google.mlkit:text-recognition",
        )
