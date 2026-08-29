"""
AWS Textract OCR engine wrapper.

Uses DetectDocumentText API via boto3.

Setup:
    1. Create an IAM user with AmazonTextractFullAccess policy.
    2. Set environment variables:
           AWS_ACCESS_KEY_ID=<your_key_id>
           AWS_SECRET_ACCESS_KEY=<your_secret>
           AWS_REGION=ap-south-1   (or your preferred region)

Credentials are NEVER hardcoded — always read from environment.

Notes:
    - DetectDocumentText accepts images ≤ 5 MB as raw bytes or S3 reference.
    - Images > 5 MB are automatically resized before sending.
    - Returns WORD and LINE blocks with geometry (normalised to [0,1]).
    - Multi-page prescriptions are processed page-by-page.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image

from ..config import AWS_ACCESS_KEY_ID, AWS_REGION, AWS_SECRET_ACCESS_KEY
from .base_engine import BaseOCREngine, EngineResult, LineBox, WordBox

logger = logging.getLogger(__name__)

# Textract maximum image size for inline bytes (5 MB)
_MAX_BYTES = 5 * 1024 * 1024
# Maximum dimension to resize to before sending (keeps aspect ratio)
_MAX_DIM = 4096


def _image_to_bytes(image_path: Path) -> bytes:
    """
    Read image as JPEG bytes, resizing if > 5 MB.
    Textract requires JPEG, PNG, TIFF, or PDF.
    """
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()

    if len(raw) <= _MAX_BYTES:
        return raw

    # Resize down until under size limit
    w, h = img.size
    scale = 0.9
    while len(raw) > _MAX_BYTES and min(w, h) > 200:
        w = int(w * scale)
        h = int(h * scale)
        resized = img.resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        raw = buf.getvalue()
        scale *= 0.9

    return raw


def _geometry_to_box(geometry: dict) -> dict:
    """Extract BoundingBox dict from Textract geometry."""
    bb = geometry.get("BoundingBox", {})
    return {
        "left": bb.get("Left"),
        "top": bb.get("Top"),
        "width": bb.get("Width"),
        "height": bb.get("Height"),
    }


class AWSTextractEngine(BaseOCREngine):
    """
    AWS Textract DetectDocumentText engine.

    Credentials sourced from environment variables only.
    """

    name = "aws_textract"
    supports_bounding_boxes = True

    def is_available(self) -> bool:
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            self.logger.warning(
                "AWS credentials not set. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars."
            )
            return False
        try:
            import boto3  # noqa: F401
            return True
        except ImportError:
            self.logger.warning("boto3 not installed. Run: pip install boto3")
            return False

    def _get_client(self):
        import boto3
        return boto3.client(
            "textract",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )

    def _process_single_image(
        self,
        client,
        image_path: Path,
        page_num: int,
    ) -> tuple[str, list[WordBox], list[LineBox]]:
        """
        Call DetectDocumentText on one image, return
        (page_text, word_boxes, line_boxes).
        """
        image_bytes = _image_to_bytes(image_path)
        response = client.detect_document_text(
            Document={"Bytes": image_bytes}
        )

        words: list[WordBox] = []
        lines: list[LineBox] = []
        line_texts: list[str] = []

        for block in response.get("Blocks", []):
            block_type = block.get("BlockType", "")
            text = block.get("Text", "")
            conf = block.get("Confidence")
            conf_norm = (conf / 100.0) if conf is not None else None
            geo = _geometry_to_box(block.get("Geometry", {}))

            if block_type == "WORD":
                words.append(WordBox(
                    text=text,
                    confidence=conf_norm,
                    left=geo["left"],
                    top=geo["top"],
                    width=geo["width"],
                    height=geo["height"],
                    page=page_num,
                ))
            elif block_type == "LINE":
                lines.append(LineBox(
                    text=text,
                    confidence=conf_norm,
                    left=geo["left"],
                    top=geo["top"],
                    width=geo["width"],
                    height=geo["height"],
                    page=page_num,
                ))
                line_texts.append(text)

        page_text = "\n".join(line_texts)
        return page_text, words, lines

    def run(
        self,
        image_paths: list[Path],
        preprocessed_arrays=None,
    ) -> EngineResult:
        client = self._get_client()

        all_words: list[WordBox] = []
        all_lines: list[LineBox] = []
        page_texts: list[str] = []
        errors: list[str] = []
        confidences: list[float] = []

        for i, img_path in enumerate(image_paths, start=1):
            try:
                self.logger.debug("Textract: processing page %d — %s", i, img_path.name)
                page_text, words, lines = self._process_single_image(
                    client, img_path, i
                )
                page_texts.append(page_text)
                all_words.extend(words)
                all_lines.extend(lines)

                # Collect confidence scores
                for w in words:
                    if w.confidence is not None:
                        confidences.append(w.confidence)

            except Exception as exc:  # noqa: BLE001
                err = f"Page {i} ({img_path.name}): {type(exc).__name__}: {exc}"
                errors.append(err)
                self.logger.error("Textract error: %s", err)
                page_texts.append("")

        full_text = "\n\n--- PAGE BREAK ---\n\n".join(
            t for t in page_texts if t
        )
        avg_conf = (sum(confidences) / len(confidences)) if confidences else None

        try:
            import boto3
            version = boto3.__version__
        except Exception:
            version = "unknown"

        return EngineResult(
            full_text=full_text,
            pages=page_texts,
            words=all_words,
            lines=all_lines,
            engine_version=version,
            overall_confidence=avg_conf,
            errors=errors,
            model_id="aws.textract.detect_document_text",
        )
