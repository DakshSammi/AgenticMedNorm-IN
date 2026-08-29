"""
Image preprocessing utilities for the OCR benchmarking pipeline.

All steps are independently toggleable via config.PREPROCESSING flags.
The pipeline applies the same preprocessing to all engines (global mode).

Steps (in order):
    1. Grayscale conversion
    2. Denoising (fastNlMeansDenoising)
    3. Adaptive thresholding (off by default — can over-sharpen)
    4. Contrast enhancement via CLAHE
    5. Deskew (rotation correction)

Returns both a PIL Image (for engines expecting PIL) and numpy array
(for engines expecting OpenCV/numpy).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import NamedTuple

from PIL import Image

logger = logging.getLogger(__name__)


class PreprocessedImage(NamedTuple):
    pil_image: Image.Image
    numpy_array: object  # np.ndarray — typed as object to avoid numpy import at top level
    applied_steps: list[str]


def _load_numpy(image_path: Path):
    """Load an image as a BGR numpy array via OpenCV."""
    import cv2
    import numpy as np
    img = cv2.imread(str(image_path))
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _to_grayscale(img):
    """Convert BGR to grayscale."""
    import cv2
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _denoise(img):
    """Apply Non-Local Means denoising."""
    import cv2
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def _adaptive_threshold(img):
    """Apply Gaussian adaptive thresholding (requires grayscale input)."""
    import cv2
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=10,
    )


def _enhance_contrast(img):
    """Apply CLAHE."""
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return clahe.apply(img)


def _compute_skew_angle(img) -> float:
    """Estimate document skew angle."""
    import cv2
    import numpy as np
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            points.extend(cnt.reshape(-1, 2).tolist())
    if len(points) < 10:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    _, _, angle = cv2.minAreaRect(pts)
    if angle < -45:
        angle = 90 + angle
    return angle


def _deskew(img):
    """Rotate image to correct skew."""
    import cv2
    angle = _compute_skew_angle(img)
    if abs(angle) < 0.5:
        return img
    logger.debug("Deskew: rotating %.2f degrees", angle)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _numpy_to_pil(img) -> Image.Image:
    """Convert a numpy array (BGR or grayscale) to PIL RGB."""
    import cv2
    if len(img.shape) == 2:
        return Image.fromarray(img).convert("RGB")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_image(
    image_path: Path,
    config: dict[str, bool] | None = None,
) -> PreprocessedImage:
    """
    Apply the configured preprocessing pipeline to a single image.

    Args:
        image_path: Path to the source image file.
        config: Preprocessing flags dict (from config.PREPROCESSING).
                If None, only grayscale is applied.

    Returns:
        PreprocessedImage named tuple with pil_image, numpy_array,
        and a list of applied step names.
    """
    if config is None:
        config = {"enabled": True, "grayscale": True}

    if not config.get("enabled", True):
        # Preprocessing disabled — return original
        img = _load_numpy(image_path)
        return PreprocessedImage(
            pil_image=_numpy_to_pil(img),
            numpy_array=img,
            applied_steps=[],
        )

    img = _load_numpy(image_path)
    applied: list[str] = []

    # 1. Grayscale
    if config.get("grayscale", True):
        img = _to_grayscale(img)
        applied.append("grayscale")

    # 2. Denoise
    if config.get("denoise", False):
        try:
            img = _denoise(img)
            applied.append("denoise")
        except Exception as exc:
            logger.warning("Denoising failed: %s", exc)

    # 3. Contrast enhancement
    if config.get("contrast_enhancement", False):
        try:
            img = _enhance_contrast(img)
            applied.append("contrast_enhancement")
        except Exception as exc:
            logger.warning("Contrast enhancement failed: %s", exc)

    # 4. Adaptive threshold (optional; applied after contrast)
    if config.get("adaptive_threshold", False):
        try:
            img = _adaptive_threshold(img)
            applied.append("adaptive_threshold")
        except Exception as exc:
            logger.warning("Adaptive threshold failed: %s", exc)

    # 5. Deskew
    if config.get("deskew", False):
        try:
            img = _deskew(img)
            applied.append("deskew")
        except Exception as exc:
            logger.warning("Deskewing failed: %s", exc)

    pil_img = _numpy_to_pil(img)
    return PreprocessedImage(
        pil_image=pil_img,
        numpy_array=img,
        applied_steps=applied,
    )


def preprocess_images(
    image_paths: list[Path],
    config: dict[str, bool] | None = None,
) -> list[PreprocessedImage]:
    """Apply preprocessing to a list of images (e.g., multi-page prescription)."""
    return [preprocess_image(p, config) for p in image_paths]
