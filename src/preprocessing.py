"""
Image Preprocessing Module for Smart Text Detection.

Provides a pipeline of OpenCV-based preprocessing steps optimized for OCR:
grayscale conversion, noise reduction, contrast enhancement, adaptive
thresholding, and resizing. Each function works independently and can be
tested in isolation.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


def load_image(path: str) -> np.ndarray:
    """Load an image from disk and validate that it was read successfully.

    Args:
        path: Path to the image file.

    Returns:
        The loaded image as a BGR NumPy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV cannot decode the image.
    """
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image (corrupt or unsupported format): {path}")

    return img


def resize_if_needed(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    """Resize the image if its largest dimension exceeds max_dimension.

    Maintains aspect ratio. Images smaller than the limit are returned
    unchanged.

    Args:
        image: Input image (BGR or grayscale).
        max_dimension: Maximum allowed width or height in pixels.

    Returns:
        Resized image, or the original if no resize was needed.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dimension:
        return image

    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale.

    If the image is already single-channel, it is returned as-is.

    Args:
        image: Input image.

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Gaussian blur to reduce high-frequency noise.

    A mild blur helps OCR by smoothing out pixel-level noise without
    destroying text edges.

    Args:
        image: Grayscale or BGR image.
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0,
                     tile_size: int = 8) -> np.ndarray:
    """Enhance local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization).

    Works on grayscale images. For BGR input the image is converted to
    grayscale first.

    Args:
        image: Grayscale image.
        clip_limit: CLAHE clip limit (controls contrast amplification).
        tile_size: Grid size for local histogram equalization.

    Returns:
        Contrast-enhanced grayscale image.
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def adaptive_threshold(image: np.ndarray, block_size: int = 11,
                       constant: int = 2) -> np.ndarray:
    """Apply adaptive thresholding to produce a binary image.

    Useful for documents with uneven lighting. Works best on grayscale
    input.

    Args:
        image: Grayscale image.
        block_size: Neighbourhood size for threshold calculation (odd number).
        constant: Constant subtracted from the mean.

    Returns:
        Binary (thresholded) image.
    """
    gray = to_grayscale(image) if len(image.shape) == 3 else image
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, constant
    )


def run_preprocessing_pipeline(image: np.ndarray,
                               apply_threshold: bool = False) -> Dict[str, np.ndarray]:
    """Run the full preprocessing pipeline and return intermediate results.

    Pipeline order:
      1. Resize if needed
      2. Grayscale conversion
      3. Noise reduction
      4. Contrast enhancement (CLAHE)
      5. Adaptive thresholding (optional)

    The function returns a dictionary containing every intermediate stage
    so callers can inspect or use any step.

    Args:
        image: Raw BGR input image.
        apply_threshold: Whether to include adaptive thresholding.

    Returns:
        Dictionary with keys: 'resized', 'grayscale', 'denoised',
        'enhanced', and optionally 'thresholded'. The 'processed' key
        always points to the final stage.
    """
    stages: Dict[str, np.ndarray] = {}

    stages["resized"] = resize_if_needed(image)
    stages["grayscale"] = to_grayscale(stages["resized"])
    stages["denoised"] = reduce_noise(stages["grayscale"])
    stages["enhanced"] = enhance_contrast(stages["denoised"])

    if apply_threshold:
        stages["thresholded"] = adaptive_threshold(stages["enhanced"])
        stages["processed"] = stages["thresholded"]
    else:
        stages["processed"] = stages["enhanced"]

    return stages
