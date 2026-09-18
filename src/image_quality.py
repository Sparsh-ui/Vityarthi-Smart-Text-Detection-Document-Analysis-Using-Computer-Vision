"""
Image Quality Analysis Module for Smart Text Detection.

Calculates basic image-quality indicators that are useful for
understanding OCR performance. These metrics are practical heuristics,
not scientifically validated universal quality scores.

Metrics computed:
    - Brightness: mean pixel intensity (0-255)
    - Contrast: standard deviation of pixel intensities
    - Sharpness: variance of the Laplacian (higher = sharper)
    - Dimensions: width and height in pixels
    - Grayscale statistics: min, max, mean pixel values
"""

import cv2
import numpy as np
from typing import Dict


def _compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity as a brightness indicator."""
    return round(float(np.mean(gray)), 2)


def _compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of pixel intensities as a contrast indicator."""
    return round(float(np.std(gray)), 2)


def _compute_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian as a sharpness/blur indicator.

    Higher values indicate sharper images; very low values suggest blur.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return round(float(laplacian.var()), 2)


def _interpret_quality(brightness: float, contrast: float,
                       sharpness: float) -> str:
    """Provide a simple qualitative interpretation of the metrics.

    This is a rough heuristic, not a rigorous quality classification.

    Rules:
        - Brightness outside 50-200 → likely too dark or washed out
        - Contrast below 30 → low contrast, text may be hard to distinguish
        - Sharpness below 50 → likely blurry

    Returns:
        'Good', 'Moderate', or 'Poor'
    """
    issues = 0

    if brightness < 50 or brightness > 200:
        issues += 1
    if contrast < 30:
        issues += 1
    if sharpness < 50:
        issues += 1

    if issues == 0:
        return "Good"
    elif issues == 1:
        return "Moderate"
    else:
        return "Poor"


def assess_quality(image: np.ndarray) -> Dict:
    """Assess basic quality metrics of an image.

    Args:
        image: Input image (BGR or grayscale NumPy array).

    Returns:
        Dictionary containing:
            - brightness: float (0-255)
            - contrast: float
            - sharpness: float (Laplacian variance)
            - width: int
            - height: int
            - channels: int
            - pixel_min: int
            - pixel_max: int
            - interpretation: str ('Good', 'Moderate', or 'Poor')
            - note: str (disclaimer about the metrics)
    """
    # Convert to grayscale for metric calculation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = image.shape[2]
    else:
        gray = image
        channels = 1

    h, w = image.shape[:2]

    brightness = _compute_brightness(gray)
    contrast = _compute_contrast(gray)
    sharpness = _compute_sharpness(gray)
    interpretation = _interpret_quality(brightness, contrast, sharpness)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "pixel_min": int(np.min(gray)),
        "pixel_max": int(np.max(gray)),
        "interpretation": interpretation,
        "note": (
            "These are basic image-quality indicators, not scientifically "
            "validated quality scores. They provide a rough sense of whether "
            "the image is suitable for OCR processing."
        ),
    }
