"""
Text Detection Module for Smart Text Detection.

Wraps EasyOCR to provide clean, structured text detection results.
Manages a reusable OCR reader instance to avoid re-downloading or
reloading the model on every call.
"""

import numpy as np
from typing import Dict, List, Optional


# Module-level reader cache to avoid re-creating the heavy model
_reader_cache: Dict[str, object] = {}


def create_reader(languages: Optional[List[str]] = None,
                  use_gpu: bool = False) -> object:
    """Create or retrieve a cached EasyOCR Reader instance.

    The reader is cached by the sorted language list so that repeated
    calls with the same languages reuse the same model.

    Args:
        languages: List of language codes (e.g. ['en']). Defaults to ['en'].
        use_gpu: Whether to use GPU acceleration.

    Returns:
        An easyocr.Reader instance.
    """
    import easyocr

    if languages is None:
        languages = ["en"]

    cache_key = ",".join(sorted(languages))

    if cache_key not in _reader_cache:
        _reader_cache[cache_key] = easyocr.Reader(languages, gpu=use_gpu)

    return _reader_cache[cache_key]


def detect_text(reader: object, image: np.ndarray,
                threshold: float = 0.25) -> List[Dict]:
    """Detect and recognize text in an image using EasyOCR.

    Args:
        reader: An EasyOCR Reader instance (from create_reader).
        image: Input image as a NumPy array (BGR or grayscale).
        threshold: Minimum confidence score to include a detection.

    Returns:
        A list of detection dictionaries, each containing:
            - 'bbox': list of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - 'text': recognized text string
            - 'confidence': float confidence score (0-1)

        Only detections with confidence >= threshold are returned.
        Results are sorted top-to-bottom by the y-coordinate of the
        first bounding box corner.
    """
    raw_results = reader.readtext(image)

    detections = []
    for bbox, text, confidence in raw_results:
        if confidence >= threshold:
            # Convert bbox points to plain lists of ints for JSON serialization
            bbox_clean = [[int(round(pt[0])), int(round(pt[1]))] for pt in bbox]
            detections.append({
                "bbox": bbox_clean,
                "text": text.strip(),
                "confidence": round(float(confidence), 4),
            })

    # Sort detections top-to-bottom by the y-coordinate of the top-left corner
    detections.sort(key=lambda d: d["bbox"][0][1])

    return detections
