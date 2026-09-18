"""
Confidence Analysis Module for Smart Text Detection.

Analyzes OCR confidence scores from text detections to provide
summary statistics and flag low-confidence results.

Thresholds:
    LOW_CONFIDENCE_THRESHOLD = 0.5
        Detections below this value are flagged as low-confidence.
        This is a practical heuristic — OCR results under 50% confidence
        are frequently incorrect or unreliable.
"""

from typing import Dict, List

# Detections below this confidence are considered unreliable
LOW_CONFIDENCE_THRESHOLD = 0.5


def analyze_confidence(detections: List[Dict],
                       low_threshold: float = LOW_CONFIDENCE_THRESHOLD) -> Dict:
    """Compute summary statistics for OCR confidence scores.

    Args:
        detections: List of detection dicts, each containing a 'confidence'
                    key with a float value between 0 and 1.
        low_threshold: Confidence values strictly below this are counted
                       as low-confidence detections.

    Returns:
        Dictionary with the following keys:
            - total_detections: int
            - average_confidence: float (0.0 if no detections)
            - min_confidence: float (0.0 if no detections)
            - max_confidence: float (0.0 if no detections)
            - low_confidence_count: int
            - low_confidence_threshold: float
            - confidence_values: list of all confidence floats
    """
    if not detections:
        return {
            "total_detections": 0,
            "average_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "low_confidence_count": 0,
            "low_confidence_threshold": low_threshold,
            "confidence_values": [],
        }

    scores = [d["confidence"] for d in detections]
    low_count = sum(1 for s in scores if s < low_threshold)

    return {
        "total_detections": len(scores),
        "average_confidence": round(sum(scores) / len(scores), 4),
        "min_confidence": round(min(scores), 4),
        "max_confidence": round(max(scores), 4),
        "low_confidence_count": low_count,
        "low_confidence_threshold": low_threshold,
        "confidence_values": [round(s, 4) for s in scores],
    }
