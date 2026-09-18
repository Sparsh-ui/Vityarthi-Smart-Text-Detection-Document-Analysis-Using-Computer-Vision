"""
Tests for the confidence analysis module.

Uses mock detection data — no OCR model or internet required.
"""

import pytest
from src.confidence_analysis import analyze_confidence, LOW_CONFIDENCE_THRESHOLD


def _make_detections(confidences):
    """Create a list of mock detection dicts with given confidences."""
    return [
        {"bbox": [[0, 0], [10, 0], [10, 10], [0, 10]],
         "text": f"word{i}",
         "confidence": c}
        for i, c in enumerate(confidences)
    ]


class TestAnalyzeConfidence:
    def test_empty_detections(self):
        result = analyze_confidence([])
        assert result["total_detections"] == 0
        assert result["average_confidence"] == 0.0
        assert result["min_confidence"] == 0.0
        assert result["max_confidence"] == 0.0
        assert result["low_confidence_count"] == 0

    def test_single_detection(self):
        dets = _make_detections([0.85])
        result = analyze_confidence(dets)
        assert result["total_detections"] == 1
        assert result["average_confidence"] == 0.85
        assert result["min_confidence"] == 0.85
        assert result["max_confidence"] == 0.85
        assert result["low_confidence_count"] == 0

    def test_multiple_detections(self):
        dets = _make_detections([0.9, 0.7, 0.3, 0.95])
        result = analyze_confidence(dets)
        assert result["total_detections"] == 4
        assert 0.71 <= result["average_confidence"] <= 0.72
        assert result["min_confidence"] == 0.3
        assert result["max_confidence"] == 0.95

    def test_low_confidence_count(self):
        dets = _make_detections([0.2, 0.4, 0.6, 0.8])
        result = analyze_confidence(dets, low_threshold=0.5)
        assert result["low_confidence_count"] == 2

    def test_all_low_confidence(self):
        dets = _make_detections([0.1, 0.2, 0.3])
        result = analyze_confidence(dets)
        assert result["low_confidence_count"] == 3

    def test_all_high_confidence(self):
        dets = _make_detections([0.9, 0.95, 0.99])
        result = analyze_confidence(dets)
        assert result["low_confidence_count"] == 0

    def test_custom_threshold(self):
        dets = _make_detections([0.6, 0.7, 0.8])
        result = analyze_confidence(dets, low_threshold=0.75)
        assert result["low_confidence_count"] == 2
        assert result["low_confidence_threshold"] == 0.75

    def test_confidence_values_list(self):
        dets = _make_detections([0.5, 0.9])
        result = analyze_confidence(dets)
        assert len(result["confidence_values"]) == 2

    def test_default_threshold_stored(self):
        result = analyze_confidence([])
        assert result["low_confidence_threshold"] == LOW_CONFIDENCE_THRESHOLD
