"""
Tests for the result export module.

Validates JSON structure, TXT output, and report generation.
Uses temporary directories — no internet required.
"""

import json
import os
import tempfile

import pytest

from src.result_export import generate_report, export_json, export_txt


def _sample_detections():
    """Create sample detection data for testing."""
    return [
        {
            "bbox": [[10, 20], [100, 20], [100, 50], [10, 50]],
            "text": "Hello",
            "confidence": 0.95,
        },
        {
            "bbox": [[10, 60], [150, 60], [150, 90], [10, 90]],
            "text": "World",
            "confidence": 0.82,
        },
    ]


def _sample_confidence_stats():
    return {
        "total_detections": 2,
        "average_confidence": 0.885,
        "min_confidence": 0.82,
        "max_confidence": 0.95,
        "low_confidence_count": 0,
        "low_confidence_threshold": 0.5,
        "confidence_values": [0.95, 0.82],
    }


def _sample_quality_metrics():
    return {
        "brightness": 150.0,
        "contrast": 45.0,
        "sharpness": 200.0,
        "width": 800,
        "height": 600,
        "channels": 3,
        "pixel_min": 10,
        "pixel_max": 245,
        "interpretation": "Good",
        "note": "Basic indicators.",
    }


class TestGenerateReport:
    def test_report_has_required_keys(self):
        report = generate_report(
            "test_image.jpg",
            {"width": 800, "height": 600},
            _sample_detections(),
            _sample_confidence_stats(),
            _sample_quality_metrics(),
        )
        assert "report_generated" in report
        assert "input_file" in report
        assert "image_dimensions" in report
        assert "text_detections" in report
        assert "confidence_statistics" in report
        assert "image_quality" in report

    def test_input_file_is_basename(self):
        report = generate_report(
            "/path/to/image.jpg",
            {"width": 100, "height": 100},
            [], {}, {},
        )
        assert report["input_file"] == "image.jpg"

    def test_detections_preserved(self):
        dets = _sample_detections()
        report = generate_report("img.jpg", {"width": 1, "height": 1},
                                 dets, {}, {})
        assert len(report["text_detections"]) == 2
        assert report["text_detections"][0]["text"] == "Hello"


class TestExportJson:
    def test_creates_valid_json_file(self):
        report = generate_report(
            "test.jpg",
            {"width": 800, "height": 600},
            _sample_detections(),
            _sample_confidence_stats(),
            _sample_quality_metrics(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            export_json(report, path)

            assert os.path.isfile(path)

            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["input_file"] == "test.jpg"
            assert len(loaded["text_detections"]) == 2

    def test_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            export_json({"key": "value"}, path)

            with open(path, "r") as f:
                data = json.load(f)
            assert data["key"] == "value"


class TestExportTxt:
    def test_creates_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "detected.txt")
            export_txt(_sample_detections(), path)
            assert os.path.isfile(path)

    def test_txt_contains_detected_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "detected.txt")
            export_txt(_sample_detections(), path)

            with open(path, "r") as f:
                content = f.read()

            assert "Hello" in content
            assert "World" in content

    def test_empty_detections_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "detected.txt")
            export_txt([], path)

            with open(path, "r") as f:
                content = f.read()

            assert "No text detected" in content

    def test_txt_contains_confidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "detected.txt")
            export_txt(_sample_detections(), path)

            with open(path, "r") as f:
                content = f.read()

            assert "Confidence" in content
