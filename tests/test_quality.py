"""
Tests for the image quality analysis module.

Uses synthetic images with known properties — no internet required.
"""

import cv2
import numpy as np
import pytest

from src.image_quality import assess_quality


def _make_bright_image() -> np.ndarray:
    """Create a bright white image."""
    return np.ones((100, 150, 3), dtype=np.uint8) * 240


def _make_dark_image() -> np.ndarray:
    """Create a dark image."""
    return np.ones((100, 150, 3), dtype=np.uint8) * 20


def _make_high_contrast_image() -> np.ndarray:
    """Create an image with strong contrast (half black, half white)."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, 100:] = 255
    return img


def _make_low_contrast_image() -> np.ndarray:
    """Create an image with very low contrast."""
    return np.ones((100, 150, 3), dtype=np.uint8) * 128


def _make_sharp_image() -> np.ndarray:
    """Create an image with sharp edges (checkerboard pattern)."""
    img = np.zeros((100, 100), dtype=np.uint8)
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            if (i // 10 + j // 10) % 2 == 0:
                img[i:i+10, j:j+10] = 255
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestAssessQuality:
    def test_returns_expected_keys(self):
        img = _make_bright_image()
        result = assess_quality(img)
        expected_keys = {
            "brightness", "contrast", "sharpness", "width", "height",
            "channels", "pixel_min", "pixel_max", "interpretation", "note"
        }
        assert expected_keys == set(result.keys())

    def test_bright_image_high_brightness(self):
        result = assess_quality(_make_bright_image())
        assert result["brightness"] > 200

    def test_dark_image_low_brightness(self):
        result = assess_quality(_make_dark_image())
        assert result["brightness"] < 50

    def test_high_contrast_image(self):
        result = assess_quality(_make_high_contrast_image())
        assert result["contrast"] > 100

    def test_low_contrast_image(self):
        result = assess_quality(_make_low_contrast_image())
        assert result["contrast"] < 5

    def test_dimensions_correct(self):
        result = assess_quality(_make_bright_image())
        assert result["width"] == 150
        assert result["height"] == 100
        assert result["channels"] == 3

    def test_grayscale_input(self):
        gray = np.ones((80, 120), dtype=np.uint8) * 128
        result = assess_quality(gray)
        assert result["channels"] == 1
        assert result["width"] == 120
        assert result["height"] == 80

    def test_interpretation_is_string(self):
        result = assess_quality(_make_bright_image())
        assert result["interpretation"] in ("Good", "Moderate", "Poor")

    def test_sharp_image_high_sharpness(self):
        result = assess_quality(_make_sharp_image())
        assert result["sharpness"] > 100

    def test_uniform_image_low_sharpness(self):
        # A completely uniform image has zero Laplacian variance
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = assess_quality(img)
        assert result["sharpness"] == 0.0

    def test_note_is_disclaimer(self):
        result = assess_quality(_make_bright_image())
        assert "not scientifically" in result["note"].lower() or "basic" in result["note"].lower()
