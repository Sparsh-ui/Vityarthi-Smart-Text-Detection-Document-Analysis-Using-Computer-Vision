"""
Tests for the preprocessing module.

Uses synthetic images (no model downloads or internet required).
"""

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    to_grayscale,
    resize_if_needed,
    reduce_noise,
    enhance_contrast,
    adaptive_threshold,
    run_preprocessing_pipeline,
)


def _make_color_image(width: int = 200, height: int = 150) -> np.ndarray:
    """Create a simple synthetic BGR image for testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Red rectangle on top half
    img[:height // 2, :, 2] = 200
    # Blue rectangle on bottom half
    img[height // 2:, :, 0] = 200
    return img


def _make_gray_image(width: int = 200, height: int = 150) -> np.ndarray:
    """Create a simple synthetic grayscale image."""
    return np.random.randint(50, 200, (height, width), dtype=np.uint8)


class TestToGrayscale:
    def test_converts_bgr_to_single_channel(self):
        img = _make_color_image()
        gray = to_grayscale(img)
        assert len(gray.shape) == 2
        assert gray.shape == (150, 200)

    def test_returns_grayscale_unchanged(self):
        gray = _make_gray_image()
        result = to_grayscale(gray)
        assert np.array_equal(result, gray)

    def test_output_dtype(self):
        img = _make_color_image()
        gray = to_grayscale(img)
        assert gray.dtype == np.uint8


class TestResizeIfNeeded:
    def test_small_image_unchanged(self):
        img = _make_color_image(100, 80)
        result = resize_if_needed(img, max_dimension=200)
        assert result.shape == img.shape

    def test_large_image_resized(self):
        img = _make_color_image(3000, 2000)
        result = resize_if_needed(img, max_dimension=1000)
        assert max(result.shape[:2]) <= 1000

    def test_aspect_ratio_preserved(self):
        img = _make_color_image(400, 200)
        result = resize_if_needed(img, max_dimension=200)
        original_ratio = 400 / 200
        new_ratio = result.shape[1] / result.shape[0]
        assert abs(original_ratio - new_ratio) < 0.1


class TestReduceNoise:
    def test_output_same_shape(self):
        img = _make_gray_image()
        result = reduce_noise(img)
        assert result.shape == img.shape

    def test_reduces_variance(self):
        # Noisy image should have lower variance after blurring
        noisy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blurred = reduce_noise(noisy, kernel_size=5)
        assert np.std(blurred) < np.std(noisy)


class TestEnhanceContrast:
    def test_output_same_shape(self):
        img = _make_gray_image()
        result = enhance_contrast(img)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = _make_gray_image()
        result = enhance_contrast(img)
        assert result.dtype == np.uint8


class TestAdaptiveThreshold:
    def test_output_is_binary(self):
        img = _make_gray_image()
        result = adaptive_threshold(img)
        unique = np.unique(result)
        assert all(v in [0, 255] for v in unique)

    def test_output_same_shape(self):
        img = _make_gray_image()
        result = adaptive_threshold(img)
        assert result.shape == img.shape


class TestPipeline:
    def test_returns_expected_keys(self):
        img = _make_color_image()
        stages = run_preprocessing_pipeline(img)
        assert "resized" in stages
        assert "grayscale" in stages
        assert "denoised" in stages
        assert "enhanced" in stages
        assert "processed" in stages

    def test_threshold_key_when_enabled(self):
        img = _make_color_image()
        stages = run_preprocessing_pipeline(img, apply_threshold=True)
        assert "thresholded" in stages

    def test_processed_is_2d(self):
        img = _make_color_image()
        stages = run_preprocessing_pipeline(img)
        assert len(stages["processed"].shape) == 2
