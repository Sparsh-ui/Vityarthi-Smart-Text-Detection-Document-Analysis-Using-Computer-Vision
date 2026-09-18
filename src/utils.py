"""
Utility functions for Smart Text Detection & Document Analysis.

Provides reusable helpers for directory management, image validation,
filename handling, and formatting.
"""

import os
import re
from typing import Optional


def ensure_directory(path: str) -> str:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The absolute path of the created/existing directory.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:
        pass  # Directory already exists (Windows edge case)
    return os.path.abspath(path)


def is_valid_image_path(path: str) -> bool:
    """Check whether a file path points to an existing, supported image.

    Supported extensions: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp

    Args:
        path: File path to validate.

    Returns:
        True if the file exists and has a supported image extension.
    """
    if not path or not os.path.isfile(path):
        return False

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    _, ext = os.path.splitext(path)
    return ext.lower() in supported


def safe_filename(name: str) -> str:
    """Sanitize a string so it can be used safely as a filename.

    Replaces spaces with underscores and strips characters that are not
    alphanumeric, underscores, hyphens, or dots.

    Args:
        name: Raw filename string.

    Returns:
        Sanitized filename.
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^\w.\-]", "", name)
    return name


def format_dimensions(width: int, height: int) -> str:
    """Format image dimensions as a human-readable string.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Formatted string like '1920 x 1080'.
    """
    return f"{width} x {height}"


def get_basename(path: str) -> str:
    """Extract the base filename from a full path.

    Args:
        path: Full file path.

    Returns:
        Base filename (e.g. 'image.jpg').
    """
    return os.path.basename(path)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a 0-1 float as a percentage string.

    Args:
        value: Float value between 0 and 1.
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string like '87.5%'.
    """
    return f"{value * 100:.{decimals}f}%"
