"""
Result Visualization Module for Smart Text Detection.

Creates annotated copies of the input image with bounding boxes, recognized
text labels, and confidence scores drawn using OpenCV drawing functions.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Tuple


# Visual style constants
BOX_COLOR = (0, 200, 0)          # Green bounding boxes
TEXT_BG_COLOR = (0, 200, 0)      # Green background for text labels
TEXT_FG_COLOR = (255, 255, 255)  # White text
LOW_CONF_BOX_COLOR = (0, 100, 255)  # Orange for low-confidence detections
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2


def annotate_image(image: np.ndarray, detections: List[Dict],
                   low_threshold: float = 0.5) -> np.ndarray:
    """Draw bounding boxes, text, and confidence on a copy of the image.

    High-confidence detections are drawn in green; low-confidence
    detections (below low_threshold) are drawn in orange.

    Args:
        image: Original BGR image.
        detections: List of detection dicts with 'bbox', 'text', 'confidence'.
        low_threshold: Confidence below this uses the warning color.

    Returns:
        Annotated copy of the image (original is not modified).
    """
    annotated = image.copy()

    for det in detections:
        bbox = det["bbox"]
        text = det["text"]
        confidence = det["confidence"]

        # Choose color based on confidence
        color = BOX_COLOR if confidence >= low_threshold else LOW_CONF_BOX_COLOR

        # Draw the bounding box as a polygon
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color,
                      thickness=BOX_THICKNESS)

        # Prepare the label text
        label = f"{text} ({confidence:.0%})"

        # Calculate label position and background rectangle
        top_left = (int(bbox[0][0]), int(bbox[0][1]))
        (label_w, label_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )

        # Position label above the bounding box
        label_y = max(top_left[1] - 5, label_h + 5)
        label_x = top_left[0]

        # Draw background rectangle for readability
        cv2.rectangle(
            annotated,
            (label_x, label_y - label_h - baseline),
            (label_x + label_w, label_y + baseline),
            color, cv2.FILLED
        )

        # Draw label text
        cv2.putText(annotated, label, (label_x, label_y),
                    FONT, FONT_SCALE, TEXT_FG_COLOR, FONT_THICKNESS,
                    cv2.LINE_AA)

    return annotated


def save_annotated(image: np.ndarray, output_path: str) -> str:
    """Save an annotated image to disk.

    Args:
        image: Annotated BGR image.
        output_path: Full path where the image will be saved.

    Returns:
        The absolute path of the saved file.

    Raises:
        IOError: If the image could not be saved.
    """
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to save annotated image to: {output_path}")
    return os.path.abspath(output_path)
