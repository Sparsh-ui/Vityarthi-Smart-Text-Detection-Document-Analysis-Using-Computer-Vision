"""
Generate a synthetic sample document image for testing.

Creates an image with several lines of text using OpenCV drawing functions.
The output is saved to samples/sample_document.jpg.

Usage:
    python scripts/create_sample.py
"""

import cv2
import numpy as np
import os


def create_sample_document(output_path: str = "samples/sample_document.jpg"):
    """Generate a synthetic document image with multiple lines of text.

    The image simulates a simple printed document with a title, body
    paragraphs, and a footer. Uses OpenCV putText for rendering.

    Args:
        output_path: Where to save the generated image.
    """
    # Create a white canvas (A4-ish proportions, scaled down)
    width, height = 800, 1000
    img = np.ones((height, width, 3), dtype=np.uint8) * 245  # Off-white

    # Add a subtle header bar
    cv2.rectangle(img, (0, 0), (width, 80), (60, 60, 80), -1)
    cv2.putText(img, "SAMPLE DOCUMENT", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 2, cv2.LINE_AA)

    # Document body text lines
    lines = [
        "Smart Text Detection Project",
        "VITyarthi Computer Vision Course",
        "",
        "This is a sample document created for testing",
        "the text detection and OCR pipeline. It contains",
        "multiple lines of text at different positions.",
        "",
        "Section 1: Introduction",
        "Computer vision enables machines to interpret",
        "visual information from the real world.",
        "",
        "Section 2: Methodology",
        "We use EasyOCR for optical character recognition",
        "and OpenCV for image preprocessing tasks.",
        "",
        "Key Metrics:",
        "  - Accuracy: 95.2%",
        "  - Precision: 93.8%",
        "  - Recall: 96.1%",
        "",
        "Date: September 2026",
        "Author: Student Name",
    ]

    y_offset = 130
    for line in lines:
        if line == "":
            y_offset += 15
            continue

        # Use different fonts for headings vs body
        if line.startswith("Section") or line.startswith("Key"):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.65
            color = (30, 30, 120)
            thickness = 2
        elif line.startswith("  -"):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            color = (50, 50, 50)
            thickness = 1
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            color = (40, 40, 40)
            thickness = 1

        cv2.putText(img, line, (40, y_offset), font, scale, color,
                    thickness, cv2.LINE_AA)
        y_offset += 32

    # Add a footer line
    cv2.line(img, (40, height - 60), (width - 40, height - 60), (180, 180, 180), 1)
    cv2.putText(img, "Page 1 of 1", (340, height - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        os.makedirs(output_dir, exist_ok=True)
    except FileExistsError:
        pass  # Directory already exists (Windows edge case)

    cv2.imwrite(output_path, img)
    print(f"Sample document created: {output_path}")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Lines of text: {sum(1 for l in lines if l.strip())}")


if __name__ == "__main__":
    create_sample_document()
