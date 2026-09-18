"""
Result Export Module for Smart Text Detection.

Exports detection results, confidence statistics, and image quality
metrics into structured TXT and JSON files.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List


def generate_report(input_path: str,
                    dimensions: Dict[str, int],
                    detections: List[Dict],
                    confidence_stats: Dict,
                    quality_metrics: Dict) -> Dict[str, Any]:
    """Assemble a complete analysis report as a dictionary.

    Args:
        input_path: Path to the original input image.
        dimensions: Dict with 'width' and 'height'.
        detections: List of detection dicts (bbox, text, confidence).
        confidence_stats: Output from confidence_analysis.analyze_confidence.
        quality_metrics: Output from image_quality.assess_quality.

    Returns:
        A dictionary ready for JSON serialization containing all results.
    """
    return {
        "report_generated": datetime.now().isoformat(),
        "input_file": os.path.basename(input_path),
        "image_dimensions": dimensions,
        "text_detections": detections,
        "confidence_statistics": confidence_stats,
        "image_quality": quality_metrics,
    }


def export_json(report_data: Dict, path: str) -> str:
    """Write the analysis report to a JSON file.

    Args:
        report_data: Report dictionary (from generate_report).
        path: Output file path.

    Returns:
        Absolute path of the written file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    return os.path.abspath(path)


def export_txt(detections: List[Dict], path: str) -> str:
    """Write detected text to a plain text file.

    Each line contains the detected text followed by the confidence
    score in parentheses.

    Args:
        detections: List of detection dicts.
        path: Output file path.

    Returns:
        Absolute path of the written file.
    """
    with open(path, "w", encoding="utf-8") as f:
        if not detections:
            f.write("No text detected in the image.\n")
        else:
            f.write(f"Detected {len(detections)} text region(s):\n")
            f.write("=" * 50 + "\n\n")
            for i, det in enumerate(detections, 1):
                f.write(f"[{i}] {det['text']}\n")
                f.write(f"    Confidence: {det['confidence']:.1%}\n")
                f.write(f"    Bounding Box: {det['bbox']}\n\n")
    return os.path.abspath(path)
