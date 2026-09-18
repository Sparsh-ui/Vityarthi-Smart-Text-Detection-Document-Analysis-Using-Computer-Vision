"""
Smart Text Detection & Document Analysis Using Computer Vision

Command-line application that accepts an image, preprocesses it, detects
text regions with OCR, analyzes confidence and image quality, and exports
structured results.

Usage:
    python main.py --input path/to/image.jpg --output output
"""

import argparse
import os
import sys
import time

# Ensure UTF-8 output on Windows to handle EasyOCR progress bar characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7

from src.preprocessing import load_image, run_preprocessing_pipeline
from src.text_detection import create_reader, detect_text
from src.confidence_analysis import analyze_confidence
from src.image_quality import assess_quality
from src.visualization import annotate_image, save_annotated
from src.result_export import generate_report, export_json, export_txt
from src.utils import ensure_directory, is_valid_image_path, format_dimensions


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smart Text Detection & Document Analysis Using Computer Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --input samples/sample_document.jpg --output output\n"
            "  python main.py --input photo.png --lang en --threshold 0.3 --preprocess\n"
        ),
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--output", "-o", default="output",
        help="Output directory for results (default: output)."
    )
    parser.add_argument(
        "--lang", "-l", default="en",
        help="OCR language code (default: en). Use comma-separated for multiple."
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.25,
        help="Minimum confidence threshold for detections (default: 0.25)."
    )
    parser.add_argument(
        "--preprocess", "-p", action="store_true",
        help="Apply adaptive thresholding during preprocessing."
    )
    return parser.parse_args()


def print_header():
    """Print the application header."""
    print()
    print("=" * 58)
    print("   SMART TEXT DETECTION & DOCUMENT ANALYSIS")
    print("   Computer Vision Project")
    print("=" * 58)
    print()


def print_results(input_path: str, dimensions: dict, detections: list,
                  conf_stats: dict, quality: dict, output_files: list):
    """Print a formatted summary of the analysis results."""
    print("-" * 58)
    print("  RESULTS")
    print("-" * 58)
    print()
    print(f"  Input image:          {os.path.basename(input_path)}")
    print(f"  Image dimensions:     {format_dimensions(dimensions['width'], dimensions['height'])}")
    print(f"  Text regions found:   {conf_stats['total_detections']}")
    print()

    if detections:
        print(f"  Average confidence:   {conf_stats['average_confidence']:.1%}")
        print(f"  Min confidence:       {conf_stats['min_confidence']:.1%}")
        print(f"  Max confidence:       {conf_stats['max_confidence']:.1%}")
        if conf_stats["low_confidence_count"] > 0:
            print(f"  Low-confidence:       {conf_stats['low_confidence_count']} detection(s)")
        print()

        print("  Detected text:")
        for i, det in enumerate(detections, 1):
            print(f"    [{i}] \"{det['text']}\" ({det['confidence']:.0%})")
        print()

    print(f"  Image quality:        {quality['interpretation']}")
    print(f"    Brightness:         {quality['brightness']:.1f}")
    print(f"    Contrast:           {quality['contrast']:.1f}")
    print(f"    Sharpness:          {quality['sharpness']:.1f}")
    print()

    print("  Output files:")
    for f in output_files:
        print(f"    - {f}")
    print()
    print("=" * 58)
    print()


def main():
    """Main application entry point."""
    args = parse_arguments()

    print_header()

    # --- Step 1: Validate input ---
    if not is_valid_image_path(args.input):
        print(f"  ERROR: Invalid or missing image: {args.input}")
        print("  Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp")
        sys.exit(1)

    print(f"  Loading image: {args.input}")

    try:
        original_image = load_image(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    h, w = original_image.shape[:2]
    dimensions = {"width": w, "height": h}
    print(f"  Image size: {format_dimensions(w, h)} ({original_image.shape[2]} channels)")

    # --- Step 2: Prepare output directory ---
    try:
        ensure_directory(args.output)
    except OSError as e:
        print(f"  ERROR: Could not create output directory: {e}")
        sys.exit(1)

    # --- Step 3: Preprocess ---
    print("  Preprocessing image...")
    stages = run_preprocessing_pipeline(original_image, apply_threshold=args.preprocess)
    processed_image = stages["processed"]

    # --- Step 4: Assess image quality ---
    print("  Analyzing image quality...")
    quality_metrics = assess_quality(original_image)

    # --- Step 5: Run OCR ---
    languages = [lang.strip() for lang in args.lang.split(",")]
    print(f"  Initializing OCR reader (languages: {languages})...")

    start_time = time.time()
    try:
        reader = create_reader(languages, use_gpu=False)
    except Exception as e:
        print(f"  ERROR: Failed to initialize OCR reader: {e}")
        sys.exit(1)

    print("  Detecting text...")
    try:
        detections = detect_text(reader, processed_image, threshold=args.threshold)
    except Exception as e:
        print(f"  ERROR: OCR detection failed: {e}")
        sys.exit(1)

    ocr_time = time.time() - start_time
    print(f"  OCR completed in {ocr_time:.1f}s — found {len(detections)} region(s)")

    if not detections:
        print("  WARNING: No text detected in the image.")

    # --- Step 6: Confidence analysis ---
    conf_stats = analyze_confidence(detections)

    # --- Step 7: Visualize ---
    print("  Creating annotated image...")
    annotated = annotate_image(original_image, detections)
    annotated_path = os.path.join(args.output, "annotated_image.jpg")
    save_annotated(annotated, annotated_path)

    # --- Step 8: Export results ---
    print("  Exporting results...")

    txt_path = os.path.join(args.output, "detected_text.txt")
    export_txt(detections, txt_path)

    report = generate_report(args.input, dimensions, detections,
                             conf_stats, quality_metrics)
    json_path = os.path.join(args.output, "analysis_report.json")
    export_json(report, json_path)

    # --- Step 9: Print summary ---
    output_files = [
        os.path.relpath(txt_path),
        os.path.relpath(json_path),
        os.path.relpath(annotated_path),
    ]
    print_results(args.input, dimensions, detections, conf_stats,
                  quality_metrics, output_files)


if __name__ == "__main__":
    main()
