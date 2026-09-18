# Problem Statement

Extracting text from images is a recurring need in document digitization, accessibility tooling, and automated data entry. However, raw images often suffer from poor lighting, low contrast, noise, and other quality issues that degrade OCR accuracy. There is a need for a practical system that combines image preprocessing with text detection, provides confidence-aware results, and assesses image quality to help users understand the reliability of the output.

# Project Scope

This project implements a command-line Computer Vision system that:

1. Accepts a single image file as input
2. Validates and preprocesses the image for optimal OCR performance
3. Detects text regions and recognizes the text using EasyOCR
4. Analyzes OCR confidence scores to identify uncertain detections
5. Evaluates basic image quality metrics (brightness, contrast, sharpness)
6. Generates an annotated visualization showing detected text regions
7. Exports structured results as TXT and JSON files

The system is designed for offline, single-image processing on a standard laptop CPU. It does not implement real-time video processing, multi-page document workflows, or cloud-based OCR services.

# Target Users

- Computer Science students studying Computer Vision
- Researchers needing a quick text extraction tool for document images
- Developers prototyping OCR-based workflows
- Anyone needing to extract text from photographs of documents, signs, or labels

# Objectives

1. Demonstrate practical application of image preprocessing techniques (filtering, contrast enhancement, thresholding) in the context of OCR
2. Build a modular, well-documented Python system that separates concerns across preprocessing, detection, analysis, visualization, and export
3. Provide quantitative feedback (confidence scores, quality metrics) alongside the OCR output so users can assess result reliability
4. Create a project that runs from the command line without requiring a GUI, web server, or GPU

# High-Level Features

- Image validation and format checking
- Configurable preprocessing pipeline (grayscale, denoise, CLAHE, adaptive threshold)
- EasyOCR-based text detection with multi-language support
- Confidence score analysis with summary statistics and low-confidence flagging
- Image quality assessment with interpretive labels
- Annotated output image with color-coded bounding boxes
- Structured JSON report and human-readable TXT output
- Clean command-line interface with configurable parameters

# Expected Input

- A single image file in a supported format (JPEG, PNG, BMP, TIFF, WebP)
- The image should contain some readable text (printed or clearly written)
- Images can range from scanned documents to photographs of signs, labels, or screens

# Expected Output

The system generates three output files:

| File | Format | Contents |
|---|---|---|
| `detected_text.txt` | Plain text | Numbered list of detected text with confidence scores |
| `analysis_report.json` | JSON | Full structured report with detections, statistics, and quality metrics |
| `annotated_image.jpg` | JPEG image | Input image with bounding boxes, text labels, and confidence overlays |

Additionally, a formatted summary is printed to the terminal.

# Limitations

- OCR accuracy depends on input image quality — blurry, noisy, or low-resolution images will produce worse results
- Only languages supported by EasyOCR are available (default: English)
- Handwritten text recognition is unreliable
- Heavily rotated or perspective-distorted text may not be detected
- Image quality metrics are basic heuristic indicators, not scientifically calibrated scores
- The system processes one image at a time (no batch mode)
- CPU-only execution means OCR processing time increases with image size
