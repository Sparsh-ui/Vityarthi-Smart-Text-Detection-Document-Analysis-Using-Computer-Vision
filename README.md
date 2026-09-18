# Smart Text Detection & Document Analysis Using Computer Vision

## 1. Overview

This is a command-line Computer Vision application that extracts text from images and evaluates OCR reliability and image quality. The pipeline loads an input image, applies a sequence of preprocessing steps to improve clarity, runs Optical Character Recognition (OCR) to detect and recognize text regions, computes confidence statistics and image-quality metrics, and exports the results as structured data files.

The application is built with Python, OpenCV, and EasyOCR, and is designed to run on a standard CPU without requiring GPU hardware.

## 2. Problem Statement

Extracting reliable text from images is a common requirement in document digitization, accessibility, and automated data entry. Real-world images are frequently affected by noise, uneven lighting, low contrast, and blur — all of which degrade OCR accuracy.

This project addresses the problem by implementing an image preprocessing pipeline that improves text clarity before OCR, combined with confidence analysis and image-quality assessment to help users understand the reliability of the extracted results.

## 3. Objectives

- Apply image preprocessing techniques (resizing, grayscale conversion, noise reduction, contrast enhancement, thresholding) to improve OCR input quality
- Detect and recognize text regions in images using EasyOCR
- Analyze OCR confidence scores to identify reliable and unreliable detections
- Assess image quality through brightness, contrast, and sharpness metrics
- Generate annotated visualizations showing detected text regions with confidence labels
- Export structured results in plain-text and JSON formats
- Provide a fully scriptable command-line interface requiring no GUI

## 4. Features

- **Image Preprocessing Pipeline** — resizing, grayscale conversion, Gaussian noise reduction, CLAHE contrast enhancement, and optional adaptive thresholding
- **Text Detection and OCR** — EasyOCR-based text recognition with configurable language and confidence threshold
- **Confidence Analysis** — statistical summary of detection confidence (mean, min, max) with low-confidence flagging
- **Image Quality Assessment** — brightness, contrast, and Laplacian-based sharpness metrics with qualitative interpretation (Good / Moderate / Poor)
- **Result Visualization** — annotated output image with color-coded bounding boxes (green for high confidence, orange for low confidence)
- **Structured Export** — plain-text report listing detected regions, and a JSON report containing all results, metrics, and metadata
- **Command-Line Interface** — fully scriptable with `argparse`; supports options for language, threshold, and preprocessing mode

## 5. Computer Vision Techniques Used

### Image Loading and Validation

The image is loaded using `cv2.imread` and validated for existence and decodability. Supported formats include JPEG, PNG, BMP, TIFF, and WebP.

### Resizing

Images whose largest dimension exceeds 2000 pixels are scaled down while preserving aspect ratio, using `cv2.resize` with `INTER_AREA` interpolation.

### Grayscale Conversion

The image is converted from BGR to a single-channel intensity image using `cv2.cvtColor`, reducing computational cost for subsequent processing steps.

### Gaussian Filtering

A Gaussian blur (`cv2.GaussianBlur`) with a 3×3 kernel is applied to reduce high-frequency pixel noise without significantly degrading text edges.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Local contrast is enhanced using `cv2.createCLAHE`. Unlike global histogram equalization, CLAHE operates on small tiles to preserve local detail, which improves text visibility in unevenly lit documents.

### Adaptive Thresholding (Optional)

When enabled via the `--preprocess` flag, `cv2.adaptiveThreshold` converts the image to binary using locally computed thresholds, which is useful for documents with varying background intensity.

### EasyOCR Text Detection and Recognition

The `easyocr.Reader` detects text regions and recognizes characters, returning bounding-box coordinates, recognized text strings, and confidence scores. The reader is cached at module level to avoid re-initializing the model on repeated calls.

### Bounding-Box Visualization

Detected text regions are drawn on a copy of the original image using `cv2.polylines` for bounding boxes and `cv2.putText` for labels. High-confidence and low-confidence detections use different colors for quick visual assessment.

### Brightness Calculation

Mean pixel intensity of the grayscale image, computed with `np.mean`, serves as a brightness indicator on a 0–255 scale.

### Contrast Calculation

Standard deviation of grayscale pixel intensities, computed with `np.std`, serves as a contrast indicator.

### Laplacian-Based Sharpness Measurement

The variance of the Laplacian (`cv2.Laplacian`) measures the presence of edges. Higher variance indicates a sharper image; low variance suggests blur.

> **Note:** The image-quality metrics are practical heuristic indicators, not scientifically validated quality scores.

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| OpenCV | Image loading, preprocessing, drawing, and quality analysis |
| NumPy | Array operations and statistical calculations |
| EasyOCR | Text detection and optical character recognition |
| pytest | Unit testing framework |
| argparse | Command-line argument parsing |

## 7. Project Architecture

```
smart-text-detection/
├── main.py                      # CLI entry point — orchestrates the full pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Image loading, validation, resizing, and preprocessing
│   ├── text_detection.py        # EasyOCR wrapper with cached reader and structured output
│   ├── confidence_analysis.py   # Statistical analysis of OCR confidence scores
│   ├── image_quality.py         # Brightness, contrast, and sharpness assessment
│   ├── visualization.py         # Annotated image generation with bounding boxes
│   ├── result_export.py         # JSON and TXT report generation
│   └── utils.py                 # Directory, filename, and formatting utilities
├── tests/                       # Unit tests (pytest)
├── samples/                     # Sample input images
├── scripts/                     # Helper scripts (sample image generation)
├── output/                      # Generated results (gitignored except .gitkeep)
├── report/                      # Project report directory
├── statement.md                 # Problem statement and project scope
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### Module Responsibilities

- **`preprocessing.py`** — Loads and validates images, then runs a configurable pipeline: resize → grayscale → denoise → contrast enhance → (optional) adaptive threshold. Each step is an independent, testable function.
- **`text_detection.py`** — Wraps EasyOCR with a module-level cached reader to avoid model re-initialization. Returns structured dictionaries with bounding boxes, text, and confidence scores, sorted top-to-bottom.
- **`confidence_analysis.py`** — Computes summary statistics (mean, min, max, low-confidence count) from detection confidence scores. Uses a configurable threshold (default 0.5) to flag unreliable detections.
- **`image_quality.py`** — Computes brightness, contrast, and Laplacian sharpness from the input image. Returns a qualitative interpretation (Good / Moderate / Poor) alongside raw numeric metrics.
- **`visualization.py`** — Draws polygonal bounding boxes and text labels on a copy of the input image. Uses green for high-confidence and orange for low-confidence detections.
- **`result_export.py`** — Serializes results to a JSON report (structured data with metadata) and a TXT file (human-readable list of detected text regions).
- **`utils.py`** — Shared helpers for directory creation, image path validation, filename sanitization, dimension formatting, and percentage formatting.

## 8. System Workflow

```
Input Image
     ↓
Image Validation         — Verify file exists and format is supported
     ↓
Image Preprocessing      — Resize → Grayscale → Denoise → CLAHE → (Threshold)
     ↓
Image Quality Analysis   — Compute brightness, contrast, and sharpness metrics
     ↓
OCR Text Detection       — Run EasyOCR to detect and recognize text regions
     ↓
Confidence Analysis      — Compute statistics and flag low-confidence results
     ↓
Result Visualization     — Draw annotated bounding boxes on the original image
     ↓
TXT + JSON Export        — Write detected text and full analysis report to files
```

1. **Image Validation** — The input file path is checked for existence and format support before loading.
2. **Image Preprocessing** — The loaded image passes through a sequential pipeline of resizing, grayscale conversion, Gaussian noise reduction, and CLAHE contrast enhancement. Adaptive thresholding is applied if requested.
3. **Image Quality Analysis** — Brightness, contrast, and sharpness are computed from the original image and a qualitative interpretation is assigned.
4. **OCR Text Detection** — EasyOCR processes the preprocessed image and returns detected text regions with bounding boxes and confidence scores. Results below the confidence threshold are filtered out.
5. **Confidence Analysis** — Summary statistics are computed from the detection confidence scores, and detections below 50% confidence are flagged.
6. **Result Visualization** — Bounding boxes and labels are drawn on a copy of the original image with color coding by confidence level.
7. **TXT + JSON Export** — A plain-text file lists detected text regions, and a JSON file contains the complete analysis report including detections, confidence statistics, and quality metrics.

## 9. Installation

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** The first time EasyOCR runs, it will download the required language model files. An internet connection is needed for this initial download only.

## 10. Running the Project

### Generate the sample document image (first time only)

```bash
python scripts/create_sample.py
```

### Run the text detection pipeline

```bash
python main.py --input samples/sample_document.jpg --output output
```

### Command-Line Options

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | *(required)* | Path to the input image file |
| `--output`, `-o` | `output` | Output directory for results |
| `--lang`, `-l` | `en` | OCR language code(s), comma-separated |
| `--threshold`, `-t` | `0.25` | Minimum confidence threshold for including a detection |
| `--preprocess`, `-p` | off | Enable adaptive thresholding during preprocessing |

### Example with all options

```bash
python main.py --input photo.png --output results --lang en --threshold 0.3 --preprocess
```

## 11. Output Files

Running the pipeline generates three files in the output directory:

| File | Description |
|---|---|
| `detected_text.txt` | Human-readable list of detected text regions with confidence scores and bounding-box coordinates |
| `analysis_report.json` | Structured JSON report containing all detections, confidence statistics, image-quality metrics, and metadata |
| `annotated_image.jpg` | Copy of the input image with bounding boxes, text labels, and confidence percentages drawn on it |

## 12. Testing

```bash
pytest -q
```

The test suite validates:

- **Preprocessing functions** — correct grayscale conversion, noise reduction, contrast enhancement, and thresholding behaviour using synthetic test images
- **Confidence analysis** — correct computation of summary statistics (mean, min, max, low-confidence count) using mock detection data
- **Image quality metrics** — correct brightness, contrast, and sharpness calculations using images with known properties
- **Result export** — correct JSON structure, TXT formatting, and content accuracy for both populated and empty detection results

All tests use synthetic images and mock data. No EasyOCR model download or internet connection is required to run the test suite.

## 13. Limitations

- **OCR accuracy depends on image quality.** Blurry, low-resolution, or poorly lit images produce significantly worse results.
- **Handwritten text** is poorly recognized by EasyOCR compared to printed text.
- **Stylized or decorative fonts** may not be recognized accurately.
- **Complex backgrounds** (e.g., text overlaid on photographs) reduce detection accuracy.
- **CPU-only execution** — the project is configured without GPU acceleration, which makes OCR slower on large images.
- **Image-quality metrics are heuristic indicators**, not calibrated against any ground truth or standardized quality assessment framework.

## 14. Future Enhancements

- Add support for batch processing of multiple images
- Implement text region grouping by spatial proximity (paragraph detection)
- Add PDF input support with page-by-page extraction
- Add per-language accuracy benchmarking with labeled test data
- Implement basic text post-processing (spell checking, formatting)
- Support rotated text detection with angle correction

## 15. Academic Relevance

This project demonstrates practical application of core Computer Vision concepts:

- **Image Preprocessing** — applying grayscale conversion, Gaussian filtering, and contrast enhancement to prepare images for downstream analysis
- **Histogram Equalization** — using CLAHE to improve local contrast in unevenly lit images
- **Edge Detection** — applying the Laplacian operator to measure image sharpness
- **Thresholding** — using adaptive thresholding to binarize images for improved text separation
- **Object Detection and Recognition** — leveraging deep-learning-based OCR to locate and recognize text regions
- **Visualization** — annotating images with bounding boxes and labels to present detection results visually
- **Statistical Analysis** — computing confidence and quality metrics to evaluate system performance

These techniques connect directly to image processing, filtering, enhancement, and feature extraction topics covered in the Computer Vision curriculum.

## 16. Project Execution Example

```bash
python main.py --input samples/sample_document.jpg --output output
```

Expected output files after execution:

- `output/detected_text.txt` — lists each detected text region with its confidence score and bounding-box coordinates
- `output/analysis_report.json` — contains the complete analysis report including input metadata, all text detections, confidence statistics, and image-quality metrics
- `output/annotated_image.jpg` — the original image annotated with color-coded bounding boxes and confidence labels

The terminal also prints a formatted summary showing the number of detected regions, confidence statistics, image-quality assessment, and the paths of the generated output files.