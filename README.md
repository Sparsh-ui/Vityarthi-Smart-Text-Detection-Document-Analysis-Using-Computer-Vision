# Smart Text Detection & Document Analysis Using Computer Vision

A Python-based command-line Computer Vision application that extracts text from images, evaluates OCR confidence, assesses image quality, and exports structured results. It uses OpenCV preprocessing and EasyOCR and is designed for CPU execution.

## Problem Statement

OCR accuracy drops with noise, blur, uneven lighting, low contrast, and complex backgrounds. This project improves the input image before OCR and combines text detection with confidence and image-quality analysis so users can judge the reliability of extracted text.

## Objectives & Features

- Preprocess images using resizing, grayscale conversion, Gaussian denoising, CLAHE contrast enhancement, and optional adaptive thresholding.
- Detect and recognize printed text with EasyOCR, returning bounding boxes, text, and confidence scores.
- Analyze OCR confidence using mean, minimum, maximum, and low-confidence counts.
- Measure image brightness, contrast, and Laplacian-based sharpness with Good/Moderate/Poor interpretation.
- Generate annotated images with confidence-based bounding boxes and labels.
- Export human-readable TXT and structured JSON reports.
- Provide a scriptable `argparse` CLI with no GUI requirement.

## Computer Vision Techniques

| Technique | Implementation / Purpose |
|---|---|
| Image validation | `cv2.imread`; supports JPEG, PNG, BMP, TIFF, WebP |
| Resizing | Images above 2000 px on the largest dimension are scaled with `INTER_AREA`, preserving aspect ratio |
| Grayscale | `cv2.cvtColor` converts BGR to a single intensity channel |
| Gaussian filtering | `cv2.GaussianBlur`, 3×3 kernel, reduces high-frequency noise |
| CLAHE | `cv2.createCLAHE` improves local contrast under uneven lighting |
| Adaptive thresholding | Optional `cv2.adaptiveThreshold` for varying backgrounds |
| OCR | EasyOCR detects text regions and recognizes characters |
| Visualization | `cv2.polylines` and `cv2.putText` draw boxes and labels |
| Brightness | `np.mean` of grayscale pixels, 0–255 scale |
| Contrast | `np.std` of grayscale pixel intensities |
| Sharpness | Variance of `cv2.Laplacian`; higher values generally indicate sharper images |

> Image-quality values are practical heuristic indicators, not scientifically validated or standardized quality scores.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| OpenCV | Image processing, visualization, quality analysis |
| NumPy | Array operations and statistics |
| EasyOCR | Text detection and OCR |
| pytest | Unit testing |
| argparse | CLI argument parsing |

## Project Structure

```
smart-text-detection/
├── main.py                    # CLI entry point and pipeline orchestration
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Validation, resizing, grayscale, denoise, CLAHE, thresholding
│   ├── text_detection.py      # Cached EasyOCR reader and structured OCR output
│   ├── confidence_analysis.py # OCR confidence statistics and low-confidence flags
│   ├── image_quality.py       # Brightness, contrast, sharpness, quality interpretation
│   ├── visualization.py       # Bounding boxes and confidence labels
│   ├── result_export.py       # TXT and JSON report generation
│   └── utils.py               # Directory, filename, path, dimension, formatting helpers
├── tests/                     # pytest unit tests
├── samples/                   # Sample input images
├── scripts/                   # Helper/sample-generation scripts
├── output/                    # Generated results (gitignored except .gitkeep)
├── report/                    # Project report
├── statement.md               # Problem statement and project scope
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Module Details

- **preprocessing.py:** Runs resize → grayscale → Gaussian denoise → CLAHE → optional adaptive threshold. Steps are independently testable.
- **text_detection.py:** Uses a module-level cached EasyOCR reader and returns bounding boxes, text, and confidence scores sorted top-to-bottom.
- **confidence_analysis.py:** Calculates mean/min/max confidence and low-confidence counts using a configurable threshold (analysis default 0.5).
- **image_quality.py:** Calculates brightness, contrast, and Laplacian sharpness and assigns Good/Moderate/Poor quality.
- **visualization.py:** Annotates the original image with polygonal bounding boxes and text labels; high- and low-confidence detections use different colors.
- **result_export.py:** Writes structured JSON and human-readable TXT results.
- **utils.py:** Handles directories, image paths, filename sanitization, dimensions, and percentage formatting.

## System Workflow

```
Input Image
    ↓
Validation & Format Check
    ↓
Preprocessing
Resize → Grayscale → Gaussian Denoise → CLAHE → Optional Threshold
    ↓
Image Quality Analysis
Brightness + Contrast + Sharpness
    ↓
EasyOCR Text Detection & Recognition
Bounding Boxes + Text + Confidence
    ↓
Confidence Analysis
Statistics + Low-Confidence Flags
    ↓
Visualization
Annotated Bounding Boxes + Labels
    ↓
TXT + JSON Export
```

Detections below the configured OCR threshold are filtered from the final results; detections below 50% confidence are flagged for reliability analysis.

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

EasyOCR downloads its language model the first time it runs, so an internet connection is required for the initial model download. Tests do not require the EasyOCR model or internet access.

## Running the Project

Generate the sample document image:

```bash
python scripts/create_sample.py
```

Run the pipeline:

```bash
python main.py --input samples/sample_document.jpg --output output
```

### Command-Line Options

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | Required | Input image path |
| `--output`, `-o` | `output` | Output directory |
| `--lang`, `-l` | `en` | OCR language code(s), comma-separated |
| `--threshold`, `-t` | `0.25` | Minimum confidence for including a detection |
| `--preprocess`, `-p` | Off | Enables adaptive thresholding |

Example:

```bash
python main.py --input photo.png --output results --lang en --threshold 0.3 --preprocess
```

## Output Files

| File | Contents |
|---|---|
| `detected_text.txt` | Detected text regions, confidence scores, and bounding-box coordinates |
| `analysis_report.json` | Detections, confidence statistics, quality metrics, metadata, and input information |
| `annotated_image.jpg` | Original image with color-coded bounding boxes, text labels, and confidence percentages |

The terminal also prints the number of detected regions, confidence statistics, image-quality assessment, and output paths.

## Testing

```bash
pytest -q
```

Tests cover:

- Preprocessing: grayscale conversion, denoising, contrast enhancement, and thresholding using synthetic images.
- Confidence analysis: mean, min, max, and low-confidence counts using mock detections.
- Image quality: brightness, contrast, and sharpness using images with known properties.
- Result export: JSON structure, TXT formatting, and populated/empty detection cases.

## Limitations

- OCR accuracy depends strongly on resolution, blur, lighting, and image quality.
- Handwritten text is less reliably recognized than printed text.
- Stylized/decorative fonts may reduce recognition accuracy.
- Text over complex photographic backgrounds can be difficult to detect.
- CPU-only execution is slower for large images.
- Image-quality metrics are heuristic and are not calibrated against ground truth or a standardized quality framework.

## Future Enhancements

- Batch processing of multiple images
- Paragraph/text-region grouping by spatial proximity
- PDF input with page-by-page extraction
- Per-language accuracy benchmarking using labeled data
- Spell checking and formatting post-processing
- Rotated text detection and angle correction

## Academic Relevance

The project demonstrates core Computer Vision concepts including:

- Image preprocessing and filtering
- Histogram/contrast enhancement using CLAHE
- Edge-based analysis using the Laplacian operator
- Adaptive thresholding
- Object detection and recognition through deep-learning-based OCR
- Bounding-box visualization
- Statistical confidence and image-quality analysis

These techniques relate directly to image formation/preprocessing, filtering, enhancement, thresholding, feature extraction, and recognition topics in a Computer Vision curriculum.

## Example Execution

```bash
python main.py --input samples/sample_document.jpg --output output
```

Expected results:

```
output/
├── detected_text.txt
├── analysis_report.json
└── annotated_image.jpg
```

The TXT file contains detected regions and confidence scores; the JSON file contains the complete analysis and metadata; and the annotated image shows detected text regions visually.
