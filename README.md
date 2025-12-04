# Lunar Crater Detector

Automatic crater detection on grayscale lunar PNG images using OpenCV, morphological operations, and ellipse fitting.

## Overview

This project implements an automated system for detecting and characterizing craters in lunar surface imagery. The pipeline uses advanced image processing techniques including:

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for local contrast enhancement
- **Canny Edge Detection** with auto-threshold calibration
- **Morphological Operations** (closing) for contour completion
- **Ellipse Fitting** for crater rim characterization
- **Gradient-based Classification** for rim strength assessment

## Project Structure

```
lunar-crater-detector/
├── crater_detector.py          # Main detection algorithm
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── solution.csv                # Output format (crater detections)
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the crater detection on a folder of PNG images:

```bash
python crater_detector.py /path/to/images/ output.csv
```

### Arguments
- `image_folder`: Path to folder containing grayscale PNG images
- `output_csv`: Output CSV file path (e.g., `solution.csv`)

### Example

```bash
python crater_detector.py ./lunar_images/ ./detections/solution.csv
```

## Output Format

The output CSV contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `center_x` | float | X-coordinate of crater center |
| `center_y` | float | Y-coordinate of crater center |
| `semi_major` | float | Semi-major axis length (pixels) |
| `semi_minor` | float | Semi-minor axis length (pixels) |
| `angle_deg` | float | Ellipse rotation angle (degrees) |
| `image_id` | str | Source image filename |
| `class_id` | int | Rim strength class (0-4, or -1 if not detected) |

## Algorithm Details

### Image Enhancement
- Converts to grayscale if needed
- Applies CLAHE with clipLimit=2.0 and tileGridSize=(8,8)
- Applies Gaussian blur (5×5 kernel)

### Edge Detection
- Uses Canny edge detector with adaptive thresholds:
  - `lower = max(0, 0.66 * median_intensity)`
  - `upper = min(255, 1.33 * median_intensity)`

### Contour Processing
- Morphological closing with 5×5 elliptical kernel (2 iterations)
- Extracts external contours
- Fits ellipses to contours with ≥5 points

### Filtering Criteria
- Semi-minor axis ≥ 40 pixels (minimum crater size)
- (semi_major + semi_minor) < 0.6 × min(image_width, image_height)
- Ellipse must not touch image borders

### Classification
- Computes gradient magnitude along crater rim
- Normalizes by factor of 40.0
- Classifies into 5 strength classes (0-4)
- Returns -1 if classifier is disabled

## Configuration

Key parameters can be adjusted in `crater_detector.py`:

```python
USE_CLASSIFIER = True              # Enable/disable rim classification

# CLAHE parameters
cliplit_limit = 2.0
tile_grid_size = (8, 8)

# Morphological kernel
kernel_size = (5, 5)
iterations = 2

# Filtering thresholds
min_semi_minor = 40.0              # Minimum crater size
max_crater_fraction = 0.6          # Max ratio to image
```

## Performance Considerations

- Processing time depends on image size and number of detected contours
- Typical processing: 0.5-2 seconds per 1024×1024 image
- Memory usage: ~50-100 MB per image during processing

## Dependencies

- **opencv-python**: Image processing
- **numpy**: Numerical operations
- **pandas**: CSV output handling

## License

Private project for lunar imagery analysis.

## Notes

- Images must be 8-bit grayscale PNG files
- Detector optimized for lunar surface imagery
- Border-touching ellipses are automatically filtered out
- Missing detections output as [-1, -1, -1, -1, -1, image_name, -1]
