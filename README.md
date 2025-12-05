# Lunar Crater Detector

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic crater detection on grayscale lunar PNG images using advanced computer vision techniques including CLAHE, Canny edge detection, morphological operations, and ellipse fitting.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm](#algorithm)
- [Output Format](#output-format)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements an automated crater detection system for lunar surface imagery analysis. The system processes grayscale PNG images and identifies crater locations, sizes, and morphological characteristics using a multi-stage image processing pipeline.

## Features

- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for local contrast enhancement
- **Edge Detection**: Canny edge detector with auto-threshold calibration
- **Morphological Operations**: Closing and opening operations for contour completion and noise removal
- **Ellipse Fitting**: Accurate crater characterization using ellipse fitting
- **Batch Processing**: Process multiple images efficiently
- **CSV Export**: Structured output with crater properties
- **Annotated Visualization**: Generate annotated images with detected craters
- **Comprehensive Testing**: Unit and integration test suite
- **Type Hints**: Full type hints for code clarity and IDE support

## Project Structure

```
lunar-crater-detector/
├── crater_detector.py           # Main detection algorithm
├── test_crater_detection.py     # Comprehensive test suite
├── requirements.txt             # Project dependencies
├── README.md                    # This file
├── .pylintrc                    # Linting configuration
└── .github/
    └── workflows/
        └── pylint.yml          # CI/CD workflow for linting
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/rammaruboina-rgb/lunar-crater-detector.git
cd lunar-crater-detector
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python crater_detector.py <image_folder> <output_csv>
```

**Arguments:**
- `image_folder`: Directory containing grayscale PNG images
- `output_csv`: Path to output CSV file (e.g., `solution.csv`)

**Example:**

```bash
python crater_detector.py ./lunar_images/ solution.csv --verbose
```

### Advanced Options

```bash
python crater_detector.py ./images/ results.csv \
  --generatetestimages \
  --verbose \
  --min-radius 5 \
  --max-radius 150
```

**Options:**
- `--verbose`: Enable verbose logging output
- `--generatetestimages`: Generate annotated visualization images
- `--min-radius`: Minimum crater radius in pixels (default: 3)
- `--max-radius`: Maximum crater radius in pixels (default: 100)

### Programmatic Usage

```python
from crater_detector import CraterDetector
from pathlib import Path

# Initialize detector
detector = CraterDetector(
    min_radius=3,
    max_radius=100,
    verbose=True
)

# Detect craters in single image
craters = detector.detect(Path('image.png'))
for crater in craters:
    print(f"Crater at ({crater['x']}, {crater['y']}) "
          f"with radius {crater['radius']:.2f}")

# Batch process directory
results = detector.detect_batch(Path('./images/'))
```

## Algorithm

The crater detection pipeline consists of the following steps:

### 1. Contrast Enhancement (CLAHE)

- Applies Contrast Limited Adaptive Histogram Equalization
- Parameters: clipLimit=2.0, tileGridSize=(8, 8)
- Enhances local contrast without amplifying noise

### 2. Edge Detection (Canny)

- Applies Gaussian blur (kernel=(5,5), sigma=1.0)
- Canny edge detection with thresholds: 50, 150
- Produces binary edge map

### 3. Morphological Operations

- Morphological closing: removes small holes
- Morphological opening: removes small noise
- Elliptical kernel (5x5)
- 2 iterations for closing, 1 for opening

### 4. Contour Detection

- Finds contours from binary edge map
- Retrieval mode: RETR_TREE
- Approximation method: CHAIN_APPROX_SIMPLE

### 5. Crater Characterization

- Filters contours by area constraints
- Fits ellipse to valid contours
- Extracts crater properties:
  - Center coordinates (x, y)
  - Radius (average of major/minor axes)
  - Major/minor axis lengths
  - Orientation angle
  - Area

## Output Format

The detector outputs a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| ImageName | string | Source image filename |
| CraterX | float | X-coordinate of crater center |
| CraterY | float | Y-coordinate of crater center |
| CraterRadius | float | Crater radius in pixels |
| MajorAxis | float | Major axis length |
| MinorAxis | float | Minor axis length |
| Angle | float | Ellipse orientation angle (degrees) |
| Area | float | Crater area (pixels²) |

**Example:**

```csv
ImageName,CraterX,CraterY,CraterRadius,MajorAxis,MinorAxis,Angle,Area
luna_01.png,256.5,128.3,25.4,50.8,25.2,45.3,2031.2
luna_01.png,512.1,384.7,18.9,37.8,19.1,22.5,1128.9
```

## Testing

### Run All Tests

```bash
python -m pytest test_crater_detection.py -v
```

### Run Specific Test Classes

```bash
python -m pytest test_crater_detection.py::TestCraterDetector -v
python -m pytest test_crater_detection.py::TestIntegration -v
```

### Run with Coverage

```bash
python -m pytest test_crater_detection.py --cov=crater_detector --cov-report=html
```

### Test Cases

**Unit Tests (TestCraterDetector):**
- Detector initialization
- CLAHE application
- Edge detection
- Morphological operations
- Single image detection
- Batch detection
- Crater validation
- CSV output
- Annotated image generation
- Edge cases
- Nonexistent file handling

**Integration Tests (TestIntegration):**
- Complete detection pipeline
- End-to-end workflow

## Configuration

### Adjusting Detection Parameters

Edit the constants in `crater_detector.py`:

```python
MIN_CRATER_RADIUS = 3      # Minimum detection radius
MAX_CRATER_RADIUS = 100    # Maximum detection radius
USE_CLASSIFIER = True      # Enable/disable classifier
```

### CLAHE Parameters

In `CraterDetector._apply_clahe()`:

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,           # Contrast limit
    tileGridSize=(8, 8)     # Tile size
)
```

### Canny Edge Detection Parameters

In `CraterDetector._detect_edges()`:

```python
edges = cv2.Canny(
    blurred,
    threshold1=50,   # Lower threshold
    threshold2=150   # Upper threshold
)
```

## Performance Considerations

- **Memory**: Suitable for images up to ~2000x2000 pixels on standard hardware
- **Speed**: ~100-500ms per image depending on size and complexity
- **Batch Processing**: Efficient for processing large image sets
- **Parallelization**: Can be parallelized using multiprocessing for batch jobs

## Troubleshooting

### No Craters Detected

1. Check image quality and format (must be grayscale PNG)
2. Adjust CLAHE parameters if image is too dark/bright
3. Modify Canny thresholds if craters have low contrast
4. Increase `--max-radius` for larger craters

### False Positives

1. Reduce `--max-radius` to filter noise
2. Adjust morphological kernel size
3. Tighten Canny thresholds

### Performance Issues

1. Process smaller images or subregions
2. Reduce image resolution
3. Use batch processing for multiple images

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
pip install -r requirements.txt
pylint crater_detector.py
black crater_detector.py
mypy crater_detector.py
pytest test_crater_detection.py
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision algorithms
- NumPy for numerical computing
- Pandas for data handling

## Citation

If you use this project in your research, please cite:

```bibtex
@software{lunar_crater_detector_2024,
  author = {Ram Maruboina},
  title = {Lunar Crater Detector},
  year = {2024},
  url = {https://github.com/rammaruboina-rgb/lunar-crater-detector}
}
```

## Contact

For questions or feedback, please open an issue on GitHub or contact the maintainer.
