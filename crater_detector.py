#!/usr/bin/env python3
"""
Automatic crater detection on grayscale lunar PNG images.

This module provides functionality for detecting and characterizing craters
in lunar surface imagery using image processing techniques including CLAHE,
Canny edge detection, morphological operations, and ellipse fitting.
"""

import math
import argparse
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USE_CLASSIFIER = True
MIN_CRATER_RADIUS = 3
MAX_CRATER_RADIUS = 100


class CraterDetector:
    """Main class for crater detection in lunar imagery."""

    def __init__(self, min_radius: int = MIN_CRATER_RADIUS,
                 max_radius: int = MAX_CRATER_RADIUS,
                 verbose: bool = False):
        """Initialize crater detector with parameters.

        Args:
            min_radius: Minimum crater radius in pixels
            max_radius: Maximum crater radius in pixels
            verbose: Enable verbose logging
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.verbose = verbose

    def _apply_clahe(self, image: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.

        Args:
            image: Input grayscale image
            clip_limit: CLAHE clip limit
            tile_size: Size of CLAHE tiles

        Returns:
            Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=tile_size)
        return clahe.apply(image)

    def _detect_edges(self, image: np.ndarray,
                     threshold1: int = 50,
                     threshold2: int = 150) -> np.ndarray:
        """Detect edges using Canny edge detector.

        Args:
            image: Input grayscale image
            threshold1: Lower Canny threshold
            threshold2: Upper Canny threshold

        Returns:
            Binary edge map
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        return edges

    def _morphological_operations(self, image: np.ndarray,
                                 kernel_size: int = 5) -> np.ndarray:
        """Apply morphological operations.

        Args:
            image: Input binary image
            kernel_size: Size of morphological kernel

        Returns:
            Processed binary image
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        return morph

    def _find_craters_from_contours(
        self, image: np.ndarray,
        contours: List[Any]
    ) -> List[Dict[str, Any]]:
        """Extract crater information from contours using ellipse fitting.

        Args:
            image: Original image
            contours: List of contours from cv2.findContours

        Returns:
            List of detected craters with properties
        """
        craters = []
        h, w = image.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < math.pi * (self.min_radius ** 2):
                continue
            if area > math.pi * (self.max_radius ** 2):
                continue

            # Fit ellipse
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                center, (major, minor), angle = ellipse
                x, y = center

                # Validate center is in image
                if 0 <= x < w and 0 <= y < h:
                    radius = (major + minor) / 4.0
                    crater = {
                        'x': float(x),
                        'y': float(y),
                        'radius': float(radius),
                        'major_axis': float(major),
                        'minor_axis': float(minor),
                        'angle': float(angle),
                        'area': float(area)
                    }
                    craters.append(crater)

        return craters

    def detect(self, image_path: Path) -> List[Dict[str, Any]]:
        """Detect craters in a lunar image.

        Args:
            image_path: Path to input grayscale PNG image

        Returns:
            List of detected craters with properties
        """
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return []

        if self.verbose:
            logger.info(f"Processing {image_path}")
            logger.info(f"Image shape: {image.shape}")

        # Apply CLAHE for contrast enhancement
        enhanced = self._apply_clahe(image)

        # Edge detection
        edges = self._detect_edges(enhanced)

        # Morphological operations
        morph = self._morphological_operations(edges)

        # Find contours
        contours, _ = cv2.findContours(
            morph,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract craters
        craters = self._find_craters_from_contours(image, contours)

        if self.verbose:
            logger.info(f"Detected {len(craters)} craters")

        return craters

    def detect_batch(self, image_dir: Path) -> Dict[str, List[Dict]]:
        """Detect craters in all images in a directory.

        Args:
            image_dir: Path to directory containing PNG images

        Returns:
            Dictionary mapping image filenames to detected craters
        """
        results = {}
        image_paths = sorted(image_dir.glob('*.png'))

        if self.verbose:
            logger.info(f"Found {len(image_paths)} PNG files")

        for idx, image_path in enumerate(image_paths, 1):
            if self.verbose:
                logger.info(f"[{idx}/{len(image_paths)}] Processing {image_path.name}")
            craters = self.detect(image_path)
            results[image_path.name] = craters

        return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Automatic crater detection on grayscale lunar PNG images."
    )

    parser.add_argument(
        "image_folder",
        type=str,
        help="Folder containing PNG images"
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Output CSV path (e.g., solution.csv)"
    )
    parser.add_argument(
        "--generatetestimages",
        action="store_true",
        help="Generate annotated test images with detected craters"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--min-radius",
        type=int,
        default=MIN_CRATER_RADIUS,
        help="Minimum crater radius"
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=MAX_CRATER_RADIUS,
        help="Maximum crater radius"
    )

    return parser.parse_args()


def save_results_to_csv(results: Dict[str, List[Dict]], output_csv: Path) -> None:
    """Save detection results to CSV.

    Args:
        results: Dictionary mapping image filenames to craters
        output_csv: Path to output CSV file
    """
    rows = []
    for image_name, craters in results.items():
        for crater in craters:
            row = {
                'ImageName': image_name,
                'CraterX': crater['x'],
                'CraterY': crater['y'],
                'CraterRadius': crater['radius'],
                'MajorAxis': crater['major_axis'],
                'MinorAxis': crater['minor_axis'],
                'Angle': crater['angle'],
                'Area': crater['area']
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")


def generate_test_images(
    image_dir: Path,
    results: Dict[str, List[Dict]],
    output_dir: Path
) -> None:
    """Generate annotated test images with detected craters.

    Args:
        image_dir: Path to original images
        results: Detection results
        output_dir: Path to save annotated images
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    for image_name, craters in results.items():
        image_path = image_dir / image_name
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        # Convert to BGR for colored annotations
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw detected craters
        for crater in craters:
            x = int(crater['x'])
            y = int(crater['y'])
            radius = int(crater['radius'])

            # Draw circle
            cv2.circle(annotated, (x, y), radius, (0, 255, 0), 2)
            # Draw center
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)

        # Save annotated image
        output_path = output_dir / f"annotated_{image_name}"
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"Saved annotated image: {output_path}")


def main() -> None:
    """Main function."""
    args = parse_args()

    image_dir = Path(args.image_folder)
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return

    # Initialize detector
    detector = CraterDetector(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        verbose=args.verbose
    )

    # Detect craters
    results = detector.detect_batch(image_dir)

    # Save results
    output_csv = Path(args.output_csv)
    save_results_to_csv(results, output_csv)

    # Generate test images if requested
    if args.generatetestimages:
        test_images_dir = Path("annotated_images")
        generate_test_images(image_dir, results, test_images_dir)

    logger.info("Crater detection completed.")


if __name__ == "__main__":
    main()
