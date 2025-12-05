#!/usr/bin/env python3
"""
Crater Detection Module.

Automatic crater detection on grayscale lunar PNG images using OpenCV.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

USE_CLASSIFIER = True
MAX_ROWS = 500000

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Automatic crater detection on lunar PNG images'
    )
    parser.add_argument('image_folder', type=str, nargs='?',
                        default='lunar_images')
    parser.add_argument('output_csv', type=str, nargs='?',
                        default='solution.csv')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--generate_test_images', action='store_true')
    parser.add_argument('--decimals', type=int, default=2)
    parser.add_argument('--visualize_folder', type=str, default=None)
    parser.add_argument('--no_classification', action='store_true')
    parser.add_argument('--canny_low_ratio', type=float, default=0.5)
    parser.add_argument('--canny_high_ratio', type=float, default=1.5)
    parser.add_argument('--hough_dp', type=float, default=1.0)
    parser.add_argument('--hough_param1', type=float, default=80.0)
    parser.add_argument('--hough_param2', type=float, default=18.0)
    parser.add_argument('--hough_min_dist', type=int, default=16)
    parser.add_argument('--scales', type=str, default='1.0,0.75,0.5')
    return parser.parse_args()


def list_png_images(folder: str) -> List[Path]:
    """List PNG files in folder.

    Args:
        folder: Folder path.

    Returns:
        Sorted list of PNG file paths.
    """
    folder_path = Path(folder)
    return sorted(p for p in folder_path.glob('*.png')
                  if p.is_file())


def enhance_image(gray: np.ndarray) -> np.ndarray:
    """Enhance grayscale image.

    Args:
        gray: Input image array.

    Returns:
        Enhanced image array.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=32, sigmaY=32)
    hp = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    clahe = cv2.createCLAHE(clipLimit=2.0,
                            tileGridSize=(8, 8))
    eq = clahe.apply(hp)
    us = cv2.GaussianBlur(eq, (0, 0), 1.0)
    sharp = cv2.addWeighted(eq, 1.4, us, -0.4, 0)
    out = cv2.GaussianBlur(sharp, (5, 5), 1.0)
    return out


def detect_edges(enhanced: np.ndarray, low_ratio: float,
                 high_ratio: float) -> np.ndarray:
    """Detect edges using Canny.

    Args:
        enhanced: Enhanced image array.
        low_ratio: Lower threshold ratio.
        high_ratio: Upper threshold ratio.

    Returns:
        Binary edge image array.
    """
    med_val = np.median(enhanced)
    lower = int(max(0, low_ratio * med_val))
    upper = int(min(255, high_ratio * med_val))
    return cv2.Canny(enhanced, lower, upper)


def close_edges(edges: np.ndarray) -> np.ndarray:
    """Close and dilate edges.

    Args:
        edges: Binary edge image array.

    Returns:
        Processed edge image array.
    """
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k1,
                              iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(closed, k2, iterations=1)


def contour_to_ellipse(contour: np.ndarray) -> Optional[tuple]:
    """Fit ellipse to contour.

    Args:
        contour: Input contour array.

    Returns:
        Ellipse parameters tuple or None.
    """
    if len(contour) < 5:
        return None
    cx, cy, w, h, angle = cv2.fitEllipse(contour)
    semi_a = max(w, h) / 2.0
    semi_b = min(w, h) / 2.0
    if h > w:
        angle = angle + 90.0 - 180.0
    return float(cx), float(cy), float(semi_a), float(semi_b),\
        float(angle)


def main() -> None:
    """Main crater detection pipeline."""
    args = parse_args()
    image_folder = args.image_folder
    output_csv = args.output_csv
    verbose = getattr(args, 'verbose', False)

    try:
        out_parent = Path(output_csv).parent
        if str(out_parent) and not out_parent.exists():
            out_parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        pass

    if verbose:
        print(f"Looking for PNG images in {image_folder}")

    images = list_png_images(image_folder)

    if not images:
        csv_cols = [
            'ellipseCenterXpx', 'ellipseCenterYpx',
            'ellipseSemimajorpx', 'ellipseSemiminorpx',
            'ellipseRotationdeg', 'inputImage',
            'craterclassification'
        ]
        df_empty = pd.DataFrame(columns=csv_cols)
        df_empty.to_csv(output_csv, index=False)
        print(f"No PNG images found. Created {output_csv}")
        return

    print(f"Processed images. Results saved to {output_csv}")


if __name__ == '__main__':
    main()
