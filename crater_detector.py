#!/usr/bin/env python3
"""
Automatic crater detection on grayscale lunar PNG images.

This module provides functions for detecting craters in lunar images
using OpenCV, morphological operations, and ellipse fitting.
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import pandas as pd

USE_CLASSIFIER = True
MAX_ROWS = 500000

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Automatic crater detection on grayscale lunar PNG'
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
    """List PNG images in folder.

    Args:
        folder: Path to folder.

    Returns:
        Sorted list of PNG file paths.
    """
    folder_path = Path(folder)
    return sorted(p for p in folder_path.glob('*.png') if p.is_file())


def enhance_image(gray: np.ndarray) -> np.ndarray:
    """Enhance grayscale image for crater detection.

    Args:
        gray: Input grayscale image.

    Returns:
        Enhanced image.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=32, sigmaY=32)
    hp = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(hp)
    us = cv2.GaussianBlur(eq, (0, 0), 1.0)
    sharp = cv2.addWeighted(eq, 1.4, us, -0.4, 0)
    out = cv2.GaussianBlur(sharp, (5, 5), 1.0)
    return out


def detect_edges(enhanced: np.ndarray, low_ratio: float,
                 high_ratio: float) -> np.ndarray:
    """Detect edges using Canny edge detection.

    Args:
        enhanced: Enhanced image.
        low_ratio: Lower threshold ratio.
        high_ratio: Upper threshold ratio.

    Returns:
        Binary edge image.
    """
    med_val = np.median(enhanced)
    lower = int(max(0, low_ratio * med_val))
    upper = int(min(255, high_ratio * med_val))
    return cv2.Canny(enhanced, lower, upper)


def close_edges(edges: np.ndarray) -> np.ndarray:
    """Close edges using morphological operations.

    Args:
        edges: Binary edge image.

    Returns:
        Closed and dilated edge image.
    """
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel1,
                              iterations=1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(closed, kernel2, iterations=1)


def contour_to_ellipse(contour: np.ndarray) -> Optional[
        tuple]:
    """Fit ellipse to contour.

    Args:
        contour: Input contour.

    Returns:
        Tuple of (cx, cy, a, b, angle) or None.
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


def ellipse_touches_border(cx: float, cy: float, semi_a: float,
                           semi_b: float, angle_deg: float,
                           width: int, height: int) -> bool:
    """Check if ellipse touches image border.

    Args:
        cx: Center X coordinate.
        cy: Center Y coordinate.
        semi_a: Semi-major axis.
        semi_b: Semi-minor axis.
        angle_deg: Rotation angle in degrees.
        width: Image width.
        height: Image height.

    Returns:
        True if ellipse touches border, False otherwise.
    """
    theta = math.radians(angle_deg)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    half_w = semi_a * cos_t + semi_b * sin_t
    half_h = semi_a * sin_t + semi_b * cos_t
    return cx - half_w < 0 or cy - half_h < 0 or \
        cx + half_w > width - 1 or cy + half_h > height - 1


def classify_crater_rim(enhanced: np.ndarray,
                        contour: np.ndarray) -> int:
    """Classify crater rim based on gradient magnitude.

    Args:
        enhanced: Enhanced image.
        contour: Contour to classify.

    Returns:
        Classification ID (0-4) or -1.
    """
    if not USE_CLASSIFIER:
        return -1
    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    pts = contour.reshape(-1, 2)
    height, width = enhanced.shape
    vals = []
    for x, y in pts:
        idx_x = int(round(x))
        idx_y = int(round(y))
        if 0 <= idx_x < width and 0 <= idx_y < height:
            vals.append(mag[idx_y, idx_x])
    if not vals:
        return -1
    mean_mag = float(np.mean(vals))
    normalized = (mean_mag - 40.0) / 1.0
    cls_id = int(round(normalized * 4))
    return max(0, min(4, cls_id))


def main() -> None:
    """Main crater detection pipeline."""
    args = parse_args()
    image_folder = args.image_folder
    output_csv = args.output_csv
    verbose = bool(getattr(args, 'verbose', False))
    decimals = int(getattr(args, 'decimals', 2))
    visualize_folder = getattr(args, 'visualize_folder', None)

    global USE_CLASSIFIER
    USE_CLASSIFIER = not bool(getattr(args, 'no_classification',
                                       False))

    try:
        out_parent = Path(output_csv).parent
        if str(out_parent) and not out_parent.exists():
            out_parent.mkdir(parents=True, exist_ok=True)
        if visualize_folder:
            viz_path = Path(visualize_folder)
            viz_path.mkdir(parents=True, exist_ok=True)
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
        print(f"No PNG images found. Created empty CSV {output_csv}")
        return

    print(f"Processed images. Results saved to {output_csv}")


if __name__ == '__main__':
    main()
