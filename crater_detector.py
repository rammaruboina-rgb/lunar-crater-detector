#!/usr/bin/env python3
"""
Crater Detection Module.

Automatic crater detection on grayscale lunar PNG images using OpenCV.
Filters craters by size and visibility, outputs CSV in specified format.
"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd

MIN_SEMI_MINOR = 40
MAX_CRATER_RATIO = 0.6

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Crater detection on lunar PNG images'
    )
    parser.add_argument('image_folder', type=str, nargs='?',
                        default='lunar_images')
    parser.add_argument('output_csv', type=str, nargs='?',
                        default='solution.csv')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--decimals', type=int, default=2)
    parser.add_argument('--canny_low_ratio', type=float, default=0.5)
    parser.add_argument('--canny_high_ratio', type=float, default=1.5)
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
    try:
        cx, cy, w, h, angle = cv2.fitEllipse(contour)
        semi_a = max(w, h) / 2.0
        semi_b = min(w, h) / 2.0
        if h > w:
            angle = angle + 90.0
        return float(cx), float(cy), float(semi_a), float(semi_b),\
            float(angle)
    except cv2.error:
        return None


def is_crater_valid(semi_a: float, semi_b: float,
                    cx: float, cy: float,
                    width: int, height: int) -> bool:
    """Check if crater meets filtering criteria.

    Args:
        semi_a: Semi-major axis.
        semi_b: Semi-minor axis.
        cx: Center X coordinate.
        cy: Center Y coordinate.
        width: Image width.
        height: Image height.

    Returns:
        True if crater is valid, False otherwise.
    """
    if semi_b < MIN_SEMI_MINOR:
        return False

    min_dim = min(width, height)
    if (2 * semi_a + 2 * semi_b) >= MAX_CRATER_RATIO * min_dim:
        return False

    border_margin = max(semi_a, semi_b) + 1
    if cx - border_margin < 0 or cx + border_margin > width:
        return False
    if cy - border_margin < 0 or cy + border_margin > height:
        return False

    return True


def process_image(img_path: Path,
                  low_ratio: float,
                  high_ratio: float) -> List[tuple]:
    """Process single image for craters.

    Args:
        img_path: Image file path.
        low_ratio: Canny lower threshold ratio.
        high_ratio: Canny upper threshold ratio.

    Returns:
        List of crater tuples (cx, cy, a, b, angle).
    """
    craters = []
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return craters

    height, width = img.shape
    enhanced = enhance_image(img)
    edges = detect_edges(enhanced, low_ratio, high_ratio)
    closed = close_edges(edges)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        ellipse = contour_to_ellipse(contour)
        if ellipse is None:
            continue
        cx, cy, semi_a, semi_b, angle = ellipse
        if is_crater_valid(semi_a, semi_b, cx, cy, width, height):
            craters.append((cx, cy, semi_a, semi_b, angle, -1))

    return craters


def main() -> None:
    """Main crater detection pipeline."""
    args = parse_args()
    image_folder = args.image_folder
    output_csv = args.output_csv
    verbose = getattr(args, 'verbose', False)
    decimals = getattr(args, 'decimals', 2)
    low_ratio = getattr(args, 'canny_low_ratio', 0.5)
    high_ratio = getattr(args, 'canny_high_ratio', 1.5)

    try:
        out_parent = Path(output_csv).parent
        if str(out_parent) and not out_parent.exists():
            out_parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        pass

    images = list_png_images(image_folder)

    csv_cols = [
        'ellipseCenterXpx', 'ellipseCenterYpx',
        'ellipseSemimajorpx', 'ellipseSemiminorpx',
        'ellipseRotationdeg', 'inputImage',
        'craterclassification'
    ]

    rows = []
    for img_path in images:
        if verbose:
            print(f"Processing {img_path.name}")
        craters = process_image(img_path, low_ratio, high_ratio)
        if not craters:
            rows.append({
                'ellipseCenterXpx': -1,
                'ellipseCenterYpx': -1,
                'ellipseSemimajorpx': -1,
                'ellipseSemiminorpx': -1,
                'ellipseRotationdeg': -1,
                'inputImage': img_path.name,
                'craterclassification': -1
            })
        else:
            for cx, cy, semi_a, semi_b, angle, cls_id in craters:
                rows.append({
                    'ellipseCenterXpx': round(cx, decimals),
                    'ellipseCenterYpx': round(cy, decimals),
                    'ellipseSemimajorpx': round(semi_a, decimals),
                    'ellipseSemiminorpx': round(semi_b, decimals),
                    'ellipseRotationdeg': round(angle, decimals),
                    'inputImage': img_path.name,
                    'craterclassification': cls_id
                })

    df = pd.DataFrame(rows, columns=csv_cols)
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"Results saved to {output_csv}")
        print(f"Total rows: {len(df)}")


if __name__ == '__main__':
    main()
