#!/usr/bin/env python3
"""Automatic crater detection on grayscale lunar PNG images.

This module provides functionality for detecting and characterizing craters
in lunar surface imagery using image processing techniques including CLAHE,
Canny edge detection, morphological operations, and ellipse fitting.
"""

import math
import argparse
from pathlib import Path
import cv2 as cv
import numpy as np
import pandas as pd

USE_CLASSIFIER = True


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments containing image_folder and output_csv.
    """
    parser = argparse.ArgumentParser(
        description="Automatic crater detection on grayscale lunar PNG images."
    )
    parser.add_argument("image_folder", type=str, help="Folder containing PNG images")
    parser.add_argument("output_csv", type=str, help="Output CSV path (e.g., solution.csv)")
    return parser.parse_args()


def list_png_images(folder):
    """List all PNG images in a folder.
    
    Args:
        folder (str): Path to folder containing PNG images.
        
    Returns:
        list: Sorted list of Path objects for PNG images.
    """
    folder_path = Path(folder)
    images = sorted([p for p in folder_path.glob("*.png") if p.is_file()])
    return images


def enhance_image(gray):
    """Enhance grayscale image using CLAHE and Gaussian blur.
    
    Args:
        gray (numpy.ndarray): Input image (grayscale or BGR).
        
    Returns:
        numpy.ndarray: Enhanced image.
    """
    if gray.ndim == 3:
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blurred = cv.GaussianBlur(eq, (5, 5), 1.0)
    return blurred


def detect_edges(enhanced):
    """Detect edges using Canny edge detector with auto-threshold.
    
    Args:
        enhanced (numpy.ndarray): Enhanced image.
        
    Returns:
        numpy.ndarray: Binary edge map.
    """
    v = np.median(enhanced)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv.Canny(enhanced, lower, upper)
    return edges


def close_edges(edges):
    """Close edge gaps using morphological closing operation.
    
    Args:
        edges (numpy.ndarray): Binary edge map.
        
    Returns:
        numpy.ndarray: Closed edge map.
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    return closed


def contour_to_ellipse(contour):
    """Fit ellipse to contour.
    
    Args:
        contour (numpy.ndarray): Contour points.
        
    Returns:
        tuple or None: (cx, cy, a, b, angle_deg) or None if contour too small.
    """
    if len(contour) < 5:
        return None
    ellipse = cv.fitEllipse(contour)
    (cx, cy), (w, h), angle = ellipse

    a = max(w, h) / 2.0
    b = min(w, h) / 2.0

    if h > w:
        angle = (angle + 90.0) % 180.0

    return float(cx), float(cy), float(a), float(b), float(angle)


def ellipse_touches_border(center_x, center_y, major, minor, angle_deg, width, height):
    """Check if ellipse touches image border.
    
    Args:
        center_x (float): X coordinate of center.
        center_y (float): Y coordinate of center.
        major (float): Semi-major axis length.
        minor (float): Semi-minor axis length.
        angle_deg (float): Rotation angle in degrees.
        width (int): Image width.
        height (int): Image height.
        
    Returns:
        bool: True if ellipse touches border, False otherwise.
    """
    theta = math.radians(angle_deg)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))

    half_w = major * cos_t + minor * sin_t
    half_h = major * sin_t + minor * cos_t

    x_min = center_x - half_w
    x_max = center_x + half_w
    y_min = center_y - half_h
    y_max = center_y + half_h

    if x_min <= 0 or y_min <= 0 or x_max >= (width - 1) or y_max >= (height - 1):
        return True
    return False


def classify_crater_rim(enhanced, contour):
    """Classify crater rim strength based on gradient magnitude.
    
    Args:
        enhanced (numpy.ndarray): Enhanced image.
        contour (numpy.ndarray): Crater rim contour points.
        
    Returns:
        int: Classification class (0-4) or -1 if classification disabled.
    """
    if not USE_CLASSIFIER:
        return -1

    grad_x = cv.Sobel(enhanced, cv.CV_32F, 1, 0, ksize=3)
    grad_y = cv.Sobel(enhanced, cv.CV_32F, 0, 1, ksize=3)
    mag = cv.magnitude(grad_x, grad_y)

    pts = contour.reshape(-1, 2)
    h, w = enhanced.shape
    vals = []
    for x_coord, y_coord in pts:
        ix = int(round(x_coord))
        iy = int(round(y_coord))
        if 0 <= ix < w and 0 <= iy < h:
            vals.append(mag[iy, ix])

    if len(vals) == 0:
        return -1

    mean_mag = float(np.mean(vals))
    normalized = min(max(mean_mag / 40.0, 0.0), 1.0)
    cls = int(round(normalized * 4))
    cls = max(0, min(4, cls))
    return cls


def process_image(image_path):
    """Process single image for crater detection.
    
    Args:
        image_path (Path): Path to input image.
        
    Returns:
        list: List of detected crater parameters or empty detection row.
    """
    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        return [[-1, -1, -1, -1, -1, image_path.name, -1]]

    h, w = img.shape[:2]
    min_dim = min(h, w)

    enhanced = enhance_image(img)
    edges = detect_edges(enhanced)
    closed = close_edges(edges)

    contours_info = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info

    rows = []
    for contour in contours:
        ellipse_params = contour_to_ellipse(contour)
        if ellipse_params is None:
            continue

        cx, cy, a, b, angle_deg = ellipse_params

        if b < 40.0:
            continue

        if (a + b) >= 0.6 * min_dim:
            continue

        if ellipse_touches_border(cx, cy, a, b, angle_deg, w, h):
            continue

        class_id = classify_crater_rim(enhanced, contour)

        rows.append([
            cx,
            cy,
            a,
            b,
            angle_deg,
            image_path.name,
            class_id,
        ])

    if not rows:
        rows.append([-1, -1, -1, -1, -1, image_path.name, -1])

    return rows


def main():
    """Main entry point for crater detection pipeline."""
    args = parse_args()
    image_folder = args.image_folder
    output_csv = args.output_csv

    images = list_png_images(image_folder)
    if not images:
        df_empty = pd.DataFrame(
            columns=[
                "center_x",
                "center_y",
                "semi_major",
                "semi_minor",
                "angle_deg",
                "image_id",
                "class_id",
            ]
        )
        df_empty.to_csv(output_csv, index=False)
        return

    all_rows = []
    for img_path in images:
        rows = process_image(img_path)
        all_rows.extend(rows)

    df = pd.DataFrame(
        all_rows,
        columns=[
            "center_x",
            "center_y",
            "semi_major",
            "semi_minor",
            "angle_deg",
            "image_id",
            "class_id",
        ],
    )
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
