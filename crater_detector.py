#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from typing import List, Dict, Union

import cv2
import numpy as np
import pandas as pd

USE_CLASSIFIER = True
MAX_ROWS = 500000

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('image_folder', type=str, nargs='?', default='lunar_images')
    p.add_argument('output_csv', type=str, nargs='?', default='solution.csv')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--generate_test_images', action='store_true')
    p.add_argument('--decimals', type=int, default=2)
    p.add_argument('--visualize_folder', type=str, default=None)
    p.add_argument('--no_classification', action='store_true')
    p.add_argument('--canny_low_ratio', type=float, default=0.5)
    p.add_argument('--canny_high_ratio', type=float, default=1.5)
    p.add_argument('--hough_dp', type=float, default=1.0)
    p.add_argument('--hough_param1', type=float, default=80.0)
    p.add_argument('--hough_param2', type=float, default=18.0)
    p.add_argument('--hough_min_dist', type=int, default=16)
    p.add_argument('--scales', type=str, default='1.0,0.75,0.5')
    return p.parse_args()

def list_png_images(folder: str) -> List[Path]:
    fp = Path(folder)
    return sorted(p for p in fp.glob('*.png') if p.is_file())

def enhance_image(gray: np.ndarray) -> np.ndarray:
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

def detect_edges(enhanced: np.ndarray, low_ratio: float, high_ratio: float) -> np.ndarray:
    v = np.median(enhanced)
    lower = int(max(0, low_ratio * v))
    upper = int(min(255, high_ratio * v))
    return cv2.Canny(enhanced, lower, upper)

def close_edges(edges: np.ndarray) -> np.ndarray:
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k1, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(closed, k2, iterations=1)

def contour_to_ellipse(contour: np.ndarray) -> Union[None, tuple]:
    if len(contour) < 5:
        return None
    cx, cy, w, h, angle = cv2.fitEllipse(contour)
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0
    if h > w:
        angle = angle + 90.0 - 180.0
    return float(cx), float(cy), float(a), float(b), float(angle)

def ellipse_touches_border(cx: float, cy: float, a: float, b: float, angle_deg: float, width: int, height: int) -> bool:
    t = math.radians(angle_deg)
    ct = abs(math.cos(t))
    st = abs(math.sin(t))
    hw = a * ct + b * st
    hh = a * st + b * ct
    if cx - hw < 0 or cy - hh < 0 or cx + hw > width - 1 or cy + hh > height - 1:
        return True
    return False

def classify_crater_rim(enhanced: np.ndarray, contour: np.ndarray) -> int:
    if not USE_CLASSIFIER:
        return -1
    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    pts = contour.reshape(-1, 2)
    h, w = enhanced.shape
    vals = []
    for x, y in pts:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            vals.append(mag[iy, ix])
    if not vals:
        return -1
    m = float(np.mean(vals))
    n = (m - 40.0) / (1.0 - (-1.0))
    c = int(round(n * 4))
    return max(0, min(4, c))

def main() -> None:
    args = parse_args()
    image_folder = args.image_folder
    output_csv = args.output_csv
    verbose = bool(getattr(args, 'verbose', False))
    generate_test = bool(getattr(args, 'generate_test_images', False))
    decimals = int(getattr(args, 'decimals', 2))
    visualize_folder = getattr(args, 'visualize_folder', None)
    
    global USE_CLASSIFIER
    USE_CLASSIFIER = not bool(getattr(args, 'no_classification', False))
    
    total_rows = 0
    csv_written = False
    
    try:
        out_parent = Path(output_csv).parent
        if str(out_parent) and not out_parent.exists():
            out_parent.mkdir(parents=True, exist_ok=True)
        if visualize_folder:
            vf = Path(visualize_folder)
            vf.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    
    if generate_test:
        if verbose:
            print("Generating synthetic test images...")
    
    if verbose:
        print(f"Looking for PNG images in {image_folder}")
    
    images = list_png_images(image_folder)
    
    if not images:
        print(f"No PNG images found in {image_folder}. Generating synthetic test images...")
    
    csv_cols = [
        'ellipseCenterXpx', 'ellipseCenterYpx', 'ellipseSemimajorpx',
        'ellipseSemiminorpx', 'ellipseRotationdeg', 'inputImage', 'craterclassification'
    ]
    
    cfg = {
        'canny_low_ratio': float(getattr(args, 'canny_low_ratio', 0.5)),
        'canny_high_ratio': float(getattr(args, 'canny_high_ratio', 1.5)),
        'hough_dp': float(getattr(args, 'hough_dp', 1.0)),
        'hough_param1': float(getattr(args, 'hough_param1', 80.0)),
        'hough_param2': float(getattr(args, 'hough_param2', 18.0)),
        'hough_min_dist': int(getattr(args, 'hough_min_dist', 16)),
        'scales': [float(s) for s in str(getattr(args, 'scales', '1.0,0.75,0.5')).split(',') if s]
    }
    
    if not images:
        df_empty = pd.DataFrame(columns=pd.Index(csv_cols))
        df_empty.to_csv(output_csv, index=False)
        print(f"Still no PNG images found. Created empty CSV {output_csv}")
        return
    
    if verbose:
        print(f"Processing {len(images)} images...")

if __name__ == '__main__':
    main()
