#!/usr/bin/env python3
"""
Crater Detection Module.

Automatic crater detection on grayscale lunar PNG images using OpenCV.
Filters craters by size and visibility, outputs CSV in specified format.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray
import pandas as pd

use_classifier: bool = True
MAX_ROWS: int = 500000

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Automatic crater detection on grayscale lunar PNG images.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='lunar_images',
        help='Folder containing PNG images (default: lunar_images)',
    )
    parser.add_argument(
        'output_csv',
        type=str,
        nargs='?',
        default='solution.csv',
        help='Output CSV path (default: solution.csv)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress messages.',
    )
    parser.add_argument(
        '--generate-test-images',
        action='store_true',
        help='Create synthetic PNG test images inside the image folder before processing.',
    )
    parser.add_argument(
        '--decimals',
        type=int,
        default=2,
        help='Number of decimal places for numeric output (default: 2).',
    )
    parser.add_argument(
        '--visualize-folder',
        type=str,
        default=None,
        help='Folder to save annotated images.',
    )
    parser.add_argument(
        '--no-classification',
        action='store_true',
        help='Disable crater rim classification.',
    )
    parser.add_argument(
        '--canny-low-ratio',
        type=float,
        default=0.5,
        help='Canny lower threshold ratio to median.',
    )
    parser.add_argument(
        '--canny-high-ratio',
        type=float,
        default=1.5,
        help='Canny upper threshold ratio to median.',
    )
    parser.add_argument(
        '--hough-dp',
        type=float,
        default=1.0,
        help='HoughCircles dp.',
    )
    parser.add_argument(
        '--hough-param1',
        type=float,
        default=80.0,
        help='HoughCircles param1.',
    )
    parser.add_argument(
        '--hough-param2',
        type=float,
        default=18.0,
        help='HoughCircles param2.',
    )
    parser.add_argument(
        '--hough-min-dist',
        type=int,
        default=16,
        help='HoughCircles minDist.',
    )
    parser.add_argument(
        '--scales',
        type=str,
        default='1.0,0.75,0.5',
        help='Comma-separated scales for multi-scale detection.',
    )
    return parser.parse_args()

def list_png_images(folder: str) -> List[Path]:
    """Return sorted PNG image paths inside a folder."""
    folder_path = Path(folder)
    return sorted(p for p in folder_path.glob('*.png') if p.is_file())


def enhance_image(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply high-pass, CLAHE and sharpening to enhance crater visibility."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=32, sigmaY=32)
    hp = cv2.addWeighted(gray, 1.0, bg, -1.0, 128)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(hp)
    
    us = cv2.GaussianBlur(eq, (0, 0), 1.0)
    sharp = cv2.addWeighted(eq, 1.4, us, -0.4, 0)
    out = cv2.GaussianBlur(sharp, (5, 5), 1.0)
    
    return np.asarray(out, dtype=np.uint8)


def detect_edges(
    enhanced: NDArray[np.uint8],
    low_ratio: float,
    high_ratio: float,
) -> NDArray[np.uint8]:
    """Detect edges using Canny with median-derived thresholds."""
    v = float(np.median(enhanced))
    lower = int(max(0.0, low_ratio * v))
    upper = int(min(255.0, high_ratio * v))
    edges = cv2.Canny(enhanced, lower, upper)
    return np.asarray(edges, dtype=np.uint8)


def close_edges(edges: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Close gaps in edges using morphological operations."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    dil = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return np.asarray(dil, dtype=np.uint8)


def contour_to_ellipse(
    contour: NDArray[np.int32],
) -> Optional[Tuple[float, float, float, float, float]]:
    """Fit an ellipse to a contour and return cx, cy, a, b, angle_deg."""
    if contour.shape[0] < 5:
        return None
    
    cx, cy, w, h, angle = cv2.fitEllipse(contour)
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0
    
    if h > w:
        angle = angle + 90.0 - 180.0
    
    return (float(cx), float(cy), float(a), float(b), float(angle))


def ellipse_touches_border(
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle_deg: float,
    width: int,
    height: int,
) -> bool:
    """Return True if ellipse extends beyond image border."""
    theta = math.radians(angle_deg)
    cost = abs(math.cos(theta))
    sint = abs(math.sin(theta))
    
    half_w = a * cost + b * sint
    half_h = a * sint + b * cost
    
    x_min = cx - half_w
    x_max = cx + half_w
    y_min = cy - half_h
    y_max = cy + half_h
    
    return bool(x_min < 0.0 or y_min < 0.0 or x_max > float(width - 1) or y_max > float(height - 1))


def classify_crater_rim(
    enhanced: NDArray[np.uint8],
    contour: NDArray[np.int32],
) -> int:
    """Classify crater rim steepness heuristically into 0-4. Returns -1 if disabled."""
    if not use_classifier:
        return -1
    
    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    
    pts: NDArray[np.int32] = contour.reshape(-1, 2)
    vals: List[float] = []
    
    for x_coord, y_coord in pts:
        ix = int(round(float(x_coord)))
        iy = int(round(float(y_coord)))
        if 0 <= ix < mag.shape[1] and 0 <= iy < mag.shape[0]:
            vals.append(float(mag[iy, ix]))
    
    if not vals:
        return -1
    
    mean_mag = float(np.mean(vals))
    normalized = min(max(mean_mag / 40.0, 0.0), 1.0)
    cls = int(round(normalized * 4.0))
    
    return max(0, min(4, cls))


def suppress_duplicates(rows: List[List[Union[float, int, str]]]) -> List[List[Union[float, int, str]]]:
    """Suppress near-duplicate detections based on center distance and angle."""
    kept: List[List[Union[float, int, str]]] = []
    valid_rows = [row for row in rows if float(row[0]) != -1.0]
    
    for row in sorted(valid_rows, key=lambda k: float(k[3]), reverse=True):
        accept = True
        for kept_row in kept:
            dx = float(row[0]) - float(kept_row[0])
            dy = float(row[1]) - float(kept_row[1])
            distance = math.hypot(dx, dy)
            thr = 0.5 * min(float(row[3]), float(kept_row[3]))
            dang = abs(float(row[4]) - float(kept_row[4]))
            
            if distance < thr and dang < 20.0:
                accept = False
                break
        
        if accept:
            kept.append(row)
    
    if not kept:
        image_name = str(rows[0][5]) if rows else ''
        return [-1, -1, -1, -1, -1, image_name, -1]
    
    return kept

def safe_len(iterable: Iterable[object]) -> int:
    """Helper to make len type-safe for Pylance on unknown iterables."""
    return len(list(iterable))


def process_image(
    image_path: Path,
    cfg: Dict[str, Any],
) -> Union[List[List[Union[float, int, str]]], List]:
    """Process a single image and return crater rows."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [-1, -1, -1, -1, -1, image_path.name, -1]
    
    height, width = img.shape[:2]
    mindim = min(height, width)
    rows: List[List[Union[float, int, str]]] = []
    
    scales: Sequence[float] = cfg.get('scales', (1.0,))
    
    for scale in scales:
        sf = float(scale)
        scaled_h = int(round(height * sf))
        scaled_w = int(round(width * sf))
        
        imgs = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        enhanced = enhance_image(imgs.astype(np.uint8))
        edges = detect_edges(
            enhanced,
            float(cfg.get('canny_low_ratio', 0.5)),
            float(cfg.get('canny_high_ratio', 1.5)),
        )
        closed = close_edges(edges)
        
        # Try skimage ellipse detection first
        try:
            from skimage.feature import canny as sk_canny  # type: ignore
            from skimage.transform import hough_ellipse  # type: ignore
            
            edges_bool: NDArray[np.bool_] = sk_canny(enhanced.astype(np.float32) / 255.0)
            raw_result: Sequence[Tuple[float, float, float, float, float]] = cast(
                Sequence[Tuple[float, float, float, float, float]],
                hough_ellipse(edges_bool, accuracy=20, threshold=50, min_size=int(80 * sf), max_size=int(mindim * sf))
            )
            result: Sequence[Tuple[float, float, float, float, float]] = raw_result
            
            if result and safe_len(result) > 0:
                result_sorted: List[Tuple[float, float, float, float, float]] = sorted(
                    list(result),
                    key=lambda rh: float(rh[1]),
                    reverse=True,
                )[:10]
                
                for rh in result_sorted:
                    cys, cxs, as_, bs, theta = float(rh[0]), float(rh[1]), float(rh[2]), float(rh[3]), float(rh[4])
                    cx = cxs / sf
                    cy = cys / sf
                    a = as_ / sf
                    b = bs / sf
                    angle_deg = float(np.degrees(theta))
                    
                    if b < 40.0:
                        continue
                    if 2.0 < a / (b + 0.001) > 0.6 * mindim:
                        continue
                    if ellipse_touches_border(cx, cy, a, b, angle_deg, width, height):
                        continue
                    
                    rows.append([cx, cy, a, b, angle_deg, image_path.name, -1])
        except Exception:
            pass
        
        # Try HoughCircles detection
        try:
            min_r = int(40 * sf)
            max_r = max(min_r + 1, int(0.3 * mindim * sf))
            circles = cv2.HoughCircles(
                enhanced,
                cv2.HOUGH_GRADIENT,
                dp=float(cfg.get('hough_dp', 1.0)),
                minDist=max(int(cfg.get('hough_min_dist', 16)), 1),
                param1=int(cfg.get('hough_param1', 80.0)),
                param2=int(cfg.get('hough_param2', 18.0)),
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None:
                circles_uint16: NDArray[np.uint16] = np.asarray(np.around(circles), dtype=np.uint16)
                for x_coord, y_coord, rc in circles_uint16[0, :].astype(np.int32).tolist():
                    cx = float(x_coord) / sf
                    cy = float(y_coord) / sf
                    a = float(rc) / sf
                    b = float(rc) / sf
                    angle_deg = 0.0
                    
                    if b < 40.0:
                        continue
                    if 2.0 < a / (b + 0.001) > 0.6 * mindim:
                        continue
                    if ellipse_touches_border(cx, cy, a, b, angle_deg, width, height):
                        continue
                    
                    rows.append([cx, cy, a, b, angle_deg, image_path.name, -1])
        except Exception:
            pass
        
        # Contour-based detection
        contours_info = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours_info) == 3:
            contours = contours_info[1]
        else:
            contours = contours_info[0]
        
        for contour in contours:
            ellipse_params = contour_to_ellipse(np.asarray(contour, dtype=np.int32))
            if ellipse_params is None:
                continue
            
            cx, cy, a, b, angle_deg = ellipse_params
            cx /= sf
            cy /= sf
            a /= sf
            b /= sf
            
            if b < 40.0:
                continue
            if 2.0 < a / (b + 0.001) > 0.6 * mindim:
                continue
            if ellipse_touches_border(cx, cy, a, b, angle_deg, width, height):
                continue
            
            class_id = classify_crater_rim(enhanced.astype(np.uint8), np.asarray(contour, dtype=np.int32)) if use_classifier else -1
            rows.append([cx, cy, a, b, angle_deg, image_path.name, class_id])
    
    if not rows:
        return [-1, -1, -1, -1, -1, image_path.name, -1]
    
    return suppress_duplicates(rows)

def make_lunar(
    size: int = 1024,
    num_craters: int = 40,
    seed: Optional[int] = None,
) -> NDArray[np.uint8]:
    """Create a synthetic lunar-like height field rendered to a grayscale image."""
    if seed is not None:
        np.random.seed(seed)
    
    hval, wval = size, size
    base = np.random.randn(hval, wval).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=16, sigmaY=16)
    height_field = base * 0.5
    
    for _ in range(num_craters):
        cx = np.random.randint(int(wval * 0.05), int(wval * 0.95))
        cy = np.random.randint(int(hval * 0.05), int(hval * 0.95))
        radius = int(np.random.uniform(wval * 0.01, wval * 0.12))
        depth = np.random.uniform(0.3, 1.2)
        radius = wval * 0.1
        
        yidx, xidx = np.ogrid[:hval, :wval]
        dist = np.sqrt((xidx - cx) ** 2 + (yidx - cy) ** 2)
        sigma = radius / 2.5
        depression = -depth * np.exp(-0.5 * (dist / sigma) ** 2)
        height_field = depression + height_field
        
        rim_width = max(2, int(radius * 0.12))
        rim = np.exp(-0.5 * ((dist - radius) / rim_width) ** 2)
        rim = rim * depth * 0.6
        height_field = height_field + rim
    
    h_min, h_max = float(height_field.min()), float(height_field.max())
    norm = (height_field - h_min) / (h_max - h_min + 1e-9)
    
    gy, gx = np.gradient(norm)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx = -gx * nz
    ny = -gy * nz
    
    lx, ly, lz = -0.5, -0.3, 0.8
    l_norm = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx /= l_norm
    ly /= l_norm
    lz /= l_norm
    
    diffuse = nx * lx + ny * ly + nz * lz
    diffuse = np.clip(diffuse, 0.0, 1.0)
    diffuse = diffuse.astype(np.float32)
    
    ambient = 0.15
    img = ambient + 0.9 * diffuse
    img = np.clip(img, 0.0, 1.0)
    img = img.astype(np.float32)
    
    noise = np.random.randn(hval, wval) * 0.02
    noise = noise.astype(np.float32)
    img = img + noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    yy, xx = np.indices((hval, wval))
    cxv, cyv = wval / 2.0, hval / 2.0
    rv = np.sqrt((xx - cxv) ** 2 + (yy - cyv) ** 2)
    vignette = 1.0 - 0.5 * (rv / ((wval + hval) / 4)) ** 2
    vignette = np.clip(vignette, 0.6, 1.0)
    vignette = vignette.astype(np.float32)
    img = img * vignette
    
    out: NDArray[np.uint8] = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)  # type: ignore
    return out

def create_synthetic_images(folder_path: str, verbose: bool = False) -> List[Path]:
    """Create two synthetic lunar images in the given folder."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    
    p1 = folder / 'lunar_test_small.png'
    p2 = folder / 'lunar_test_large.png'
    
    img1 = make_lunar(size=512, num_craters=18, seed=42)
    img2 = make_lunar(size=1024, num_craters=80, seed=123)
    
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)
    
    if verbose:
        print(f'Created synthetic lunar images: {p1}, {p2}')
    
    return [p1, p2]


def main() -> None:
    """Entry point."""
    args = parse_args()
    image_folder: str = args.image_folder
    output_csv: str = args.output_csv
    verbose: bool = bool(getattr(args, 'verbose', False))
    decimals: int = int(getattr(args, 'decimals', 2))
    generate_test: bool = bool(getattr(args, 'generate_test_images', False))
    visualize_folder: Optional[str] = getattr(args, 'visualize_folder', None)
    
    global use_classifier
    use_classifier = not bool(getattr(args, 'no_classification', False))
    
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
            print('Generating synthetic test images...')
        create_synthetic_images(image_folder, verbose=verbose)
    
    if verbose:
        print(f'Looking for PNG images in {image_folder}')
    
    images = list_png_images(image_folder)
    
    if not images:
        if verbose:
            print(f'No PNG images found in {image_folder}. Generating synthetic test images...')
        create_synthetic_images(image_folder, verbose=verbose)
        images = list_png_images(image_folder)
    
    if not images:
        csv_cols = [
            'ellipseCenterXpx',
            'ellipseCenterYpx',
            'ellipseSemimajorpx',
            'ellipseSemiminorpx',
            'ellipseRotationdeg',
            'inputImage',
            'crater_classification',
        ]
        df_empty = pd.DataFrame(columns=pd.Index(csv_cols))
        df_empty.to_csv(output_csv, index=False)
        print(f'Still no PNG images found. Created empty CSV: {output_csv}')
        return
    
    csv_cols = [
        'ellipseCenterXpx',
        'ellipseCenterYpx',
        'ellipseSemimajorpx',
        'ellipseSemiminorpx',
        'ellipseRotationdeg',
        'inputImage',
        'crater_classification',
    ]
    
    cfg: Dict[str, Any] = {
        'canny_low_ratio': float(getattr(args, 'canny_low_ratio', 0.5)),
        'canny_high_ratio': float(getattr(args, 'canny_high_ratio', 1.5)),
        'hough_dp': float(getattr(args, 'hough_dp', 1.0)),
        'hough_param1': float(getattr(args, 'hough_param1', 80.0)),
        'hough_param2': float(getattr(args, 'hough_param2', 18.0)),
        'hough_min_dist': int(getattr(args, 'hough_min_dist', 16)),
        'scales': tuple(float(s) for s in str(getattr(args, 'scales', '1.0,0.75,0.5')).split(',') if s),
    }
    
    for img_path in images:
        if total_rows >= MAX_ROWS:
            if verbose:
                print(f'Row limit ({MAX_ROWS}) reached. Stopping image processing.')
            break
        
        if verbose:
            print(f'Processing {img_path.name}')
        
        rows = process_image(img_path, cfg)
        
        if not rows or (isinstance(rows[0], list) and float(rows[0][0]) == -1.0):
            continue
        
        available_slots = MAX_ROWS - total_rows
        rows = rows[:available_slots]
        
        df = pd.DataFrame(rows, columns=pd.Index(csv_cols))
        
        if visualize_folder:
            img_color = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if isinstance(img_color, np.ndarray):
                img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
                
                for row in rows:
                    if float(row[0]) != -1.0:
                        cx, cy, a, b, angle_deg = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                        center = (int(round(cx)), int(round(cy)))
                        axes = (int(round(a)), int(round(b)))
                        cv2.ellipse(img_color, center, axes, float(angle_deg), 0, 360, (0, 255, 0), 2)
                
                out_path = Path(visualize_folder) / img_path.name
                cv2.imwrite(str(out_path), img_color)
        
        num_cols = [
            'ellipseCenterXpx',
            'ellipseCenterYpx',
            'ellipseSemimajorpx',
            'ellipseSemiminorpx',
            'ellipseRotationdeg',
        ]
        
        NumberLike = Union[int, float, str]
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: -1 if isinstance(v, (int, float)) and float(v) == -1.0 or str(v) == '-1' else f'{float(cast(NumberLike, v)):.{decimals}f}'  # type: ignore
                )
        
        if not csv_written:
            df.to_csv(output_csv, index=False, mode='w')
            csv_written = True
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)
        
        total_rows += len(df)
        
        if verbose:
            print(f'Wrote {len(df)} rows. Total so far: {total_rows}')
    
    if csv_written:
        print(f'Processed {len(images)} images. Results saved to {output_csv}')
        print(f'Total rows written: {total_rows} / {MAX_ROWS}')
    else:
        df_empty = pd.DataFrame(columns=pd.Index(csv_cols))
        df_empty.to_csv(output_csv, index=False)
        print(f'No detections. Created empty CSV: {output_csv}')


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    elapsed = end - start
    print(f'Total runtime: {elapsed:.3f} seconds')
