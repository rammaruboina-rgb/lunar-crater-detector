#!/usr/bin/env python3
"""
Automatic crater detection on grayscale lunar PNG images.

For each image:
- Detect crater-like ellipses using multi-scale edge detection and contour analysis.
- Output center X/Y, semimajor, semiminor, and rotation in degrees.
- Optionally classify craters by a simple rim-steepness heuristic into 0–4, or -1 if disabled.

If no craters are detected for an image, emit:
-1,-1,-1,-1,-1,<inputImage>,-1

Output: solution.csv with columns:
ellipseCenterX(px),ellipseCenterY(px),
ellipseSemimajor(px),ellipseSemiminor(px),
ellipseRotation(deg),inputImage,crater_classification
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Ellipse = Tuple[float, float, float, float, float]


@dataclass
class DetectorConfig:
    """Configuration values for crater detection and filtering."""

    scales: Tuple[float, float, float]
    canny_low_ratio: float
    canny_high_ratio: float
    canny_kernel: int
    min_radius_px: int
    max_radius_px: int
    min_axis_ratio: float
    max_axis_ratio: float
    min_score: float
    max_overlap: float
    classify: bool
    decimals: int
    verbose: bool

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Lunar crater detector: detect crater ellipses in PNG images "
        "and write solution.csv in NASA challenge format."
    )
    parser.add_argument(
        "image_folder",
        type=str,
        help="Folder containing input PNG images.",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        help="Path to output CSV file (solution.csv).",
    )
    parser.add_argument(
        "--no_classify",
        action="store_true",
        help="Disable crater classification; use -1 for all rows.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimals for numeric output fields (default: 2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image diagnostics.",
    )
    args = parser.parse_args(argv)
    return args


def build_config(args: argparse.Namespace) -> DetectorConfig:
    """Create a DetectorConfig from parsed CLI arguments."""
    scales = (1.0, 0.75, 0.5)
    return DetectorConfig(
        scales=scales,
        canny_low_ratio=0.5,
        canny_high_ratio=1.5,
        canny_kernel=3,
        min_radius_px=10,
        max_radius_px=800,
        min_axis_ratio=0.4,
        max_axis_ratio=1.0,
        min_score=0.5,
        max_overlap=0.5,
        classify=not args.no_classify,
        decimals=args.decimals,
        verbose=args.verbose,
    )

def iter_png_images(folder: str) -> Iterable[str]:
    """Yield PNG filenames in a folder, sorted for reproducibility."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    names = [name for name in os.listdir(folder) if name.lower().endswith(".png")]
    names.sort()
    for name in names:
        yield name


def load_grayscale(path: str) -> np.ndarray:
    """Load an image as grayscale float32 in [0, 1]."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    image = image.astype(np.float32) / 255.0
    return image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Improve local contrast using CLAHE."""
    uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(uint8)
    return equalized.astype(np.float32) / 255.0


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Sharpen the image using an unsharp masking kernel."""
    kernel = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    sharp = cv2.filter2D(uint8, -1, kernel)
    return sharp.astype(np.float32) / 255.0


def detect_edges(image: np.ndarray, cfg: DetectorConfig, scale: float) -> np.ndarray:
    """Run Canny edge detection at the given scale."""
    if scale != 1.0:
        height, width = image.shape
        new_size = (int(width * scale), int(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    v = np.median(image)
    lower = int(max(0.0, (1.0 - cfg.canny_low_ratio) * v * 255.0))
    upper = int(min(255.0, (1.0 + cfg.canny_high_ratio) * v * 255.0))
    edges = cv2.Canny(
        (image * 255.0).astype(np.uint8),
        lower,
        upper,
        apertureSize=cfg.canny_kernel,
    )
    return edges

def estimate_circle_score(contour: np.ndarray, center: Tuple[float, float]) -> float:
    """Estimate how circular a contour is around a given center."""
    cx, cy = center
    distances: List[float] = []
    for point in contour:
        px, py = point[0]
        dx = float(px) - cx
        dy = float(py) - cy
        distances.append(math.hypot(dx, dy))
    if not distances:
        return 0.0
    mean_r = float(sum(distances) / len(distances))
    if mean_r <= 0:
        return 0.0
    variance = float(sum((d - mean_r) ** 2 for d in distances) / len(distances))
    std_r = math.sqrt(variance)
    score = max(0.0, 1.0 - std_r / (mean_r + 1e-5))
    return score


def contour_to_ellipse(
    contour: np.ndarray,
    image_shape: Tuple[int, int],
    cfg: DetectorConfig,
) -> Optional[Ellipse]:
    """Convert a contour to a validated ellipse, or return None."""
    if contour.shape[0] < 5:
        return None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (major, minor), angle = ellipse

    if major < minor:
        major, minor = minor, major

    if major <= 0 or minor <= 0:
        return None

    height, width = image_shape
    if not (0.0 <= cx < width and 0.0 <= cy < height):
        return None

    radius = 0.5 * (major + minor)
    if radius < cfg.min_radius_px or radius > cfg.max_radius_px:
        return None

    axis_ratio = float(minor / max(major, 1e-5))
    if axis_ratio < cfg.min_axis_ratio or axis_ratio > cfg.max_axis_ratio:
        return None

    score = estimate_circle_score(contour, (cx, cy))
    if score < cfg.min_score:
        return None

    return (float(cx), float(cy), float(0.5 * major), float(0.5 * minor), float(angle))

def non_max_suppression(
    ellipses: List[Ellipse],
    max_overlap: float,
) -> List[Ellipse]:
    """Suppress overlapping ellipses with lower radii."""
    if not ellipses:
        return []

    indexed = []
    for idx, ellipse in enumerate(ellipses):
        cx, cy, a, b, angle = ellipse
        score = 0.5 * (a + b)
        indexed.append((idx, score, cx, cy, a, b, angle))

    indexed.sort(key=lambda item: item[1], reverse=True)

    kept_indices: List[int] = []
    for idx, _, cx, cy, a, b, angle in indexed:
        keep = True
        for kept_idx in kept_indices:
            _, _, kcx, kcy, ka, kb, _ = indexed[kept_idx]
            dx = cx - kcx
            dy = cy - kcy
            dist = math.hypot(dx, dy)
            radius = 0.5 * (a + b)
            keep_radius = 0.5 * (ka + kb)
            min_r = min(radius, keep_radius)
            if dist < max_overlap * min_r:
                keep = False
                break
        if keep:
            kept_indices.append(idx)

    result: List[Ellipse] = []
    for idx in kept_indices:
        _, _, cx, cy, a, b, angle = indexed[idx]
        result.append((cx, cy, a, b, angle))
    return result


def classify_crater(ellipse: Ellipse) -> int:
    """Simple heuristic crater classifier based on ellipse size."""
    _, _, a, b, _ = ellipse
    radius = 0.5 * (a + b)
    if radius < 20:
        return 0
    if radius < 40:
        return 1
    if radius < 80:
        return 2
    if radius < 160:
        return 3
    return 4


def process_image(
    image: np.ndarray,
    cfg: DetectorConfig,
) -> List[Ellipse]:
    """
    Detect crater-like ellipses in a single image.

    Returns a list of ellipses (cx, cy, semimajor, semiminor, angle_degrees).
    """
    height, width = image.shape
    pre = apply_clahe(image)
    pre = sharpen_image(pre)

    all_ellipses: List[Ellipse] = []

    for scale in cfg.scales:
        edges = detect_edges(pre, cfg, scale)
        if scale != 1.0:
            edges = cv2.resize(
                edges,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        contours, _ = cv2.findContours(
            edges,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_NONE,
        )
        for contour in contours:
            ellipse = contour_to_ellipse(contour, (height, width), cfg)
            if ellipse is not None:
                all_ellipses.append(ellipse)

    deduped = non_max_suppression(all_ellipses, cfg.max_overlap)
    return deduped

def ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of a file path exists."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def write_header(writer: csv.writer) -> None:
    """Write the CSV header row."""
    writer.writerow(
        [
            "ellipseCenterXpx",
            "ellipseCenterYpx",
            "ellipseSemimajorpx",
            "ellipseSemiminorpx",
            "ellipseRotationdeg",
            "inputImage",
            "craterclassification",
        ],
    )


def format_ellipse(
    ellipse: Ellipse,
    image_name: str,
    cfg: DetectorConfig,
) -> List[str]:
    """Format an ellipse into a CSV row with the requested decimal precision."""
    cx, cy, a, b, angle = ellipse
    fmt = f"{{:.{cfg.decimals}f}}"
    if cfg.classify:
        label = classify_crater(ellipse)
    else:
        label = -1
    return [
        fmt.format(cx),
        fmt.format(cy),
        fmt.format(a),
        fmt.format(b),
        fmt.format(angle),
        image_name,
        str(label),
    ]


def format_empty_row(image_name: str) -> List[str]:
    """Return a CSV row for an image with no detections."""
    return [
        "-1",
        "-1",
        "-1",
        "-1",
        "-1",
        image_name,
        "-1",
    ]


def run_detector(
    image_folder: str,
    output_csv: str,
    cfg: DetectorConfig,
) -> None:
    """Run crater detection on all PNG images and write the solution CSV."""
    ensure_parent_dir(output_csv)

    total_rows = 0
    images_processed = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        write_header(writer)

        for image_name in iter_png_images(image_folder):
            path = os.path.join(image_folder, image_name)
            try:
                image = load_grayscale(path)
            except ValueError as exc:
                print(f"[WARN] Skipping {image_name}: {exc}", file=sys.stderr)
                continue

            ellipses = process_image(image, cfg)
            images_processed += 1

            if not ellipses:
                if cfg.verbose:
                    print(f"[INFO] No detections for {image_name}")
                writer.writerow(format_empty_row(image_name))
                total_rows += 1
                continue

            for ellipse in ellipses:
                writer.writerow(format_ellipse(ellipse, image_name, cfg))
                total_rows += 1

            if cfg.verbose:
                print(
                    f"[INFO] {image_name}: {len(ellipses)} ellipse(s) "
                    f"written to CSV",
                )

    print(
        f"Processed {images_processed} image(s). "
        f"Results saved to: {output_csv}",
    )
    print(f"Total rows written: {total_rows}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    cfg = build_config(args)
    run_detector(args.image_folder, args.output_csv, cfg)


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.3f} seconds")
