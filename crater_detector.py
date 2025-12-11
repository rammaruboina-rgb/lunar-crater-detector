from __future__ import annotations
import os
import argparse
import math
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Global is now a module-level variable, not modified with 'global' in main()
_use_classifier: bool = True
MAX_ROWS: int = 500_000

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatic crater detection on grayscale lunar PNG images."
    )
    parser.add_argument(
        "image_folder",
        type=str,
        nargs="?",
        default="lunar_images",
        help="Folder containing PNG images (default: lunar_images)",
    )
    parser.add_argument(
        "output_csv",
        type=str,
        nargs="?",
        default="solution.csv",
        help="Output CSV path (default: solution.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    parser.add_argument(
        "--generate_test_images",
        action="store_true",
        help="Create synthetic PNG test images inside the image folder before processing.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places for numeric output (default: 2).",
    )
    parser.add_argument(
        "--visualize_folder",
        type=str,
        default=None,
        help="Folder to save annotated images.",
    )
    parser.add_argument(
        "--no_classification",
        action="store_true",
        help="Disable crater rim classification.",
    )
    parser.add_argument(
        "--canny_low_ratio",
        type=float,
        default=0.5,
        help="Canny lower threshold ratio to median.",
    )
    parser.add_argument(
        "--canny_high_ratio",
        type=float,
        default=1.5,
        help="Canny upper threshold ratio to median.",
    )
    parser.add_argument(
        "--hough_dp",
        type=float,
        default=1.0,
        help="HoughCircles dp.",
    )
    parser.add_argument(
        "--hough_param1",
        type=float,
        default=80.0,
        help="HoughCircles param1.",
    )
    parser.add_argument(
        "--hough_param2",
        type=float,
        default=18.0,
        help="HoughCircles param2.",
    )
    parser.add_argument(
        "--hough_minDist",
        type=int,
        default=16,
        help="HoughCircles minDist.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="1.0,0.75,0.5",
        help="Comma-separated scales for multi-scale detection.",
    )
    return parser.parse_args()


def list_png_images(folder: str) -> List[Path]:
    """Return sorted PNG image paths inside a folder."""
    folder_path = Path(folder)
    return sorted([p for p in folder_path.glob("*.png") if p.is_file()])


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
    dil = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return np.asarray(dil, dtype=np.uint8)


def contour_to_ellipse(
    contour: NDArray[np.int32],
) -> Optional[Tuple[float, float, float, float, float]]:
    """Fit an ellipse to a contour and return (cx, cy, a, b, angle_deg)."""
    if contour.shape[0] < 5:
        return None

    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
    a = max(w, h) / 2.0
    b = min(w, h) / 2.0

    if h > w:
        angle = (angle + 90.0) % 180.0

    return float(cx), float(cy), float(a), float(b), float(angle)


def ellipse_touches_border(
    cx: float,
    cy: float,
    a: float,
    b: float,
    angle_deg: float,
    size: Tuple[int, int],
) -> bool:
    """Return True if ellipse extends beyond image border."""
    width, height = size
    theta = math.radians(angle_deg)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))

    half_w = a * cos_t + b * sin_t
    half_h = a * sin_t + b * cos_t

    x_min = cx - half_w
    x_max = cx + half_w
    y_min = cy - half_h
    y_max = cy + half_h

    return bool(
        x_min <= 0.0
        or y_min <= 0.0
        or x_max >= float(width - 1)
        or y_max >= float(height - 1)
    )


def classify_crater_rim(
    enhanced: NDArray[np.uint8],
    contour: NDArray[np.int32],
    use_classifier_flag: bool,
) -> int:
    """
    Classify crater rim steepness heuristically into 0–4.

    Returns -1 if classification is disabled or not meaningful.
    """
    if not use_classifier_flag:
        return -1

    grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)

    pts: NDArray[np.int32] = contour.reshape(-1, 2)
    h, w = enhanced.shape
    vals: List[float] = []
    for x_coord, y_coord in pts:
        ix = int(round(float(x_coord)))
        iy = int(round(float(y_coord)))
        if 0 <= ix < w and 0 <= iy < h:
            vals.append(float(mag[iy, ix]))

    if not vals:
        return -1

    mean_mag = float(np.mean(vals))
    normalized = min(max(mean_mag / 40.0, 0.0), 1.0)
    cls = int(round(normalized * 4.0))
    return max(0, min(4, cls))


def suppress_duplicates(
    rows: List[List[Union[float, int, str]]],
) -> List[List[Union[float, int, str]]]:
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
        image_name = str(rows[0][5]) if rows else ""
        return [[-1, -1, -1, -1, -1, image_name, -1]]
    return kept


def _safe_len(iterable: Iterable[object]) -> int:
    """Helper to make len() type-safe for Pylance on unknown iterables."""
    return len(list(iterable))


# --- Helper functions for process_image ---

def _detect_ellipses_hough(
    enhanced: NDArray[np.uint8],
    image_path_name: str,
    s_f: float,
    min_dim: int,
    image_size: Tuple[int, int],
) -> List[List[Union[float, int, str]]]:
    """Perform Hough Ellipse detection using scikit-image."""
    rows: List[List[Union[float, int, str]]] = []
    width, height = image_size

    try:
        from skimage.feature import canny as sk_canny  # type: ignore[import-untyped]
        from skimage.transform import hough_ellipse  # type: ignore[import-untyped]
    except ImportError:
        return rows

    try:
        edges_bool: Any = sk_canny( # type: ignore
            enhanced.astype(np.float32) / 255.0
        )

        raw_result: Sequence[
            Tuple[float, float, float, float, float]
        ] = cast(
            Sequence[Tuple[float, float, float, float, float]],
            hough_ellipse(
                edges_bool,
                accuracy=20,
                threshold=50,
                min_size=int(80 * s_f),
                max_size=int(min_dim * s_f),
            ),
        )

        result: Sequence[
            Tuple[float, float, float, float, float]
        ] = raw_result

        if result and _safe_len(result) > 0:
            result_sorted: List[
                Tuple[float, float, float, float, float]
            ] = sorted(result, key=lambda r_h: float(r_h[1]), reverse=True)[
                :10
            ]
            for r_h in result_sorted:
                cy_s, cx_s, a_s, b_s, theta = (
                    float(r_h[0]),
                    float(r_h[1]),
                    float(r_h[2]),
                    float(r_h[3]),
                    float(r_h[4]),
                )
                cx = cx_s / s_f
                cy = cy_s / s_f
                a = a_s / s_f
                b = b_s / s_f
                angle_deg = float(np.degrees(theta))

                if b < 40.0:
                    continue
                if (2.0 * (a + b)) >= (0.6 * min_dim):
                    continue
                if ellipse_touches_border(cx, cy, a, b, angle_deg, (width, height)):
                    continue

                rows.append(
                    [
                        cx,
                        cy,
                        a,
                        b,
                        angle_deg,
                        image_path_name,
                        -1,
                    ]
                )
    except Exception:
        pass

    return rows


def _detect_circles_hough(
    enhanced: NDArray[np.uint8],
    image_path_name: str,
    s_f: float,
    min_dim: int,
    image_size: Tuple[int, int],
    cfg: Dict[str, Any],
) -> List[List[Union[float, int, str]]]:
    """Perform Hough Circle detection using OpenCV."""
    rows: List[List[Union[float, int, str]]] = []
    width, height = image_size

    try:
        min_r = int(40 * s_f)
        max_r = max(min_r + 1, int(0.3 * min_dim * s_f))
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=float(cfg.get("hough_dp", 1.0)),
            minDist=max(
                int(cfg.get("hough_minDist", 16)),
                int(min_dim * s_f) // 30,
            ),
            param1=int(cfg.get("hough_param1", 80)),
            param2=int(cfg.get("hough_param2", 18)),
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:  # type: ignore[truthy-function]
            circles_uint16: NDArray[np.uint16] = np.asarray(
                np.around(circles), dtype=np.uint16
            )
            for x_coord, y_coord, r_c in circles_uint16[0, :].astype(
                np.int32
            ).tolist():
                cx = float(x_coord) / s_f
                cy = float(y_coord) / s_f
                a = float(r_c) / s_f
                b = float(r_c) / s_f
                angle_deg = 0.0

                if b < 40.0:
                    continue
                if (2.0 * (a + b)) >= (0.6 * min_dim):
                    continue
                if ellipse_touches_border(cx, cy, a, b, angle_deg, (width, height)):
                    continue

                rows.append([cx, cy, a, b, angle_deg, image_path_name, -1])
    except Exception:
        pass

    return rows


def _detect_contours_fit(
    closed_edges: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    image_path_name: str,
    s_f: float,
    min_dim: int,
    image_size: Tuple[int, int],
    use_classifier_flag: bool,
) -> List[List[Union[float, int, str]]]:
    """Perform contour finding and ellipse fitting using OpenCV."""
    rows: List[List[Union[float, int, str]]] = []
    width, height = image_size

    contours_info = cv2.findContours(
        closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )
    if len(contours_info) == 3:
        contours = contours_info[1]
    else:
        contours, _ = contours_info

    for contour in contours:
        ellipse_params = contour_to_ellipse(
            np.asarray(contour, dtype=np.int32)
        )
        if ellipse_params is None:
            continue

        cx_s, cy_s, a_s, b_s, angle_deg = ellipse_params
        cx = cx_s / s_f
        cy = cy_s / s_f
        a = a_s / s_f
        b = b_s / s_f

        if b < 40.0:
            continue
        if (2.0 * (a + b)) >= (0.6 * min_dim):
            continue
        if ellipse_touches_border(cx, cy, a, b, angle_deg, (width, height)):
            continue

        class_id = classify_crater_rim(
            enhanced.astype(np.uint8),
            np.asarray(contour, dtype=np.int32),
            use_classifier_flag,
        )

        rows.append(
            [cx, cy, a, b, angle_deg, image_path_name, class_id]
        )

    return rows


# --- Main processing function ---


def process_image(
    image_path: Path,
    cfg: Dict[str, Any],
) -> List[List[Union[float, int, str]]]:
    """Process a single image and return crater rows."""

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [[-1, -1, -1, -1, -1, image_path.name, -1]]

    height, width = img.shape[:2]
    min_dim = min(height, width)
    rows: List[List[Union[float, int, str]]] = []
    scales: Sequence[float] = cfg.get("scales", [1.0])
    image_size = (width, height)
    use_classifier_flag = cfg.get("use_classifier", False)

    for scale in scales:
        s_f = float(scale)
        scaled_h = int(round(height * s_f))
        scaled_w = int(round(width * s_f))
        img_s = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        enhanced = enhance_image(img_s.astype(np.uint8))
        edges = detect_edges(
            enhanced,
            float(cfg.get("canny_low_ratio", 0.5)),
            float(cfg.get("canny_high_ratio", 1.5)),
        )
        closed = close_edges(edges)

        # Accumulate detection_results from refactored functions
        rows.extend(
            _detect_ellipses_hough(
                enhanced, image_path.name, s_f, min_dim, image_size
            )
        )
        rows.extend(
            _detect_circles_hough(
                enhanced, image_path.name, s_f, min_dim, image_size, cfg
            )
        )
        rows.extend(
            _detect_contours_fit(
                closed, enhanced, image_path.name, s_f, min_dim, image_size, use_classifier_flag
            )
        )

    if not rows:
        return [[-1, -1, -1, -1, -1, image_path.name, -1]]

    return suppress_duplicates(rows)


def build_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration dictionary from parsed arguments."""
    return {
        "canny_low_ratio": float(getattr(args, "canny_low_ratio", 0.5)),
        "canny_high_ratio": float(getattr(args, "canny_high_ratio", 1.5)),
        "hough_dp": float(getattr(args, "hough_dp", 1.0)),
        "hough_param1": float(getattr(args, "hough_param1", 80.0)),
        "hough_param2": float(getattr(args, "hough_param2", 18.0)),
        "hough_minDist": int(getattr(args, "hough_minDist", 16)),
        "scales": [
            float(s)
            for s in str(getattr(args, "scales", "1.0,0.75,0.5")).split(",")
            if s
        ],
        "use_classifier": not bool(getattr(args, "no_classification", False)),
    }


def _make_lunar(
    size: int = 1024,
    num_craters: int = 40,
    seed: Optional[int] = None,
) -> NDArray[np.uint8]:
    """Create a synthetic lunar-like height field rendered to a grayscale image."""
    if seed is not None:
        np.random.seed(seed)
    h_val, w_val = size, size

    base: NDArray[np.float32] = np.random.randn(h_val, w_val).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=16, sigmaY=16) # pyright: ignore[reportAssignmentType]
    height_field: NDArray[np.float32] = base * 0.5

    def add_crater(height_field: NDArray[np.float32], w_val: int, h_val: int):
        """Helper to add a single crater."""
        cx = np.random.randint(int(w_val * 0.05), int(w_val * 0.95))
        cy = np.random.randint(int(h_val * 0.05), int(h_val * 0.95))

        radius = int(np.random.uniform(w_val * 0.01, w_val * 0.12))
        depth = np.random.uniform(0.3, 1.2) * (radius / (w_val * 0.1))

        y_idx, x_idx = np.ogrid[:h_val, :w_val]
        dist: NDArray[np.float64] = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

        sigma = radius / 2.5
        depression: NDArray[np.float64] = -depth * np.exp(-0.5 * (dist / sigma) ** 2)

        rim_width = max(2, int(radius * 0.12))
        rim: NDArray[np.float64] = np.exp(-0.5 * ((dist - radius) / rim_width) ** 2)
        rim = rim * (depth * 0.6)

        height_field += depression
        height_field += rim
        return height_field

    for _ in range(num_craters):
        height_field = add_crater(height_field, w_val, h_val)

    h_min, h_max = float(height_field.min()), float(height_field.max())
    norm: NDArray[np.float32] = (height_field - h_min) / (h_max - h_min + 1e-9)

    gy, gx = np.gradient(norm)
    nz: NDArray[np.float64] = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx: NDArray[np.float64] = -gx * nz
    ny: NDArray[np.float64] = -gy * nz

    lx, ly, lz = -0.5, -0.3, 0.8
    l_norm = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx /= l_norm
    ly /= l_norm
    lz /= l_norm

    diffuse: NDArray[np.float64] = (nx * lx + ny * ly + nz * lz)
    diffuse = np.clip(diffuse, 0.0, 1.0)
    diffuse_f32: NDArray[np.float32] = diffuse.astype(np.float32)

    ambient = 0.15
    img: NDArray[np.float32] = ambient + 0.9 * diffuse_f32
    img = np.clip(img, 0.0, 1.0)
    img = img.astype(np.float32)

    noise: NDArray[np.float32] = (np.random.randn(h_val, w_val) * 0.02).astype(np.float32)
    img = img + noise
    img = cv2.GaussianBlur(img, (3, 3), 0) # pyright: ignore[reportAssignmentType]

    yy, xx = np.indices((h_val, w_val))
    cx_v, cy_v = w_val / 2.0, h_val / 2.0
    rv: NDArray[np.float64] = np.sqrt(((xx - cx_v) / cx_v) ** 2 + ((yy - cy_v) / cy_v) ** 2)
    vignette: NDArray[np.float64] = 1.0 - 0.5 * (rv ** 2)
    vignette = np.clip(vignette, 0.6, 1.0)
    vignette_f32: NDArray[np.float32] = vignette.astype(np.float32)
    img = img * vignette_f32

    out: NDArray[np.uint8] = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def create_synthetic_images(
    folder_path: str,
    verbose: bool = False,
) -> List[Path]:
    """Create two synthetic lunar images in the given folder."""
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    p1 = folder / "lunar_test_small.png"
    p2 = folder / "lunar_test_large.png"
    img1 = _make_lunar(size=512, num_craters=18, seed=42)
    img2 = _make_lunar(size=1024, num_craters=80, seed=123)

    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)
    if verbose:
        print(f"Created synthetic lunar images: {p1}, {p2}")
    return [p1, p2]


# --- Test-compatible wrapper functions ---

def detect_craters(
    image_path: Union[str, Path],
    cfg: Optional[Dict[str, Any]] = None,
) -> List[List[Union[float, int, str]]]:
    """
    Detect craters in a single image.
    
    Wrapper function for test compatibility.
    """
    if cfg is None:
        cfg = {
            "canny_low_ratio": 0.5,
            "canny_high_ratio": 1.5,
            "hough_dp": 1.0,
            "hough_param1": 80.0,
            "hough_param2": 18.0,
            "hough_minDist": 16,
            "scales": [1.0, 0.75, 0.5],
            "use_classifier": True,
        }
    return process_image(Path(image_path), cfg)


def process_test_images(
    image_folder: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Process all test images in a folder and return DataFrame.
    
    Wrapper function for test compatibility.
    """
    if cfg is None:
        cfg = {
            "canny_low_ratio": 0.5,
            "canny_high_ratio": 1.5,
            "hough_dp": 1.0,
            "hough_param1": 80.0,
            "hough_param2": 18.0,
            "hough_minDist": 16,
            "scales": [1.0, 0.75, 0.5],
            "use_classifier": True,
        }
    
    images = list_png_images(image_folder)
    all_rows: List[List[Union[float, int, str]]] = []
    
    for img_path in images:
        detection_results = process_image(img_path, cfg)
        all_rows.extend(detection_results)
    
    csv_cols = [
        "ellipseCenterX(px)",
        "ellipseCenterY(px)",
        "ellipseSemimajor(px)",
        "ellipseSemiminor(px)",
        "ellipseRotation(deg)",
        "inputImage",
        "crater_classification",
    ]
    
    return pd.DataFrame(all_rows, columns=pd.Index(csv_cols))


def main() -> None:
    """Entry point."""
    args = parse_args()
    image_folder: str = args.image_folder
    output_csv: str = args.output_csv
    verbose: bool = bool(getattr(args, "verbose", False))
    decimals: int = int(getattr(args, "decimals", 2))
    generate_test: bool = bool(getattr(args, "generate_test_images", False))
    visualize_folder: Optional[str] = getattr(args, "visualize_folder", None)

    total_rows = 0
    csv_written = False
    cfg = build_cfg(args)

    try:
        out_parent = Path(output_csv).parent
        if str(out_parent) and not out_parent.exists():
            out_parent.mkdir(parents=True, exist_ok=True)
        if visualize_folder:
            vf = Path(visualize_folder)
            vf.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        print(f"Error setting up directories: {err}")

    if generate_test:
        if verbose:
            print("Generating synthetic test images...")
        create_synthetic_images(image_folder, verbose=verbose)

    if verbose:
        print(f"Looking for PNG images in: {image_folder}")
    images = list_png_images(image_folder)
    if not images:
        print(
            f"No PNG images found in '{image_folder}'. Generating synthetic test images..."
        )
        create_synthetic_images(image_folder, verbose=verbose)
        images = list_png_images(image_folder)
        if not images:
            csv_cols = [
                "ellipseCenterX(px)",
                "ellipseCenterY(px)",
                "ellipseSemimajor(px)",
                "ellipseSemiminor(px)",
                "ellipseRotation(deg)",
                "inputImage",
                "crater_classification",
            ]
            df_empty = pd.DataFrame(columns=pd.Index(csv_cols))
            df_empty.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"Still no PNG images found. Created empty CSV: {output_csv}")
            return

    csv_cols = [
        "ellipseCenterX(px)",
        "ellipseCenterY(px)",
        "ellipseSemimajor(px)",
        "ellipseSemiminor(px)",
        "ellipseRotation(deg)",
        "inputImage",
        "crater_classification",
    ]


    for img_path in images:
        if total_rows >= MAX_ROWS:
            if verbose:
                print(
                    f"Row limit ({MAX_ROWS}) reached. Stopping image processing."
                )
            break

        if verbose:
            print(f"Processing: {img_path.name}")
        detection_results = process_image(img_path, cfg)
        if not detection_results:
            continue

        available_slots = MAX_ROWS - total_rows
        detection_results = detection_results[:available_slots]

        df = pd.DataFrame(detection_results, columns=pd.Index(csv_cols))

        if visualize_folder:
            img_original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if isinstance(img_original, np.ndarray):
                img_color = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
                for row in detection_results:
                    if float(row[0]) == -1.0:
                        continue
                    cx, cy, a, b, angle_deg = (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                    )
                    center = (int(round(cx)), int(round(cy)))
                    axes = (int(round(a)), int(round(b)))
                    cv2.ellipse(
                        img_color,
                        center,
                        axes,
                        float(angle_deg),
                        0,
                        360,
                        (0, 255, 0),
                        2,
                    )
                out_path = Path(visualize_folder) / img_path.name
                cv2.imwrite(str(out_path), img_color)

        num_cols = [
            "ellipseCenterX(px)",
            "ellipseCenterY(px)",
            "ellipseSemimajor(px)",
            "ellipseSemiminor(px)",
            "ellipseRotation(deg)",
        ]

        def format_numeric_value(v: Any) -> str:
            """Format numeric values with proper decimal places."""
            if isinstance(v, (int, float)) and float(v) == -1.0:
                return "-1"
            if str(v) == "-1":
                return "-1"
            return f"{float(v):.{decimals}f}"

        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].apply(format_numeric_value)  # type: ignore[call-overload]

        if not csv_written:
            df.to_csv(output_csv, index=False, mode="w", encoding="utf-8")
            csv_written = True
        else:
            df.to_csv(output_csv, index=False, mode="a", header=False, encoding="utf-8")

        total_rows += len(df)
        if verbose:
            print(f"  Wrote {len(df)} rows. Total so far: {total_rows}")

    if csv_written:
        print(f"Processed images. Results saved to: {output_csv}")
        print(f"Total rows written: {total_rows} / {MAX_ROWS}")
    else:
        df_empty = pd.DataFrame(columns=pd.Index(csv_cols))
        df_empty.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"No detections. Created empty CSV: {output_csv}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    elapsed = end - start
    print(f"Total runtime: {elapsed:.3f} seconds")
