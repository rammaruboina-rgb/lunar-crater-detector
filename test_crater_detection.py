#!/usr/bin/env python3
"""
Test runner for crater detection validation.
Process images, validate against ground truth, and report scores.
"""

import sys
from pathlib import Path
from typing import TypedDict, Dict, List, Optional, Union

import argparse
import pandas as pd

try:
    from crater_validator import scoreDetections
except ImportError:
    scoreDetections = None


class CraterEllipse(TypedDict):
    """Type definition for crater ellipse coordinates."""
    cx: float
    cy: float
    a: float
    b: float
    angle: float


class DetectionScore(TypedDict):
    """Type definition for detection scoring results."""
    image: str
    ground_truth: int
    detected: int
    matched: int
    false_positives: int
    false_negatives: int
    avg_iou_score: float
    precision: float
    recall: float
    f1: float


class DetectionOnly(TypedDict):
    """Type definition for detection-only results."""
    detected: int


def run_test(image_folder: str, output_csv: str,
             ground_truth_csv: Optional[str] = None,
             verbose: bool = False) -> bool:
    """
    Run crater detection and score against ground truth if available.

    Args:
        image_folder: Path to folder containing images.
        output_csv: Path to output CSV file.
        ground_truth_csv: Optional path to ground truth CSV.
        verbose: Enable verbose output.

    Returns:
        True if test passed, False otherwise.
    """
    print("Crater Detection Test Runner")
    print(f"Images: {image_folder}")
    print(f"Output: {output_csv}")
    if ground_truth_csv:
        print(f"Ground truth: {ground_truth_csv}")
    print()

    if not Path(output_csv).exists():
        print(f"ERROR: {output_csv} not found. Run detector first.")
        return False

    detections_df = pd.read_csv(output_csv)
    print(f"Loaded {len(detections_df)} detection rows from {output_csv}")

    results: Dict[str, Union[DetectionScore, DetectionOnly]] = {}

    for img_name in detections_df['inputImage'].unique():
        img_detections = detections_df[
            detections_df['inputImage'] == img_name
        ]

        ellipses: List[CraterEllipse] = []
        for _, row in img_detections.iterrows():
            try:
                # FIXED: Match actual column names from crater_detector.py
                cx_val = float(row['ellipseCenterX(px)'])
                if cx_val == -1:
                    continue
                ellipses.append({
                    'cx': cx_val,
                    'cy': float(row['ellipseCenterY(px)']),
                    'a': float(row['ellipseSemimajor(px)']),
                    'b': float(row['ellipseSemiminor(px)']),
                    'angle': float(row['ellipseRotation(deg)'])
                })
            except (ValueError, KeyError) as e:
                if verbose:
                    print(f"Warning: Skipping invalid row in {img_name}: {e}")
                continue

        if ground_truth_csv and Path(ground_truth_csv).exists():
            if scoreDetections:
                score = scoreDetections(str(img_name), ellipses,
                                       ground_truth_csv)
                if score:
                    results[img_name] = score
                    if verbose:
                        _print_detection_info(img_name, score)
        else:
            results[img_name] = {'detected': len(ellipses)}

    print("\nSummary")
    print(f"Images processed: {len(results)}")

    if ground_truth_csv and Path(ground_truth_csv).exists():
        return _print_metrics_and_check(results, ground_truth_csv)

    total_detected = sum(
        r.get('detected', 0) for r in results.values()
    )
    print(f"Total craters detected: {total_detected}")
    return True


def _print_detection_info(img_name: str, score: DetectionScore) -> None:
    """Print detection information for an image."""
    print(f"\n{img_name}")
    print(f"  GT: {score.get('ground_truth', 0)} craters")
    print(f"  Detected: {score.get('detected', 0)}, "
          f"Matched: {score.get('matched', 0)}")
    print(f"  Precision: {score.get('precision', 0):.3f}, "
          f"Recall: {score.get('recall', 0):.3f}, "
          f"F1: {score.get('f1', 0):.3f}")
    print(f"  Avg IoU: {score.get('avg_iou_score', 0):.3f}")

def your_test_function(_ground_truth_csv):
def _print_metrics_and_check(
    results: Dict[str, Union[DetectionScore, DetectionOnly]],
    _ground_truth_csv: str) -> bool:
    """Print metrics and check if test passed."""
    total_matched = sum(
        r.get('matched', 0) for r in results.values()
    )
    total_gt = sum(
        r.get('ground_truth', 0) for r in results.values()
    )
    total_detected = sum(
        r.get('detected', 0) for r in results.values()
    )
    avg_f1 = sum(
        r.get('f1', 0) for r in results.values()
    ) / len(results) if results else 0
    avg_iou = sum(
        r.get('avg_iou_score', 0) for r in results.values()
    ) / len(results) if results else 0

    print(f"Total GT craters: {total_gt}")
    print(f"Total detected: {total_detected}")
    print(f"Total matched: {total_matched}")
    print(f"Average F1 Score: {avg_f1:.3f}")
    print(f"Average IoU Score: {avg_iou:.3f}")

    return avg_f1 >= 0.7


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test crater detector against ground truth.'
    )
    parser.add_argument('image_folder', nargs='?', default='lunar_images',
                        help='Folder with PNG images')
    parser.add_argument('output_csv', nargs='?', default='solution.csv',
                        help='Output CSV file')
    parser.add_argument('--ground_truth', type=str,
                        help='Ground truth CSV file')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()
    success = run_test(args.image_folder, args.output_csv,
                       args.ground_truth, args.verbose)
    sys.exit(0 if success else 1)
