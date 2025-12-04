#!/usr/bin/env python3
"""Test script for lunar crater detection.

This script demonstrates the crater_detector module by:
1. Generating synthetic lunar-like test images
2. Running crater detection on the test images
3. Displaying and saving the results to CSV
"""

import os
import sys
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from crater_detector import (
    enhance_image, detect_edges, close_edges, contour_to_ellipse,
    ellipse_touches_border, classify_crater_rim, process_image
)


def create_synthetic_lunar_image(width=512, height=512, num_craters=5):
    """Generate a synthetic lunar-like image with circular craters.
    
    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        num_craters (int): Number of craters to generate.
        
    Returns:
        numpy.ndarray: Synthetic lunar image.
    """
    # Create base image with lunar surface texture
    image = np.ones((height, width), dtype=np.uint8) * 100
    
    # Add noise for texture
    noise = np.random.randint(0, 30, (height, width), dtype=np.uint8)
    image = cv.add(image, noise)
    
    # Add random craters
    for _ in range(num_craters):
        center_x = np.random.randint(50, width - 50)
        center_y = np.random.randint(50, height - 50)
        radius = np.random.randint(20, 80)
        
        # Draw crater rim (brighter edge)
        cv.circle(image, (center_x, center_y), radius, 180, 2)
        
        # Draw crater interior (darker)
        cv.circle(image, (center_x, center_y), radius - 5, 70, -1)
    
    # Apply blur for realistic appearance
    image = cv.GaussianBlur(image, (5, 5), 1.0)
    
    return image


def run_test():
    """Run the crater detection test."""
    print("=" * 60)
    print("LUNAR CRATER DETECTION - TEST DEMONSTRATION")
    print("=" * 60)
    
    # Create test directory
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate synthetic test images
    print("\n[1] Generating synthetic lunar images...")
    num_test_images = 3
    for i in range(num_test_images):
        print(f"  - Creating test_image_{i}.png with {5 + i*2} craters")
        lunar_image = create_synthetic_lunar_image(512, 512, 5 + i*2)
        output_path = test_dir / f"test_image_{i}.png"
        cv.imwrite(str(output_path), lunar_image)
    
    print(f"\n✓ Generated {num_test_images} test images in '{test_dir}' directory")
    
    # Run crater detection
    print("\n[2] Running crater detection on test images...")
    all_detections = []
    
    for img_file in sorted(test_dir.glob("*.png")):
        print(f"  - Processing: {img_file.name}")
        detections = process_image(img_file)
        
        # Display detection results
        num_craters = len(detections) - (1 if detections[0][0] == -1 else 0)
        print(f"    → Detected {num_craters} crater(s)")
        
        for detection in detections:
            print(f"      Center: ({detection[0]:.1f}, {detection[1]:.1f}), "
                  f"Semi-axes: {detection[2]:.1f}, {detection[3]:.1f}, "
                  f"Angle: {detection[4]:.1f}°")
        
        all_detections.extend(detections)
    
    # Save results to CSV
    print("\n[3] Saving results to CSV...")
    csv_path = output_dir / "crater_detections.csv"
    df = pd.DataFrame(
        all_detections,
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
    df.to_csv(csv_path, index=False)
    print(f"✓ Results saved to: {csv_path}")
    
    # Display summary statistics
    print("\n[4] DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {num_test_images}")
    print(f"Total detections: {len(all_detections)}")
    
    valid_detections = df[df['center_x'] != -1]
    print(f"Valid crater detections: {len(valid_detections)}")
    
    if len(valid_detections) > 0:
        print(f"\nCrater Statistics:")
        print(f"  Semi-major axis (pixels):")
        print(f"    - Min: {valid_detections['semi_major'].min():.2f}")
        print(f"    - Max: {valid_detections['semi_major'].max():.2f}")
        print(f"    - Mean: {valid_detections['semi_major'].mean():.2f}")
        print(f"  Semi-minor axis (pixels):")
        print(f"    - Min: {valid_detections['semi_minor'].min():.2f}")
        print(f"    - Max: {valid_detections['semi_minor'].max():.2f}")
        print(f"    - Mean: {valid_detections['semi_minor'].mean():.2f}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("=" * 60)
    
    # Display CSV content
    print("\n[5] DETECTION RESULTS (CSV)")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\n" + "=" * 60)
    
    return df


if __name__ == "__main__":
    results_df = run_test()
    print("\n✓ All tests completed. Results saved to 'test_results/crater_detections.csv'")
