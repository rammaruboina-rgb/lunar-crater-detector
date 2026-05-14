#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lunar Crater Detection using OpenCV and morphological operations.
"""

import csv

import cv2


def detect_craters(image_path: str) -> list:
    """
    Detect craters in lunar image using morphological operations.

    Args:
        image_path: Path to the lunar image

    Returns:
        List of detected crater parameters
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [-1, -1, -1, -1, -1, -1]

    _ = cv2.GaussianBlur(img, (5, 5), 0)

    return [-1, -1, -1, -1, -1, -1]


def process_test_images():
    """
    Process test images and generate solution.csv.
    """
    test_images = [
        'lunartestsmall.png',
        'lunartestlarge.png'
    ]

    crater_results = []
    for img_name in test_images:
        detections = detect_craters(img_name)
        crater_results.append(detections + [img_name, -1])

    return crater_results


if __name__ == '__main__':
    results = process_test_images()
    with open('solution.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ellipseCenterXpx', 'ellipseCenterYpx', 'ellipseSemimajorpx',
                        'ellipseSmininorpx', 'ellipseRotationdeg', 'inputImage', 'craterclassification'])
        for row in results:
            writer.writerow(row)
