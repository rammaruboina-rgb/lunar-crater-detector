#!/usr/bin/env python3
"""
Comprehensive test suite for lunar crater detection.

This module tests the CraterDetector class with synthetic and real imagery.
"""

import unittest
import tempfile
from pathlib import Path
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd

from crater_detector import CraterDetector, save_results_to_csv, generate_test_images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCraterDetector(unittest.TestCase):
    """Test cases for CraterDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = CraterDetector(
            min_radius=3,
            max_radius=100,
            verbose=False
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def create_synthetic_crater_image(
        self,
        width: int = 512,
        height: int = 512,
        num_craters: int = 5
    ) -> np.ndarray:
        """Generate synthetic lunar-like image with craters.

        Args:
            width: Image width
            height: Image height
            num_craters: Number of craters to generate

        Returns:
            Grayscale image with synthetic craters
        """
        # Create base image with lunar texture
        image = np.random.randint(50, 150, (height, width), dtype=np.uint8)

        # Add Gaussian smoothing for texture
        image = cv2.GaussianBlur(image, (5, 5), 1.0)

        # Generate random craters
        for _ in range(num_craters):
            center_x = np.random.randint(50, width - 50)
            center_y = np.random.randint(50, height - 50)
            radius = np.random.randint(10, 40)
            depth = np.random.randint(30, 100)

            # Draw crater rim (dark circle)
            cv2.circle(
                image,
                (center_x, center_y),
                radius,
                max(0, 100 - depth),
                2
            )

            # Draw crater floor (darker)
            cv2.circle(
                image,
                (center_x, center_y),
                radius - 5,
                max(0, 80 - depth),
                -1
            )

        return image

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = CraterDetector(
            min_radius=5,
            max_radius=150,
            verbose=True
        )
        self.assertEqual(detector.min_radius, 5)
        self.assertEqual(detector.max_radius, 150)
        self.assertTrue(detector.verbose)

    def test_clahe_application(self):
        """Test CLAHE enhancement."""
        image = self.create_synthetic_crater_image()
        enhanced = self.detector._apply_clahe(image)

        self.assertEqual(enhanced.shape, image.shape)
        self.assertTrue(enhanced.dtype == np.uint8)

    def test_edge_detection(self):
        """Test Canny edge detection."""
        image = self.create_synthetic_crater_image()
        edges = self.detector._detect_edges(image)

        self.assertEqual(edges.shape, image.shape)
        # Edges should be binary (0 or 255)
        unique_values = np.unique(edges)
        self.assertTrue(all(v in [0, 255] for v in unique_values))

    def test_morphological_operations(self):
        """Test morphological operations."""
        image = self.create_synthetic_crater_image()
        edges = self.detector._detect_edges(image)
        morph = self.detector._morphological_operations(edges)

        self.assertEqual(morph.shape, edges.shape)
        self.assertTrue(morph.dtype == np.uint8)

    def test_single_image_detection(self):
        """Test crater detection on single image."""
        # Create test image
        image = self.create_synthetic_crater_image(num_craters=10)
        test_image_path = self.test_dir / "test_crater.png"
        cv2.imwrite(str(test_image_path), image)

        # Detect craters
        craters = self.detector.detect(test_image_path)

        # Validate results
        self.assertIsInstance(craters, list)
        for crater in craters:
            self.assertIn('x', crater)
            self.assertIn('y', crater)
            self.assertIn('radius', crater)
            self.assertGreater(crater['radius'], 0)

    def test_batch_detection(self):
        """Test batch crater detection."""
        # Create multiple test images
        num_images = 3
        for i in range(num_images):
            image = self.create_synthetic_crater_image(num_craters=5)
            image_path = self.test_dir / f"test_image_{i}.png"
            cv2.imwrite(str(image_path), image)

        # Detect craters in batch
        results = self.detector.detect_batch(self.test_dir)

        # Validate results
        self.assertEqual(len(results), num_images)
        for image_name, craters in results.items():
            self.assertTrue(image_name.endswith('.png'))
            self.assertIsInstance(craters, list)

    def test_crater_validation(self):
        """Test that detected craters meet radius constraints."""
        image = self.create_synthetic_crater_image(num_craters=20)
        test_image_path = self.test_dir / "test_constraints.png"
        cv2.imwrite(str(test_image_path), image)

        craters = self.detector.detect(test_image_path)

        # All craters should be within radius constraints
        for crater in craters:
            radius = crater['radius']
            self.assertGreaterEqual(radius, self.detector.min_radius)
            self.assertLessEqual(radius, self.detector.max_radius)

    def test_csv_output(self):
        """Test CSV output functionality."""
        # Create test data
        results = {
            'image1.png': [
                {'x': 100, 'y': 150, 'radius': 20, 'major_axis': 40,
                 'minor_axis': 38, 'angle': 45, 'area': 1256}
            ],
            'image2.png': [
                {'x': 200, 'y': 250, 'radius': 25, 'major_axis': 50,
                 'minor_axis': 48, 'angle': 30, 'area': 1963}
            ]
        }

        csv_path = self.test_dir / "test_results.csv"
        save_results_to_csv(results, csv_path)

        # Verify CSV was created and has correct content
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertIn('ImageName', df.columns)
        self.assertIn('CraterX', df.columns)
        self.assertIn('CraterRadius', df.columns)

    def test_annotated_images_generation(self):
        """Test generation of annotated test images."""
        # Create test image
        image = self.create_synthetic_crater_image(num_craters=5)
        test_image_path = self.test_dir / "test_orig.png"
        cv2.imwrite(str(test_image_path), image)

        # Detect and get results
        craters = self.detector.detect(test_image_path)
        results = {'test_orig.png': craters}

        # Generate annotated images
        output_dir = self.test_dir / "annotated"
        generate_test_images(self.test_dir, results, output_dir)

        # Verify annotated images were created
        self.assertTrue(output_dir.exists())
        annotated_files = list(output_dir.glob('*.png'))
        self.assertGreater(len(annotated_files), 0)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty detector with different thresholds
        detector = CraterDetector(min_radius=1, max_radius=500)
        image = self.create_synthetic_crater_image(num_craters=1)
        test_image_path = self.test_dir / "test_edge.png"
        cv2.imwrite(str(test_image_path), image)

        craters = detector.detect(test_image_path)
        self.assertIsInstance(craters, list)

    def test_nonexistent_image(self):
        """Test handling of nonexistent image file."""
        nonexistent_path = self.test_dir / "nonexistent.png"
        craters = self.detector.detect(nonexistent_path)

        # Should return empty list for missing file
        self.assertEqual(craters, [])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete crater detection pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.detector = CraterDetector(verbose=True)

    def tearDown(self):
        """Clean up integration test fixtures."""
        self.temp_dir.cleanup()

    def test_complete_pipeline(self):
        """Test complete detection and output pipeline."""
        # Create test images
        for i in range(3):
            image = np.random.randint(50, 150, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(self.test_dir / f"image_{i}.png"), image)

        # Detect
        results = self.detector.detect_batch(self.test_dir)

        # Save to CSV
        csv_path = self.test_dir / "results.csv"
        save_results_to_csv(results, csv_path)

        # Generate annotated images
        output_dir = self.test_dir / "annotated"
        generate_test_images(self.test_dir, results, output_dir)

        # Verify all outputs
        self.assertEqual(len(results), 3)
        self.assertTrue(csv_path.exists())
        self.assertTrue(output_dir.exists())


if __name__ == '__main__':
    unittest.main()
