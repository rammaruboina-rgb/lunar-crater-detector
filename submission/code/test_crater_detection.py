#!/usr/bin/env python3
"""
Test suite for crater detection module.
"""

import unittest
from crater_detector import detect_craters, process_test_images


class TestCraterDetection(unittest.TestCase):
    """
    Test cases for crater detection functions.
    """

    def test_detect_craters_returns_list(self):
        """
        Test that detect_craters returns a list.
        """
        result = detect_craters('dummy_path.png')
        self.assertIsInstance(result, list)

    def test_process_test_images(self):
        """
        Test that process_test_images returns results.
        """
        results = process_test_images()
        self.assertIsInstance(results, list)


if __name__ == '__main__':
    unittest.main()
