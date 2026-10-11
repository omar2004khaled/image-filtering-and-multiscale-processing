"""Tests for Part 3.2 resizing comparisons and energy selection."""
import unittest

import numpy as np

from filters import derivative_of_gaussian_edges
from part3_2 import compare_width_reduction, target_width
from seam_carving import compute_dog_energy, compute_energy


class SeamCarvingComparisonTests(unittest.TestCase):
    def test_target_width_rounds_seam_count_consistently(self):
        self.assertEqual(target_width(714), 500)
        self.assertEqual(target_width(11), 8)
        self.assertEqual(target_width(10), 7)

    def test_resize_crop_and_seam_results_have_equal_rgb_dimensions(self):
        random = np.random.default_rng(321)
        image = random.random((18, 21, 3))

        results = compare_width_reduction(image)

        self.assertEqual(results["target_width"], 15)
        for key in ("resized", "center_crop", "seam_carved"):
            self.assertEqual(results[key].shape, (18, 15, 3))
        self.assertEqual(results["original"].shape, image.shape)
        self.assertEqual(results["seams"].shape, (6, 18))

    def test_grayscale_comparisons_preserve_height_and_target_width(self):
        image = np.arange(9 * 17, dtype=np.float64).reshape(9, 17)

        results = compare_width_reduction(image)

        self.assertEqual(results["target_width"], 12)
        self.assertEqual(results["resized"].shape, (9, 12))
        self.assertEqual(results["center_crop"].shape, (9, 12))
        self.assertEqual(results["seam_carved"].shape, (9, 12))

    def test_both_energy_functions_return_valid_image_sized_maps(self):
        random = np.random.default_rng(322)
        image = random.random((15, 19, 3))

        finite_difference = compute_energy(image)
        dog = compute_dog_energy(image, sigma=1.4)
        expected_dog = derivative_of_gaussian_edges(
            np.dot(image, [0.299, 0.587, 0.114]), sigma=1.4
        )[5]

        self.assertEqual(finite_difference.shape, image.shape[:2])
        self.assertEqual(dog.shape, image.shape[:2])
        self.assertTrue(np.isfinite(finite_difference).all())
        self.assertTrue(np.isfinite(dog).all())
        np.testing.assert_allclose(dog, expected_dog, rtol=0, atol=0)

    def test_selectable_energy_function_drives_seam_carving(self):
        image = np.zeros((6, 9), dtype=np.float64)
        image[:, 3] = 1.0
        calls = []

        def prefer_left(image_to_score):
            calls.append(image_to_score.shape)
            energy = np.ones(image_to_score.shape[:2])
            energy[:, 0] = 0.0
            return energy

        results = compare_width_reduction(
            image, energy_function=prefer_left, reduction_fraction=0.1
        )

        self.assertGreater(len(calls), 0)
        self.assertTrue(np.all(results["seams"] == 0))

    def test_invalid_width_and_fraction_are_rejected(self):
        for width, fraction in ((1, 0.3), (8, 0), (8, 1.0), (8, np.nan)):
            with self.subTest(width=width, fraction=fraction):
                with self.assertRaises(ValueError):
                    target_width(width, fraction)
        with self.assertRaises(ValueError):
            compare_width_reduction(np.zeros((5, 1)))


if __name__ == "__main__":
    unittest.main()