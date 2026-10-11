"""Tests for Part 2.3 Gaussian and Laplacian pyramids."""
import unittest

import numpy as np

from pyramids import (
    BINOMIAL_FILTER,
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    expand,
    reduce,
    reconstruct_from_laplacian,
)


class PyramidTests(unittest.TestCase):
    def test_reduce_uses_binomial_filter_and_ceil_half_dimensions(self):
        image = np.zeros((9, 11), dtype=np.float64)
        image[4, 4] = 1.0

        reduced = reduce(image)

        self.assertEqual(reduced.shape, (5, 6))
        self.assertAlmostEqual(reduced[2, 2], (6 / 16) ** 2)
        self.assertAlmostEqual(reduced[2, 1], (6 / 16) * (1 / 16))
        np.testing.assert_array_equal(BINOMIAL_FILTER, np.array([1, 4, 6, 4, 1]) / 16)

    def test_expand_uses_binomial_filter_and_factor_four(self):
        image = np.zeros((5, 5), dtype=np.float64)
        image[2, 2] = 1.0

        expanded = expand(image)

        self.assertEqual(expanded.shape, (10, 10))
        self.assertAlmostEqual(expanded[4, 4], 4 * (6 / 16) ** 2)
        self.assertAlmostEqual(expanded[4, 5], 4 * (6 / 16) * (4 / 16))

    def test_gaussian_levels_are_successive_reductions(self):
        random = np.random.default_rng(231)
        image = random.random((35, 41, 3))

        levels = build_gaussian_pyramid(image, min_size=8)

        self.assertEqual(levels[0].shape, image.shape)
        self.assertEqual([level.shape[:2] for level in levels], [
            (35, 41), (18, 21), (9, 11), (5, 6)
        ])
        for previous, current in zip(levels, levels[1:]):
            np.testing.assert_allclose(current, reduce(previous), rtol=0, atol=0)

    def test_laplacian_levels_follow_gaussian_prediction_residuals(self):
        random = np.random.default_rng(232)
        gaussian = build_gaussian_pyramid(random.random((19, 23)), min_size=4)

        laplacian = build_laplacian_pyramid(gaussian)

        self.assertEqual(len(laplacian), len(gaussian))
        for index in range(len(gaussian) - 1):
            prediction = expand(gaussian[index + 1])[:gaussian[index].shape[0], :gaussian[index].shape[1]]
            np.testing.assert_allclose(
                laplacian[index], gaussian[index] - prediction, rtol=0, atol=0
            )
        np.testing.assert_array_equal(laplacian[-1], gaussian[-1])

    def test_odd_dimensions_crop_correctly_and_reconstruct(self):
        random = np.random.default_rng(233)
        image = random.random((19, 23))
        gaussian = build_gaussian_pyramid(image, min_size=4)
        laplacian = build_laplacian_pyramid(gaussian)

        reconstructed = reconstruct_from_laplacian(laplacian)

        self.assertEqual(reconstructed.shape, image.shape)
        np.testing.assert_allclose(reconstructed, image, rtol=0, atol=1e-14)

    def test_reconstruction_error_is_near_machine_precision(self):
        random = np.random.default_rng(234)
        image = random.random((32, 27, 3))
        gaussian = build_gaussian_pyramid(image, min_size=3)
        laplacian = build_laplacian_pyramid(gaussian)

        reconstructed = reconstruct_from_laplacian(laplacian)

        self.assertLess(float(np.mean(np.abs(reconstructed - image))), 1e-14)
        self.assertLess(float(np.max(np.abs(reconstructed - image))), 1e-13)

    def test_constant_image_is_preserved_by_reduce_and_expand_blur(self):
        image = np.full((9, 13, 3), 0.42)

        np.testing.assert_allclose(reduce(image), 0.42, rtol=0, atol=1e-14)

    def test_single_pixel_dimensions_are_supported(self):
        image = np.array([[0.7]])

        reduced = reduce(image)
        expanded = expand(image)
        gaussian = build_gaussian_pyramid(image)
        reconstructed = reconstruct_from_laplacian(
            build_laplacian_pyramid(gaussian)
        )

        self.assertEqual(reduced.shape, (1, 1))
        self.assertEqual(expanded.shape, (2, 2))
        np.testing.assert_allclose(reconstructed, image, rtol=0, atol=1e-14)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            reduce(np.zeros((0, 5)))
        with self.assertRaises(ValueError):
            expand(np.zeros((3, 4, 0)))
        with self.assertRaises(ValueError):
            build_gaussian_pyramid(np.ones((5, 5)), min_size=0)
        with self.assertRaises(ValueError):
            build_laplacian_pyramid([])
        with self.assertRaises(ValueError):
            reconstruct_from_laplacian([])


if __name__ == "__main__":
    unittest.main()