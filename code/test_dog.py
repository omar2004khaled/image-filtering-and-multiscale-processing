"""Tests for Part 1.3 derivative-of-Gaussian filtering."""
import unittest

import numpy as np

from filters import (
    derivative_of_gaussian_edges,
    derivative_of_gaussian_filters,
    gaussian_kernel,
)


class DifferenceOfGaussianTests(unittest.TestCase):
    def test_gaussian_is_normalized_symmetric_and_odd_sized(self):
        gaussian = gaussian_kernel(sigma=1.2)

        self.assertEqual(gaussian.shape[0] % 2, 1)
        self.assertEqual(gaussian.shape[1] % 2, 1)
        self.assertAlmostEqual(float(gaussian.sum()), 1.0, places=14)
        np.testing.assert_allclose(gaussian, gaussian[::-1, :], atol=1e-15)
        np.testing.assert_allclose(gaussian, gaussian[:, ::-1], atol=1e-15)

    def test_dog_filters_have_expected_shapes_and_derivative_direction(self):
        gaussian, dog_x, dog_y = derivative_of_gaussian_filters(sigma=1.0)
        self.assertEqual(dog_x.shape, (gaussian.shape[0], gaussian.shape[1] + 2))
        self.assertEqual(dog_y.shape, (gaussian.shape[0] + 2, gaussian.shape[1]))
        self.assertEqual(dog_x.shape[0] % 2, 1)
        self.assertEqual(dog_x.shape[1] % 2, 1)
        self.assertEqual(dog_y.shape[0] % 2, 1)
        self.assertEqual(dog_y.shape[1] % 2, 1)
        self.assertAlmostEqual(float(dog_x.sum()), 0.0, places=14)
        self.assertAlmostEqual(float(dog_y.sum()), 0.0, places=14)

        horizontal_ramp = np.tile(np.arange(25, dtype=np.float64), (21, 1))
        _, _, _, ix, iy, _, _ = derivative_of_gaussian_edges(
            horizontal_ramp, sigma=1.0, threshold=0.1
        )
        self.assertGreater(ix[10, 12], 0)
        self.assertAlmostEqual(float(iy[10, 12]), 0.0, places=12)

    def test_zero_padding_produces_dog_frame_response_on_constant_image(self):
        image = np.full((12, 16), 0.37)
        results = derivative_of_gaussian_edges(
            image, sigma=1.1, threshold=1e-10
        )
        ix, iy, magnitude, edges = results[3:]

        for result in (ix, iy, magnitude, edges):
            self.assertEqual(result.shape, image.shape)
        np.testing.assert_allclose(ix[4:-4, 5:-5], 0.0, atol=1e-14)
        np.testing.assert_allclose(iy[5:-5, 4:-4], 0.0, atol=1e-14)
        self.assertGreater(float(magnitude[0, :].max()), 1e-10)
        self.assertTrue(np.any(edges[0, :]))


if __name__ == "__main__":
    unittest.main()