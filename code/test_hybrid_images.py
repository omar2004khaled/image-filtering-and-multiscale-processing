"""Synthetic tests for Part 2.1 hybrid-image functions."""
import unittest

import numpy as np

from hybrid_images import (
    fourier_log_magnitude,
    gaussian_blur,
    make_hybrid,
)


class HybridImageTests(unittest.TestCase):
    def test_components_and_hybrid_have_matching_dimensions(self):
        random = np.random.default_rng(21)
        first = random.random((12, 15, 3))
        second = random.random((12, 15, 3))

        low_pass, high_pass, hybrid = make_hybrid(
            first, second, sigma_low=1.0, sigma_high=1.4
        )
        self.assertEqual(low_pass.shape, first.shape)
        self.assertEqual(high_pass.shape, first.shape)
        self.assertEqual(hybrid.shape, first.shape)

    def test_low_and_high_pass_components_follow_definition(self):
        random = np.random.default_rng(22)
        first = random.random((13, 11))
        second = random.random((13, 11))
        sigma_low, sigma_high = 0.8, 1.2

        low_pass, high_pass, hybrid = make_hybrid(
            first, second, sigma_low=sigma_low, sigma_high=sigma_high
        )
        np.testing.assert_allclose(
            low_pass, gaussian_blur(first, sigma_low), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            high_pass, second - gaussian_blur(second, sigma_high),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(hybrid, low_pass + high_pass, rtol=0, atol=0)

    def test_reflection_padding_preserves_constant_image_at_edges(self):
        image = np.full((8, 10), 0.65)
        blurred = gaussian_blur(image, sigma=1.1)
        np.testing.assert_allclose(blurred, image, rtol=0, atol=1e-14)

    def test_identical_images_produce_finite_correctly_shaped_hybrid(self):
        random = np.random.default_rng(23)
        image = random.random((9, 12, 3))
        low_pass, high_pass, hybrid = make_hybrid(
            image, image, sigma_low=1.0, sigma_high=1.0
        )

        self.assertEqual(hybrid.shape, image.shape)
        self.assertTrue(np.isfinite(low_pass).all())
        self.assertTrue(np.isfinite(high_pass).all())
        self.assertTrue(np.isfinite(hybrid).all())
        np.testing.assert_allclose(hybrid, image, atol=1e-14)

    def test_fourier_spectrum_is_finite_when_frequency_bins_are_zero(self):
        image = np.zeros((8, 10, 3))
        spectrum = fourier_log_magnitude(image)

        self.assertEqual(spectrum.shape, image.shape[:2])
        self.assertTrue(np.isfinite(spectrum).all())

    def test_mismatched_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            make_hybrid(np.zeros((8, 9)), np.zeros((9, 8)))


if __name__ == "__main__":
    unittest.main()