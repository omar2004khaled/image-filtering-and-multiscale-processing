"""Tests for Part 2.2 factor-four downsampling and prefiltering."""
import unittest

import numpy as np

from aliasing import aliasing_comparison, downsample_by_factor
from hybrid_images import gaussian_blur


class AliasingTests(unittest.TestCase):
    def test_downsampling_keeps_every_fourth_sample_and_dimensions(self):
        image = np.arange(19 * 23, dtype=np.float64).reshape(19, 23)
        reduced = downsample_by_factor(image, factor=4)

        self.assertEqual(reduced.shape, (5, 6))
        np.testing.assert_array_equal(reduced, image[::4, ::4])

    def test_pipeline_returns_full_blur_and_quarter_size_images(self):
        random = np.random.default_rng(2026)
        image = random.random((19, 23))
        blurred, direct, prefiltered = aliasing_comparison(
            image, sigma=0.8, factor=4
        )

        self.assertEqual(blurred.shape, image.shape)
        self.assertEqual(direct.shape, (5, 6))
        self.assertEqual(prefiltered.shape, (5, 6))
        np.testing.assert_array_equal(direct, image[::4, ::4])
        np.testing.assert_allclose(
            prefiltered,
            gaussian_blur(image, 0.8)[::4, ::4],
            rtol=0,
            atol=0,
        )

    def test_reflected_prefilter_preserves_constant_image(self):
        image = np.full((16, 20), 0.42)
        blurred, direct, prefiltered = aliasing_comparison(
            image, sigma=1.0, factor=4
        )

        np.testing.assert_allclose(blurred, image, atol=1e-14)
        np.testing.assert_allclose(direct, 0.42, atol=1e-14)
        np.testing.assert_allclose(prefiltered, 0.42, atol=1e-14)

    def test_invalid_factor_is_rejected(self):
        with self.assertRaises(ValueError):
            downsample_by_factor(np.ones((8, 8)), factor=0)


if __name__ == "__main__":
    unittest.main()
