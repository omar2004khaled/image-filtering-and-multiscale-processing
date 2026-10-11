"""Tests for Part 2.4 multiresolution image blending."""
import unittest

import numpy as np

from blending import build_blended_laplacian_pyramid, multiresolution_blend
from pyramids import build_laplacian_pyramid, expand


class MultiresolutionBlendingTests(unittest.TestCase):
    def test_all_white_and_black_masks_select_the_expected_image(self):
        random = np.random.default_rng(241)
        first = random.random((19, 23, 3))
        second = random.random((19, 23, 3))

        choose_first = multiresolution_blend(
            first, second, np.ones((19, 23)), min_size=4
        )
        choose_second = multiresolution_blend(
            first, second, np.zeros((19, 23)), min_size=4
        )

        np.testing.assert_allclose(choose_first, first, rtol=0, atol=1e-14)
        np.testing.assert_allclose(choose_second, second, rtol=0, atol=1e-14)

    def test_each_level_uses_the_gaussian_mask_weighting_equation(self):
        random = np.random.default_rng(242)
        first = random.random((17, 21))
        second = random.random((17, 21))
        mask = np.zeros((17, 21))
        mask[:, :10] = 1.0

        blended, first_laplacian, second_laplacian, mask_gaussian = (
            build_blended_laplacian_pyramid(first, second, mask, min_size=4)
        )

        self.assertEqual(len(blended), len(mask_gaussian))
        np.testing.assert_array_equal(mask_gaussian[0], mask)
        for first_level, second_level, mask_level, blended_level in zip(
            first_laplacian, second_laplacian, mask_gaussian, blended
        ):
            np.testing.assert_allclose(
                blended_level,
                mask_level * first_level + (1.0 - mask_level) * second_level,
                rtol=0,
                atol=0,
            )
        self.assertTrue(np.any((mask_gaussian[1] > 0) & (mask_gaussian[1] < 1)))

    def test_odd_size_rgb_blend_reconstructs_at_original_dimensions(self):
        random = np.random.default_rng(243)
        first = random.random((19, 23, 3))
        second = random.random((19, 23, 3))
        mask = random.random((19, 23))

        blended = multiresolution_blend(first, second, mask, min_size=4)

        self.assertEqual(blended.shape, first.shape)
        self.assertTrue(np.isfinite(blended).all())

    def test_single_level_blend_obeys_mask_pixelwise(self):
        first = np.full((5, 7), 0.2)
        second = np.full((5, 7), 0.8)
        mask = np.linspace(0.0, 1.0, 35).reshape(5, 7)

        blended = multiresolution_blend(first, second, mask, min_size=8)

        np.testing.assert_allclose(blended, mask * first + (1 - mask) * second)

    def test_blending_rejects_mismatched_or_invalid_inputs(self):
        image = np.zeros((8, 9, 3))
        mask = np.ones((8, 9))

        with self.assertRaises(ValueError):
            multiresolution_blend(image, np.zeros((8, 10, 3)), mask)
        with self.assertRaises(ValueError):
            multiresolution_blend(image, image, np.ones((8, 8)))
        with self.assertRaises(ValueError):
            multiresolution_blend(image, image, np.full((8, 9), 1.1))
        with self.assertRaises(ValueError):
            multiresolution_blend(image, image, np.full((8, 9), np.nan))


if __name__ == "__main__":
    unittest.main()