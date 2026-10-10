"""Tests for Part 1.2 finite differences and edge detection."""
import unittest

import numpy as np

from filters import finite_difference_edges


class FiniteDifferenceTests(unittest.TestCase):
    def test_constant_image_has_no_edges_including_at_borders(self):
        image = np.full((5, 8), 0.37)
        dx, dy, magnitude, edges = finite_difference_edges(image, threshold=0.1)

        for result in (dx, dy, magnitude, edges):
            self.assertEqual(result.shape, image.shape)
        np.testing.assert_allclose(dx, np.zeros_like(image), atol=1e-14)
        np.testing.assert_allclose(dy, np.zeros_like(image), atol=1e-14)
        np.testing.assert_allclose(magnitude, np.zeros_like(image), atol=1e-14)
        self.assertFalse(np.any(edges))

    def test_horizontal_step_has_expected_derivatives(self):
        image = np.zeros((6, 8), dtype=np.float64)
        image[:, 4:] = 1.0
        dx, dy, magnitude, edges = finite_difference_edges(image, threshold=0.5)

        self.assertEqual(dx.shape, image.shape)
        self.assertEqual(dy.shape, image.shape)
        np.testing.assert_allclose(dx[:, 3:5], np.ones((6, 2)), atol=1e-14)
        np.testing.assert_allclose(dy, np.zeros_like(image), atol=1e-14)
        np.testing.assert_allclose(
            magnitude[:, 3:5], np.ones((6, 2)), atol=1e-14
        )
        self.assertTrue(np.all(edges[:, 3:5]))
        self.assertFalse(np.any(edges[:, :3]))
        self.assertFalse(np.any(edges[:, 5:]))

        _, _, _, higher_threshold_edges = finite_difference_edges(
            image, threshold=1.1
        )
        self.assertFalse(np.any(higher_threshold_edges))


if __name__ == "__main__":
    unittest.main()