"""Tests for Part 1.2 finite differences and edge detection."""
import unittest

import numpy as np

from filters import finite_difference_edges


class FiniteDifferenceTests(unittest.TestCase):
    def test_zero_padding_produces_frame_edges_on_constant_image(self):
        image = np.full((5, 8), 0.37)
        dx, dy, magnitude, edges = finite_difference_edges(image, threshold=0.1)

        for result in (dx, dy, magnitude, edges):
            self.assertEqual(result.shape, image.shape)
        np.testing.assert_allclose(dx[:, 0], 0.37, atol=1e-14)
        np.testing.assert_allclose(dx[:, -1], -0.37, atol=1e-14)
        np.testing.assert_allclose(dx[:, 1:-1], 0.0, atol=1e-14)
        np.testing.assert_allclose(dy[0, :], 0.37, atol=1e-14)
        np.testing.assert_allclose(dy[-1, :], -0.37, atol=1e-14)
        np.testing.assert_allclose(dy[1:-1, :], 0.0, atol=1e-14)
        self.assertTrue(np.all(edges[0, :]))
        self.assertTrue(np.all(edges[-1, :]))
        self.assertTrue(np.all(edges[:, 0]))
        self.assertTrue(np.all(edges[:, -1]))
        self.assertFalse(np.any(edges[1:-1, 1:-1]))

    def test_horizontal_step_has_expected_derivatives(self):
        image = np.zeros((6, 8), dtype=np.float64)
        image[:, 4:] = 1.0
        dx, dy, magnitude, edges = finite_difference_edges(image, threshold=0.5)

        self.assertEqual(dx.shape, image.shape)
        self.assertEqual(dy.shape, image.shape)
        np.testing.assert_allclose(dx[:, 3:5], np.ones((6, 2)), atol=1e-14)
        np.testing.assert_allclose(dy[0, :], image[0, :], atol=1e-14)
        np.testing.assert_allclose(dy[-1, :], -image[-1, :], atol=1e-14)
        np.testing.assert_allclose(dy[1:-1, :], 0.0, atol=1e-14)
        np.testing.assert_allclose(
            magnitude[1:-1, 3:5], np.ones((4, 2)), atol=1e-14
        )
        np.testing.assert_allclose(
            magnitude[[0, -1], 3:5], [[1.0, np.sqrt(2)], [1.0, np.sqrt(2)]],
            atol=1e-14,
        )
        self.assertTrue(np.all(edges[:, 3:5]))
        self.assertFalse(np.any(edges[1:-1, :3]))
        self.assertFalse(np.any(edges[1:-1, 5:-1]))
        self.assertTrue(np.all(edges[1:-1, -1]))

        _, _, _, higher_threshold_edges = finite_difference_edges(
            image, threshold=1.1
        )
        self.assertFalse(np.any(higher_threshold_edges[1:-1, :]))


if __name__ == "__main__":
    unittest.main()