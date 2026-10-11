"""Tests for Part 3.1 seam-carving operations."""
import itertools
import unittest

import numpy as np

from seam_carving import (
    carve_horizontal,
    carve_vertical,
    compute_energy,
    draw_seams,
    find_vertical_seam,
    remove_vertical_seam,
)


class SeamCarvingTests(unittest.TestCase):
    def test_energy_map_matches_image_spatial_dimensions(self):
        image = np.zeros((8, 11, 3), dtype=np.float64)
        image[:, 6:] = 1.0

        energy = compute_energy(image)

        self.assertEqual(energy.shape, image.shape[:2])
        self.assertTrue(np.isfinite(energy).all())
        self.assertGreater(float(energy[:, 5:7].max()), 0.0)

    def test_dynamic_program_finds_minimum_connected_seam(self):
        energy = np.array(
            [[8, 1, 7, 9], [7, 5, 2, 8], [6, 4, 1, 6], [9, 3, 5, 7]],
            dtype=np.float64,
        )

        seam = find_vertical_seam(energy)

        self.assertEqual(seam.shape, (energy.shape[0],))
        self.assertTrue(np.all(np.abs(np.diff(seam)) <= 1))
        costs = []
        for path in itertools.product(range(energy.shape[1]), repeat=energy.shape[0]):
            if all(abs(path[row] - path[row - 1]) <= 1 for row in range(1, len(path))):
                costs.append(sum(energy[row, column] for row, column in enumerate(path)))
        self.assertAlmostEqual(float(energy[np.arange(energy.shape[0]), seam].sum()), min(costs))

    def test_remove_vertical_seam_preserves_remaining_rgb_pixels(self):
        image = np.arange(3 * 5 * 3).reshape(3, 5, 3)
        seam = np.array([1, 2, 3])

        carved = remove_vertical_seam(image, seam)

        self.assertEqual(carved.shape, (3, 4, 3))
        for row, column in enumerate(seam):
            expected = np.delete(image[row], column, axis=0)
            np.testing.assert_array_equal(carved[row], expected)

    def test_grayscale_carving_and_original_column_tracking(self):
        image = np.zeros((5, 8), dtype=np.float64)
        image[:, 4] = 1.0

        carved, seams = carve_vertical(image, 2)

        self.assertEqual(carved.shape, (5, 6))
        self.assertEqual(seams.shape, (2, 5))
        self.assertTrue(np.all(np.abs(np.diff(seams, axis=1)) <= 1))
        self.assertTrue(np.all((seams >= 0) & (seams < image.shape[1])))

    def test_horizontal_carving_reuses_vertical_behavior(self):
        image = np.zeros((6, 9, 3), dtype=np.float64)

        carved, seams = carve_horizontal(image, 2)

        self.assertEqual(carved.shape, (4, 9, 3))
        self.assertEqual(seams.shape, (2, 9))
        self.assertTrue(np.all(np.abs(np.diff(seams, axis=1)) <= 1))

    def test_draw_seams_marks_original_coordinates_without_mutating_image(self):
        image = np.full((4, 6), 0.4)
        original = image.copy()
        seams = np.array([[1, 2, 2, 3]])

        overlay = draw_seams(image, seams)

        self.assertEqual(overlay.shape, (4, 6, 3))
        expected_red = np.tile([1.0, 0.0, 0.0], (4, 1))
        np.testing.assert_array_equal(overlay[np.arange(4), seams[0]], expected_red)
        np.testing.assert_array_equal(image, original)

    def test_zero_seams_returns_a_copy_and_invalid_counts_are_rejected(self):
        image = np.ones((4, 5, 3))
        carved, seams = carve_vertical(image, 0)

        self.assertEqual(seams.shape, (0, 4))
        np.testing.assert_array_equal(carved, image)
        self.assertIsNot(carved, image)
        with self.assertRaises(ValueError):
            carve_vertical(image, 5)
        with self.assertRaises(ValueError):
            carve_vertical(image, -1)

    def test_invalid_seams_and_images_are_rejected(self):
        with self.assertRaises(ValueError):
            find_vertical_seam(np.ones((3, 0)))
        with self.assertRaises(ValueError):
            remove_vertical_seam(np.ones((3, 1)), np.zeros(3, dtype=int))
        with self.assertRaises(ValueError):
            remove_vertical_seam(np.ones((3, 4)), np.array([0, 2, 0]))
        with self.assertRaises(ValueError):
            compute_energy(np.full((3, 4), np.nan))


if __name__ == "__main__":
    unittest.main()