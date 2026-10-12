"""Tests for Bonus object removal and seam insertion."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bonus import (
    find_insertion_locations,
    insert_vertical_seams,
    polygon_mask,
    remove_masked_object,
    restore_width_by_insertion,
    save_bonus_figure,
)


class SeamCarvingBonusTests(unittest.TestCase):
    def test_polygon_mask_matches_image_and_covers_polygon(self):
        mask = polygon_mask((8, 10), [(2, 2), (6, 2), (6, 6), (2, 6)])

        self.assertEqual(mask.shape, (8, 10))
        self.assertTrue(mask[4, 4])
        self.assertFalse(mask[0, 0])

    def test_masked_object_is_removed_and_k_matches_width_reduction(self):
        image = np.zeros((9, 12, 3), dtype=np.float64)
        image[2:7, 5:7] = 1.0
        mask = np.zeros((9, 12), dtype=bool)
        mask[2:7, 5:7] = True
        original = image.copy()
        original_mask = mask.copy()

        reduced, seam_count, remaining_mask = remove_masked_object(image, mask)

        self.assertEqual(seam_count, image.shape[1] - reduced.shape[1])
        self.assertGreater(seam_count, 0)
        self.assertFalse(np.any(remaining_mask))
        np.testing.assert_array_equal(image, original)
        np.testing.assert_array_equal(mask, original_mask)

    def test_border_seam_insertion_averages_with_pixel_itself(self):
        image = np.array([[2.0, 4.0, 8.0], [10.0, 20.0, 30.0]])
        locations = np.array([[0, 2]])

        restored, positions = insert_vertical_seams(image, locations)

        self.assertEqual(restored.shape, (2, 4))
        np.testing.assert_allclose(restored[0], [2.0, 3.0, 4.0, 8.0])
        np.testing.assert_allclose(restored[1], [10.0, 20.0, 25.0, 30.0])
        self.assertTrue(np.all((positions >= 0) & (positions < 4)))

    def test_copy_seam_locations_are_unique_and_width_restores(self):
        image = np.arange(5 * 9, dtype=np.float64).reshape(5, 9)

        locations = find_insertion_locations(image, 3)
        restored, mapped_locations, output_positions = restore_width_by_insertion(
            image, 3
        )

        self.assertEqual(locations.shape, (3, 5))
        self.assertEqual(restored.shape, (image.shape[0], image.shape[1] + 3))
        self.assertEqual(mapped_locations.shape, (3, 5))
        self.assertEqual(output_positions.shape, (3, 5, 2))
        for row in range(image.shape[0]):
            self.assertEqual(np.unique(locations[:, row]).size, 3)
            self.assertEqual(np.unique(output_positions[:, row, :]).size, 6)

    def test_insertion_positions_retain_seam_identity_across_rows(self):
        image = np.arange(2 * 5, dtype=np.float64).reshape(2, 5)
        locations = np.array([[1, 3], [3, 1]])

        _, positions = insert_vertical_seams(image, locations)

        np.testing.assert_array_equal(positions[0], [[1, 2], [4, 5]])
        np.testing.assert_array_equal(positions[1], [[4, 5], [1, 2]])

    def test_remove_and_restore_pipeline_preserves_input_and_restores_shape(self):
        image = np.zeros((12, 18, 3), dtype=np.float64)
        image[3:10, 8:11] = 0.9
        mask = np.zeros((12, 18), dtype=bool)
        mask[2:11, 7:12] = True
        original = image.copy()

        reduced, count, remaining = remove_masked_object(image, mask)
        restored, _, positions = restore_width_by_insertion(reduced, count)

        self.assertFalse(np.any(remaining))
        self.assertEqual(count, image.shape[1] - reduced.shape[1])
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(positions.shape, (count, image.shape[0], 2))
        np.testing.assert_array_equal(image, original)

    def test_three_panel_figure_is_saved(self):
        image = np.full((6, 8, 3), 0.4)
        mask = np.zeros((6, 8), dtype=bool)
        mask[2:4, 3:5] = True
        reduced = image[:, :-1]
        restored_overlay = np.zeros((6, 8, 3))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bonus.png"
            figure = save_bonus_figure(path, image, mask, reduced, restored_overlay)
            try:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            finally:
                import matplotlib.pyplot as plt
                plt.close(figure)

    def test_invalid_mask_and_duplicate_insertion_locations_are_rejected(self):
        image = np.zeros((4, 6))
        with self.assertRaises(ValueError):
            remove_masked_object(image, np.zeros((3, 6), dtype=bool))
        with self.assertRaises(ValueError):
            insert_vertical_seams(image, np.array([[2, 2, 2, 2], [2, 2, 2, 2]]))


if __name__ == "__main__":
    unittest.main()