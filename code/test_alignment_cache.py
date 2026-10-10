"""Tests for Part 2.1's reusable alignment-point cache."""
import tempfile
import unittest
from pathlib import Path

from main import (
    _get_cached_alignment,
    _load_alignment_cache,
    _save_cached_alignment,
)


class AlignmentCacheTests(unittest.TestCase):
    def test_saved_points_round_trip_for_same_image_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "alignment_points.json"
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            first_points = [(10.5, 20.25), (30.75, 40.0)]
            second_points = [(11.0, 21.0), (31.0, 41.0)]

            _save_cached_alignment(
                cache_path,
                {},
                "pair",
                first_path,
                second_path,
                first_points,
                second_points,
            )
            cache = _load_alignment_cache(cache_path)

            self.assertEqual(
                _get_cached_alignment(cache, "pair", first_path, second_path),
                ([[10.5, 20.25], [30.75, 40.0]], [[11.0, 21.0], [31.0, 41.0]]),
            )

    def test_cache_is_ignored_when_source_image_path_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "alignment_points.json"
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            _save_cached_alignment(
                cache_path,
                {},
                "pair",
                first_path,
                second_path,
                [(1, 2), (3, 4)],
                [(5, 6), (7, 8)],
            )
            cache = _load_alignment_cache(cache_path)

            self.assertIsNone(
                _get_cached_alignment(cache, "pair", first_path, Path(directory) / "other.png")
            )

    def test_force_reselect_bypasses_a_valid_cached_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "alignment_points.json"
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            _save_cached_alignment(
                cache_path,
                {},
                "pair",
                first_path,
                second_path,
                [(1, 2), (3, 4)],
                [(5, 6), (7, 8)],
            )
            cache = _load_alignment_cache(cache_path)

            self.assertIsNone(
                _get_cached_alignment(
                    cache,
                    "pair",
                    first_path,
                    second_path,
                    force_reselect=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
