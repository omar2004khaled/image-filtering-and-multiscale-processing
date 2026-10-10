"""Command-line entry point for Computer Vision Assignment 1."""
import argparse
from pathlib import Path
import sys
import unittest

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from benchmark_filters import main as run_convolution_benchmark
from filters import finite_difference_edges
from utils import load_image, show


PART_SECTIONS = {
    "1": ("1.1", "1.2", "1.3"),
    "2": ("2.1", "2.2", "2.3", "2.4"),
    "3": ("3.1", "3.2"),
}
ALL_SECTIONS = PART_SECTIONS["1"] + PART_SECTIONS["2"] + PART_SECTIONS["3"]
SUPPORTED_SELECTIONS = tuple(PART_SECTIONS) + ALL_SECTIONS + ("all",)


def _signed_image_for_display(image):
    """Map signed derivative values to [0, 1] with zero at mid-gray."""
    maximum = float(abs(image).max())
    if maximum == 0:
        return np.full(image.shape, 0.5)
    return 0.5 + 0.5 * image / maximum


def run_part_1_2(threshold):
    """Test and display finite-difference edges on the provided cameraman image."""
    print("Running Part 1.2 correctness tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_edges.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = CODE_DIR.parent / "images" / "part1_filters" / "cameraman.png"
    image = load_image(image_path, gray=True)
    dx, dy, magnitude, edges = finite_difference_edges(image, threshold)
    magnitude_maximum = float(magnitude.max())
    magnitude_display = (
        magnitude / magnitude_maximum if magnitude_maximum > 0 else magnitude
    )

    print(f"Displaying cameraman edges with threshold {threshold:g}.")
    show(
        image,
        _signed_image_for_display(dx),
        _signed_image_for_display(dy),
        magnitude_display,
        edges.astype(float),
        titles=(
            "Original grayscale",
            "Ix (zero = mid-gray)",
            "Iy (zero = mid-gray)",
            "Gradient magnitude",
            f"Edges (threshold={threshold:g})",
        ),
        size=4,
    )
    return True


def run_section(section, threshold=0.2):
    """Run an implemented section, or report that it is not ready yet."""
    if section == "1.2":
        return run_part_1_2(threshold)
    if section != "1.1":
        print(f"Section {section} is not implemented yet.")
        return True

    print("Running Part 1.1 correctness tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_filters.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    print("\nRunning the Part 1.1 benchmark...")
    run_convolution_benchmark()
    return True


def run_selection(selection, threshold=0.2):
    """Run one section, every section in a part, or all known sections."""
    if selection == "all":
        sections = ALL_SECTIONS
    elif selection in PART_SECTIONS:
        sections = PART_SECTIONS[selection]
    else:
        sections = (selection,)

    success = True
    for section in sections:
        if not run_section(section, threshold):
            success = False
            break
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run sections of Computer Vision Assignment 1."
    )
    parser.add_argument(
        "--part",
        required=True,
        choices=SUPPORTED_SELECTIONS,
        help="section (for example 1.1), part (1, 2, or 3), or all",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Part 1.2 edge threshold (default: 0.2)",
    )
    arguments = parser.parse_args()
    return 0 if run_selection(arguments.part, arguments.threshold) else 1


if __name__ == "__main__":
    raise SystemExit(main())
