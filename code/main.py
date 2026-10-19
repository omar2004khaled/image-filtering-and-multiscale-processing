"""Command-line entry point for Computer Vision Assignment 1."""
import argparse
import json
from pathlib import Path
import sys
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from benchmark_filters import main as run_convolution_benchmark
from align_images import align_images, pick_points
from aliasing import DEFAULT_SIGMA_SWEEP, aliasing_comparison, downsample_by_factor, sigma_sweep
from bonus import (
    draw_inserted_seams,
    polygon_mask,
    remove_masked_object,
    restore_width_by_insertion,
    save_bonus_outputs,
    select_polygon_mask,
)
from blending import build_blended_laplacian_pyramid
from filters import derivative_of_gaussian_edges, finite_difference_edges
from hybrid_images import fourier_log_magnitude, make_hybrid
from part3_2 import compare_width_reduction, target_width
from pyramids import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian,
)
from seam_carving import (
    carve_vertical,
    compute_dog_energy,
    compute_energy,
    draw_seams,
)
from utils import load_image, save_image, show


PART_SECTIONS = {
    "1": ("1.1", "1.2", "1.3"),
    "2": ("2.1", "2.2", "2.3", "2.4"),
    "3": ("3.1", "3.2"),
}
ALL_SECTIONS = PART_SECTIONS["1"] + PART_SECTIONS["2"] + PART_SECTIONS["3"]
SUPPORTED_SELECTIONS = tuple(PART_SECTIONS) + ALL_SECTIONS + ("all",)
SUPPORTED_SELECTIONS += ("bonus",)


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

    image_path = CODE_DIR.parent / "images" / "part1_filters" / "facade.png"
    image = load_image(image_path, gray=True)
    dx, dy, magnitude, edges = finite_difference_edges(image, threshold)
    magnitude_maximum = float(magnitude.max())
    magnitude_display = (
        magnitude / magnitude_maximum if magnitude_maximum > 0 else magnitude
    )

    print(f"Displaying facade edges with threshold {threshold:g}.")
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


def run_part_1_3(threshold, sigma, finite_difference_threshold=0.2):
    """Test and display derivative-of-Gaussian edges on facade."""
    print("Running Part 1.3 correctness tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_dog.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = CODE_DIR.parent / "images" / "part1_filters" / "facade.png"
    image = load_image(image_path, gray=True)
    gaussian, dog_x, dog_y, ix, iy, magnitude, edges = (
        derivative_of_gaussian_edges(image, sigma=sigma, threshold=threshold)
    )
    _, _, _, finite_difference_edge_map = finite_difference_edges(
        image, threshold=finite_difference_threshold
    )

    magnitude_maximum = float(magnitude.max())
    magnitude_display = magnitude / magnitude_maximum if magnitude_maximum > 0 else magnitude
    gaussian_display = gaussian / float(gaussian.max())
    print(f"DoG sigma: {sigma:g}; edge threshold: {threshold:g}.")
    print(
        "DoG combines smoothing and differentiation, generally reducing "
        "sensitivity to small-scale noise."
    )
    print(
        f"Edge pixels: Part 1.2 (threshold={finite_difference_threshold:g}) = "
        f"{np.count_nonzero(finite_difference_edge_map)}, Part 1.3 = "
        f"{np.count_nonzero(edges)}."
    )
    show(
        image,
        gaussian_display,
        _signed_image_for_display(dog_x),
        _signed_image_for_display(dog_y),
        _signed_image_for_display(ix),
        _signed_image_for_display(iy),
        magnitude_display,
        edges.astype(float),
        titles=(
            "Original grayscale",
            f"Gaussian (sigma={sigma:g})",
            "DoG x (zero = mid-gray)",
            "DoG y (zero = mid-gray)",
            "Ix response (zero = mid-gray)",
            "Iy response (zero = mid-gray)",
            "Gradient magnitude",
            f"Edges (threshold={threshold:g})",
        ),
        size=3,
    )
    return True


def _limit_image_size(image, maximum_side=1000):
    """Resize oversized inputs while preserving their aspect ratio."""
    height, width = image.shape[:2]
    scale = min(1.0, maximum_side / max(height, width))
    if scale == 1.0:
        return image
    new_size = (round(width * scale), round(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _load_hybrid_image(path):
    """Load RGB images, handling non-ASCII Windows paths unsupported by imread."""
    path = Path(path)
    if str(path).isascii():
        return load_image(path)

    encoded_image = np.fromfile(path, dtype=np.uint8)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    if decoded_image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _scale_points(points, original_shape, resized_shape):
    """Map (x, y) landmark points to an image resized with OpenCV."""
    scale_x = resized_shape[1] / original_shape[1]
    scale_y = resized_shape[0] / original_shape[0]
    return [
        ((x + 0.5) * scale_x - 0.5, (y + 0.5) * scale_y - 0.5)
        for x, y in points
    ]


def _parse_alignment_points(records):
    """Convert CLI records into original-image point pairs by pair name."""
    parsed = {}
    for record in records or []:
        pair_name = record[0]
        coordinates = [float(value) for value in record[1:]]
        parsed[pair_name] = (
            [(coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])],
            [(coordinates[4], coordinates[5]), (coordinates[6], coordinates[7])],
        )
    return parsed


def _load_alignment_cache(cache_path):
    """Read saved original-image landmarks, or return an empty cache."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return {}
    with cache_path.open("r", encoding="utf-8") as cache_file:
        cache = json.load(cache_file)
    if not isinstance(cache, dict):
        raise ValueError(f"Alignment cache must contain a JSON object: {cache_path}")
    return cache


def _get_cached_alignment(
    cache, pair_name, first_path, second_path, force_reselect=False
):
    """Return cached landmarks only when both source paths still match."""
    if force_reselect:
        return None
    entry = cache.get(pair_name)
    if not isinstance(entry, dict):
        return None
    if entry.get("first_image") != str(Path(first_path).resolve()):
        return None
    if entry.get("second_image") != str(Path(second_path).resolve()):
        return None
    first_points = entry.get("points_first")
    second_points = entry.get("points_second")
    if not isinstance(first_points, list) or not isinstance(second_points, list):
        return None
    if len(first_points) != 2 or len(second_points) != 2:
        return None
    return first_points, second_points


def _save_cached_alignment(cache_path, cache, pair_name, first_path, second_path,
                           points_first, points_second):
    """Store landmark coordinates in original-image pixels for later runs."""
    cache[pair_name] = {
        "first_image": str(Path(first_path).resolve()),
        "second_image": str(Path(second_path).resolve()),
        "points_first": [[float(x), float(y)] for x, y in points_first],
        "points_second": [[float(x), float(y)] for x, y in points_second],
    }
    Path(cache_path).write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _signed_for_display(image):
    """Map signed high-pass values to [0, 1] with zero at mid-gray."""
    maximum = float(np.max(np.abs(image)))
    if maximum == 0:
        return np.full(image.shape, 0.5)
    return 0.5 + 0.5 * image / maximum


def _display_alignment(first_image, aligned_second, pair_name):
    overlay = 0.5 * first_image + 0.5 * aligned_second
    print(f"Alignment preview for {pair_name}: compare the pair and overlay.")
    show(
        first_image,
        aligned_second,
        overlay,
        titles=("First image", "Aligned second image", "50/50 overlay"),
        size=4,
    )


def _save_favorite_comparison(output_directory, images, names):
    favorite_directory = output_directory / "favorite"
    favorite_directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 5, figsize=(18, 7))

    for index, (image, name) in enumerate(zip(images, names), start=1):
        image_path = favorite_directory / f"image_{index:02d}_{name}.png"
        if index == 4:
            image_to_save = _signed_for_display(image)
            image_to_show = image_to_save
        else:
            image_to_save = np.clip(image, 0, 1)
            image_to_show = image_to_save
        save_image(image_path, image_to_save)

        spectrum = fourier_log_magnitude(image)
        spectrum_path = favorite_directory / f"spectrum_{index:02d}_{name}.png"
        plt.imsave(spectrum_path, spectrum, cmap="magma")

        axes[0, index - 1].imshow(image_to_show, cmap="gray" if image_to_show.ndim == 2 else None)
        axes[0, index - 1].set_title(name.replace("_", " "))
        axes[1, index - 1].imshow(spectrum, cmap="magma")
        axes[1, index - 1].set_title(f"Log spectrum: {name.replace('_', ' ')}")
        axes[0, index - 1].axis("off")
        axes[1, index - 1].axis("off")

    figure.tight_layout()
    figure.savefig(favorite_directory / "image_and_spectrum_comparison.png", dpi=150)
    plt.show()


def run_part_2_1(options):
    """Create three aligned hybrids and save the favorite pair's spectra."""
    print("Running Part 2.1 hybrid-image tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_hybrid_images.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    images_directory = CODE_DIR.parent / "images" / "part2_frequencies"
    pairs = [
        (
            "man-cat",
            images_directory / "hybrid_man.jpg",
            images_directory / "hybrid_cat.jpg",
            "Required starter pair; visual review shows facial features overlap and the hybrid is interpretable.",
            ([(299, 341), (443, 332)], [(605, 290), (760, 372)]),
        ),
        (
            "cat-tiger",
            CODE_DIR.parent / "images" / "part1_filters" / "cat.jpg",
            CODE_DIR.parent / "images" / "part1_filters" / "Tiger.jpg",
            "User-provided cat/tiger pair; align the eyes and nose, then review the result.",
            None,
        ),
        (
            "dodge-motorcycle",
            CODE_DIR.parent / "images" / "part1_filters" / "Dodge Challenger.jpg",
            CODE_DIR.parent / "images" / "part1_filters" / "2017 Kawasaki Ninja ZX-10RR Earns that Extra “R”.jpg",
            "User-provided front-view vehicle pair; align corresponding front features and inspect the hybrid.",
            None,
        ),
    ]
    custom_pair_paths = options.custom_pair or []
    if custom_pair_paths and len(custom_pair_paths) != 2:
        print("Use --custom-pair twice to replace both custom pairs.")
        return False
    if custom_pair_paths:
        pairs = pairs[:1]
        for index, paths in enumerate(custom_pair_paths, start=1):
            first_path = Path(paths[0]).expanduser().resolve()
            second_path = Path(paths[1]).expanduser().resolve()
            if not first_path.is_file() or not second_path.is_file():
                print(
                    f"Custom pair {index} image path does not exist: "
                    f"{first_path if not first_path.is_file() else second_path}"
                )
                return False
            pairs.append(
                (
                    f"custom-{index}",
                    first_path,
                    second_path,
                    "User-provided replacement pair; review structural alignment and hybrid quality.",
                    None,
                )
            )

    favorite_name = options.favorite

    output_directory = Path(options.output_dir).expanduser()
    if not output_directory.is_absolute():
        output_directory = CODE_DIR.parent / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    report_entries = []
    favorite_images = None
    favorite_pair_name = None
    summary_path = output_directory / "summary.txt"
    summary_path.write_text(
        "Part 2.1 hybrid-image run\n"
        f"sigma low-pass: {options.sigma_low:g}\n"
        f"sigma high-pass: {options.sigma_high:g}\n\n",
        encoding="utf-8",
    )
    alignment_cache_path = output_directory / "alignment_points.json"
    alignment_cache = _load_alignment_cache(alignment_cache_path)
    supplied_points = _parse_alignment_points(options.alignment_points)
    reselect_pairs = set(options.reselect_points or [])
    for pair_name, first_path, second_path, note, saved_points in pairs:
        print(f"\nPair: {pair_name}\n  first: {first_path}\n  second: {second_path}")
        first_original = _load_hybrid_image(first_path)
        second_original = _load_hybrid_image(second_path)
        first_image = _limit_image_size(first_original)
        second_image = _limit_image_size(second_original)

        if saved_points is not None and pair_name not in reselect_pairs:
            original_points_first, original_points_second = saved_points
            alignment_source = "README-provided points"
            print("Using matching points documented in the starter README.")
        elif pair_name in supplied_points:
            original_points_first, original_points_second = supplied_points[pair_name]
            alignment_source = "command-line point coordinates"
            print("Using landmark coordinates supplied on the command line.")
        else:
            cached_points = _get_cached_alignment(
                alignment_cache,
                pair_name,
                first_path,
                second_path,
                force_reselect=pair_name in reselect_pairs,
            )
            if cached_points is not None:
                original_points_first, original_points_second = cached_points
                alignment_source = "cached point selection"
                print("Reusing saved landmark points.")
            else:
                print(
                    "Select two corresponding landmarks in the first image, then the "
                    "same two landmarks in the second image, in the same order."
                )
                selected_points_first = pick_points(first_image, n=2)
                selected_points_second = pick_points(second_image, n=2)
                if len(selected_points_first) != 2 or len(selected_points_second) != 2:
                    print(
                        "Alignment needs two points per image. If the picker does not "
                        "accept clicks, pass original-image coordinates with "
                        "--alignment-points PAIR X1 Y1 X2 Y2 U1 V1 U2 V2."
                    )
                with summary_path.open("a", encoding="utf-8") as summary_file:
                    summary_file.write(
                        f"{pair_name}:\n"
                        f"  first image: {first_path}\n"
                        f"  second image: {second_path}\n"
                        f"  sigma low-pass: {options.sigma_low:g}\n"
                        f"  sigma high-pass: {options.sigma_high:g}\n"
                        "  alignment status: incomplete; no points were selected.\n\n"
                    )
                    return False
                original_points_first = _scale_points(
                    selected_points_first, first_image.shape, first_original.shape
                )
                original_points_second = _scale_points(
                    selected_points_second, second_image.shape, second_original.shape
                )
                alignment_source = "interactive point selection"

        points_first = _scale_points(
            original_points_first, first_original.shape, first_image.shape
        )
        points_second = _scale_points(
            original_points_second, second_original.shape, second_image.shape
        )
        _save_cached_alignment(
            alignment_cache_path,
            alignment_cache,
            pair_name,
            first_path,
            second_path,
            original_points_first,
            original_points_second,
        )

        aligned_second = align_images(
            first_image, second_image, points_first, points_second
        )
        if aligned_second.shape != first_image.shape:
            print("Aligned image shape does not match the first image.")
            return False
        _display_alignment(first_image, aligned_second, pair_name)

        low_pass, high_pass, hybrid = make_hybrid(
            first_image,
            aligned_second,
            sigma_low=options.sigma_low,
            sigma_high=options.sigma_high,
        )

        pair_directory = output_directory / pair_name
        pair_directory.mkdir(parents=True, exist_ok=True)
        save_image(pair_directory / "first_image.png", first_image)
        save_image(pair_directory / "aligned_second_image.png", aligned_second)
        save_image(pair_directory / "low_pass_first.png", low_pass)
        save_image(pair_directory / "high_pass_second.png", _signed_for_display(high_pass))
        save_image(pair_directory / "hybrid.png", np.clip(hybrid, 0, 1))
        np.save(pair_directory / "low_pass_first.npy", low_pass)
        np.save(pair_directory / "high_pass_second.npy", high_pass)
        np.save(pair_directory / "hybrid_float.npy", hybrid)

        report_entries.append(
            f"{pair_name}:\n"
            f"  first image: {first_path}\n"
            f"  second image: {second_path}\n"
            f"  sigma low-pass: {options.sigma_low:g}\n"
            f"  sigma high-pass: {options.sigma_high:g}\n"
            f"  alignment points first: {points_first}\n"
            f"  alignment points second: {points_second}\n"
            f"  alignment status: succeeded using {alignment_source}; preview displayed.\n"
            f"  note: {note}\n"
        )
        with summary_path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(report_entries[-1] + "\n")
        if pair_name == favorite_name:
            favorite_pair_name = pair_name
            favorite_images = (
                first_image,
                aligned_second,
                low_pass,
                high_pass,
                hybrid,
            )
            labels = {
                "man-cat": ("man", "aligned_cat", "low_pass_man", "high_pass_cat", "hybrid"),
                "cat-tiger": ("cat", "aligned_tiger", "low_pass_cat", "high_pass_tiger", "cat_tiger_hybrid"),
                "dodge-motorcycle": ("dodge", "aligned_motorcycle", "low_pass_dodge", "high_pass_motorcycle", "dodge_motorcycle_hybrid"),
                "custom-1": ("custom_1_first", "custom_1_aligned_second", "custom_1_low_pass", "custom_1_high_pass", "custom_1_hybrid"),
                "custom-2": ("custom_2_first", "custom_2_aligned_second", "custom_2_low_pass", "custom_2_high_pass", "custom_2_hybrid"),
            }
            _save_favorite_comparison(
                output_directory,
                favorite_images,
                labels[favorite_pair_name],
            )

    if favorite_images is None:
        print(f"Favorite pair {options.favorite!r} was not produced.")
        return False

    print(f"Favorite pair: {favorite_pair_name}")
    print(f"Saved Part 2.1 results under: {output_directory}")
    return True


def run_part_2_2(sigma):
    """Compare direct factor-four sampling with Gaussian-prefiltered sampling."""
    print("Running Part 2.2 correctness tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_aliasing.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = CODE_DIR.parent / "images" / "part2_frequencies" / "rings.png"
    image = load_image(image_path, gray=True)
    factor = 4
    blurred, direct_downsampled, prefiltered_downsampled = aliasing_comparison(
        image, sigma=sigma, factor=factor
    )
    output_directory = CODE_DIR.parent / "outputs" / "part2_2"
    output_directory.mkdir(parents=True, exist_ok=True)

    outputs = (
        ("original.png", image),
        ("direct_downsampled.png", direct_downsampled),
        ("gaussian_blurred.png", blurred),
        ("prefiltered_downsampled.png", prefiltered_downsampled),
    )
    for filename, output_image in outputs:
        save_image(output_directory / filename, output_image)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    comparison_images = (image, direct_downsampled, blurred, prefiltered_downsampled)
    comparison_titles = (
        f"Original ({image.shape[1]}x{image.shape[0]})",
        f"Direct every {factor}th pixel ({direct_downsampled.shape[1]}x{direct_downsampled.shape[0]})",
        f"Gaussian blur (sigma={sigma:g})",
        f"Blur then downsample ({prefiltered_downsampled.shape[1]}x{prefiltered_downsampled.shape[0]})",
    )
    for axis, output_image, title in zip(axes, comparison_images, comparison_titles):
        axis.imshow(output_image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_directory / "comparison.png", dpi=150)
    plt.show()

    sweep_results = sigma_sweep(image, DEFAULT_SIGMA_SWEEP, factor=factor)
    columns = 4
    rows = (len(sweep_results) + columns - 1) // columns
    sweep_figure, sweep_axes = plt.subplots(rows, columns, figsize=(14, 3.5 * rows))
    for axis, (candidate_sigma, reduced_image) in zip(
        np.ravel(sweep_axes), sweep_results
    ):
        axis.imshow(
            reduced_image,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set_title(f"sigma={candidate_sigma:g}")
        axis.axis("off")
    for axis in np.ravel(sweep_axes)[len(sweep_results):]:
        axis.axis("off")
    sweep_figure.suptitle("Rings: Gaussian prefilter sigma sweep before /4 sampling")
    sweep_figure.tight_layout()
    sweep_figure.savefig(output_directory / "sigma_sweep.png", dpi=150)
    plt.show()

    summary = (
        "Part 2.2: Aliasing and Gaussian pre-filtering\n"
        f"Input: {image_path}\n"
        f"Input dimensions: {image.shape[1]} x {image.shape[0]}\n"
        f"Downsampling factor: {factor} in each dimension\n"
        f"Output dimensions: {direct_downsampled.shape[1]} x {direct_downsampled.shape[0]}\n"
        f"Gaussian sigma used for saved comparison: {sigma:g}\n"
        "Visual sweep candidates: "
        + ", ".join(f"{candidate:g}" for candidate in DEFAULT_SIGMA_SWEEP)
        + "\n"
        "Visual result: sigma 2.0 was the smallest tested value without visible "
        "false circles in the high-frequency corners at the inspected display scale; "
        "sigma 1.95 still showed faint residual structure. This is a subjective, "
        "image- and viewing-scale-dependent criterion, not a universal cutoff.\n"
        "Direct downsampling keeps one sample out of every 4x4 block without first "
        "removing frequencies above the new Nyquist limit, so high-frequency rings "
        "fold into false lower-frequency patterns (aliasing). Gaussian pre-filtering "
        "attenuates those frequencies before sampling.\n"
        "Gaussian pyramids use the same principle: blur first, then reduce the image "
        "size at each level.\n"
    )
    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved Part 2.2 images and summary under: {output_directory}")
    return True


def _pyramid_figure(levels, output_path, prefix, signed=False):
    """Save a labeled figure and level images for one image pyramid."""
    figure, axes = plt.subplots(1, len(levels), figsize=(3.2 * len(levels), 3.6))
    axes = np.atleast_1d(axes)
    for index, (axis, level) in enumerate(zip(axes, levels)):
        height, width = level.shape[:2]
        if signed and index < len(levels) - 1:
            maximum = float(np.max(np.abs(level)))
            displayed = (
                np.full(level.shape, 0.5)
                if maximum == 0
                else np.clip(0.5 + 0.5 * level / maximum, 0, 1)
            )
        else:
            displayed = np.clip(level, 0, 1)
        axis.imshow(
            displayed,
            cmap="gray" if level.ndim == 2 else None,
            vmin=0 if level.ndim == 2 else None,
            vmax=1 if level.ndim == 2 else None,
            interpolation="nearest",
        )
        axis.set_title(f"{prefix}{index} ({width}x{height})")
        axis.axis("off")
        level_directory = output_path.parent / (
            "laplacian_pyramid" if signed else "gaussian_pyramid"
        )
        level_directory.mkdir(parents=True, exist_ok=True)
        save_image(level_directory / f"level_{index:02d}.png", displayed)
        np.save(level_directory / f"level_{index:02d}.npy", level)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    return figure


def run_part_2_3():
    """Build and reconstruct Gaussian and Laplacian pyramids for facade.png."""
    print("Running Part 2.3 pyramid tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_pyramids.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = CODE_DIR.parent / "images" / "part1_filters" / "facade.png"
    image = load_image(image_path)
    gaussian = build_gaussian_pyramid(image)
    laplacian = build_laplacian_pyramid(gaussian)
    reconstructed = reconstruct_from_laplacian(laplacian)
    absolute_error = np.abs(reconstructed - image)
    mean_absolute_error = float(np.mean(absolute_error))
    maximum_absolute_error = float(np.max(absolute_error))

    output_directory = CODE_DIR.parent / "outputs" / "part2_3"
    output_directory.mkdir(parents=True, exist_ok=True)
    save_image(output_directory / "original.png", image)
    save_image(output_directory / "reconstructed.png", reconstructed)
    error_display_scale = maximum_absolute_error or 1.0
    error_display = np.clip(absolute_error / error_display_scale, 0, 1)
    save_image(output_directory / "absolute_error.png", error_display)
    np.save(output_directory / "original.npy", image)
    np.save(output_directory / "reconstructed.npy", reconstructed)
    np.save(output_directory / "absolute_error.npy", absolute_error)

    gaussian_figure = _pyramid_figure(
        gaussian, output_directory / "gaussian_pyramid.png", "G"
    )
    laplacian_figure = _pyramid_figure(
        laplacian,
        output_directory / "laplacian_pyramid.png",
        "L",
        signed=True,
    )
    comparison_figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(np.clip(reconstructed, 0, 1))
    axes[1].set_title("Reconstructed")
    axes[2].imshow(error_display)
    axes[2].set_title("Absolute error (contrast boosted)")
    for axis in axes:
        axis.axis("off")
    comparison_figure.tight_layout()
    comparison_figure.savefig(
        output_directory / "reconstruction_comparison.png", dpi=150
    )
    plt.show()
    plt.close(gaussian_figure)
    plt.close(laplacian_figure)
    plt.close(comparison_figure)

    spatial_pixels = [level.shape[0] * level.shape[1] for level in gaussian]
    total_pixel_count = sum(spatial_pixels)
    pixel_ratio = total_pixel_count / spatial_pixels[0]
    level_shapes = ", ".join(
        f"{level.shape[1]}x{level.shape[0]}" for level in gaussian
    )
    summary = (
        "Part 2.3: Gaussian and Laplacian Pyramids\n"
        f"Input: {image_path}\n"
        f"Input dimensions: {image.shape[1]} x {image.shape[0]}\n"
        f"Gaussian level dimensions (largest to smallest): {level_shapes}\n"
        f"Gaussian pyramid spatial pixel count: {total_pixel_count}\n"
        f"Gaussian pyramid pixel ratio: {pixel_ratio:.6f} x the original\n"
        f"Reconstruction mean absolute error: {mean_absolute_error:.12g}\n"
        f"Reconstruction maximum absolute error: {maximum_absolute_error:.12g}\n"
        f"Absolute-error PNG is contrast-boosted by {error_display_scale:.12g} for visibility only; "
        "raw errors are in absolute_error.npy.\n\n"
        "Theory\n"
        "1. reduce() filters horizontally and vertically with the normalized "
        "[1, 4, 6, 4, 1] kernel, then keeps every other row and column. Blurring "
        "before sampling attenuates frequencies that would otherwise alias into "
        "false coarse-scale patterns.\n"
        "2. expand() inserts each sample at even row and column positions in a "
        "zero-filled image, applies the same separable filter, and multiplies by "
        "4 to compensate for the quarter-density 2D zero insertion.\n"
        "3. The Gaussian pyramid starts with the original image and repeatedly "
        "applies reduce(), storing levels from largest to smallest.\n"
        "4. Each non-final Laplacian level is its Gaussian level minus the next "
        "smaller Gaussian level expanded and cropped to the current size. It stores "
        "the detail not predicted by the coarser level; the final level is the "
        "smallest Gaussian image.\n"
        "5. Reconstruction starts at the smallest Gaussian level and repeatedly "
        "expands then adds the corresponding Laplacian residual. Since each residual "
        "was defined as current level minus that exact expanded prediction, the sum "
        "recovers the current level (up to floating-point roundoff). Odd-sized "
        "levels are matched by dropping only the expanded image's last row and/or column.\n"
        "6. Lecture 3's residual subtracts a level's own blurred copy, rather than "
        "the expanded next Gaussian level. That residual is not the difference from "
        "the reconstruction predictor, so expanding the coarser level and adding it "
        "back does not generally recover the source exactly.\n"
        "7. With dimensions halved in both directions, successive Gaussian levels "
        "contain approximately 1, 1/4, 1/16, ... as many spatial pixels as the "
        "original. The geometric series approaches 1/(1 - 1/4) = 4/3 for many "
        "levels. A finite pyramid stops before that limit, and odd dimensions use "
        "ceil-halving, so their actual ratio differs slightly from the ideal series.\n"
    )
    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved Part 2.3 images, arrays, and summary under: {output_directory}")
    return True


def _ellipse_mask(shape, center, radii):
    """Return a hard-edged mask that selects the second image inside an ellipse."""
    height, width = shape
    center_x, center_y = center
    radius_x, radius_y = radii
    coordinates_y, coordinates_x = np.ogrid[:height, :width]
    inside = (
        ((coordinates_x - center_x) / radius_x) ** 2
        + ((coordinates_y - center_y) / radius_y) ** 2
        <= 1.0
    )
    mask = np.ones((height, width), dtype=np.float64)
    mask[inside] = 0.0
    return mask


def _ellipse_union_mask(shape, ellipses):
    """Return a mask whose black region is the union of the supplied ellipses."""
    mask = np.ones(shape, dtype=np.float64)
    for center, axes, angle in ellipses:
        cv2.ellipse(mask, center, axes, angle, 0, 360, 0.0, thickness=-1)
    return mask


def _save_blend_pyramid(levels, output_directory, folder_name, figure_name, prefix,
                        signed=False, reconstruction=None):
    """Save pyramid arrays and a labeled display figure for a blend example."""
    level_directory = output_directory / folder_name
    level_directory.mkdir(parents=True, exist_ok=True)
    panel_count = len(levels) + int(reconstruction is not None)
    figure, axes = plt.subplots(1, panel_count, figsize=(3.0 * panel_count, 3.5))
    axes = np.atleast_1d(axes)
    for index, (axis, level) in enumerate(zip(axes, levels)):
        if signed and index < len(levels) - 1:
            maximum = float(np.max(np.abs(level)))
            displayed = (
                np.full(level.shape, 0.5)
                if maximum == 0
                else np.clip(0.5 + 0.5 * level / maximum, 0, 1)
            )
        else:
            displayed = np.clip(level, 0, 1)
        axis.imshow(
            displayed,
            cmap="gray" if level.ndim == 2 else None,
            vmin=0 if level.ndim == 2 else None,
            vmax=1 if level.ndim == 2 else None,
            interpolation="nearest",
        )
        axis.set_title(f"{prefix}{index} ({level.shape[1]}x{level.shape[0]})")
        axis.axis("off")
        np.save(level_directory / f"level_{index:02d}.npy", level)
        save_image(level_directory / f"level_{index:02d}.png", displayed)
    if reconstruction is not None:
        axes[-1].imshow(np.clip(reconstruction, 0, 1))
        axes[-1].set_title("Sum / reconstruction")
        axes[-1].axis("off")
    figure.tight_layout()
    figure.savefig(output_directory / figure_name, dpi=150)
    return figure


def _run_blend_example(name, first, second, mask, description, min_size=16):
    """Build, save and summarize one multiresolution blend example."""
    if first.shape[:2] != mask.shape or second.shape != first.shape:
        raise ValueError("blend images and mask must have matching dimensions")
    blended_levels, first_laplacian, second_laplacian, mask_gaussian = (
        build_blended_laplacian_pyramid(first, second, mask, min_size=min_size)
    )
    blended = reconstruct_from_laplacian(blended_levels)
    output_directory = CODE_DIR.parent / "outputs" / "part2_4" / name
    output_directory.mkdir(parents=True, exist_ok=True)

    save_image(output_directory / "first_image.png", first)
    save_image(output_directory / "second_image.png", second)
    save_image(output_directory / "mask.png", mask)
    save_image(output_directory / "blended.png", blended)
    hard_cut = mask[..., np.newaxis] * first + (
        1.0 - mask[..., np.newaxis]
    ) * second
    save_image(output_directory / "hard_cut.png", hard_cut)
    np.save(output_directory / "first_image.npy", first)
    np.save(output_directory / "second_image.npy", second)
    np.save(output_directory / "mask.npy", mask)
    np.save(output_directory / "blended.npy", blended)
    np.save(output_directory / "hard_cut.npy", hard_cut)

    for label, levels in (
        ("first_laplacian_pyramid", first_laplacian),
        ("second_laplacian_pyramid", second_laplacian),
    ):
        level_directory = output_directory / label
        level_directory.mkdir(parents=True, exist_ok=True)
        for index, level in enumerate(levels):
            np.save(level_directory / f"level_{index:02d}.npy", level)
    mask_figure = _save_blend_pyramid(
        mask_gaussian,
        output_directory,
        "mask_gaussian_pyramid",
        "mask_gaussian_pyramid.png",
        "M",
    )
    blended_figure = _save_blend_pyramid(
        blended_levels,
        output_directory,
        "blended_laplacian_pyramid",
        "blended_laplacian_pyramid.png",
        "L",
        signed=True,
        reconstruction=blended,
    )

    comparison, axes = plt.subplots(1, 5, figsize=(19, 4))
    comparison_images = (first, second, mask, hard_cut, np.clip(blended, 0, 1))
    comparison_titles = (
        "First image",
        "Second image",
        "Sharp input mask",
        "Hard cut",
        "Multiresolution blend",
    )
    for axis, image, title in zip(axes, comparison_images, comparison_titles):
        axis.imshow(
            image,
            cmap="gray" if image.ndim == 2 else None,
            vmin=0 if image.ndim == 2 else None,
            vmax=1 if image.ndim == 2 else None,
        )
        axis.set_title(title)
        axis.axis("off")
    comparison.tight_layout()
    comparison.savefig(output_directory / "comparison.png", dpi=150)
    plt.show()
    plt.close(mask_figure)
    plt.close(blended_figure)
    plt.close(comparison)

    level_shapes = ", ".join(
        f"{level.shape[1]}x{level.shape[0]}" for level in mask_gaussian
    )
    summary = (
        f"Part 2.4 multiresolution blend: {name}\n"
        f"Description: {description}\n"
        f"Pyramid levels (largest to smallest): {level_shapes}\n"
        f"Blend formula at each level: M * L1 + (1 - M) * L2\n"
        "The input mask is binary and unchanged. Its Gaussian pyramid softens the "
        "transition over wider areas at coarser levels. Source Laplacian pyramids "
        "and the blended reconstruction are stored as floating-point .npy files.\n"
    )
    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return output_directory


def run_part_2_4():
    """Create the oraple, practice blend, and two irregular custom blends."""
    print("Running Part 2.4 multiresolution blending tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_blending.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_directory = CODE_DIR.parent / "images" / "part2_frequencies"
    apple = load_image(image_directory / "apple.jpg")
    orange = load_image(image_directory / "orange.jpg")
    oraple_mask = np.zeros(apple.shape[:2], dtype=np.float64)
    oraple_mask[:, :apple.shape[1] // 2] = 1.0
    _run_blend_example(
        "apple-orange",
        apple,
        orange,
        oraple_mask,
        "Apple is selected on the left; orange is selected on the right.",
    )

    hand = load_image(image_directory / "hand.jpg")
    eye = load_image(image_directory / "eye.jpg")
    hand_points = [(170, 400), (390, 410)]
    eye_points = [(165, 255), (470, 265)]
    aligned_eye = align_images(hand, eye, hand_points, eye_points)
    eye_mask = _ellipse_mask(hand.shape[:2], center=(270, 385), radii=(140, 75))
    _run_blend_example(
        "eye-on-hand",
        hand,
        aligned_eye,
        eye_mask,
        "Eye aligned to the palm using README landmarks; the black elliptical mask "
        "selects the eye and the white region keeps the hand.",
    )

    sunflower_path = image_directory / "sunflower.jpg"
    sunflower = load_image(sunflower_path)
    cat_path = CODE_DIR.parent / "images" / "part1_filters" / "cat.jpg"
    cat = load_image(cat_path)
    cat_points = [(330, 300), (440, 300)]
    sunflower_points = [(190, 204), (250, 204)]
    aligned_cat = align_images(sunflower, cat, sunflower_points, cat_points)
    cat_mask = _ellipse_union_mask(
        sunflower.shape[:2],
        [
            ((220, 200), (78, 76), -2),
            ((178, 145), (31, 43), -18),
            ((262, 145), (31, 43), 18),
        ],
    )
    _run_blend_example(
        "cat-on-sunflower",
        sunflower,
        aligned_cat,
        cat_mask,
        "Cat eyes aligned across the sunflower center; overlapping ellipses form "
        "a head-and-ears mask that selects the cat while keeping the petals.",
    )

    man = load_image(image_directory / "hybrid_man.jpg")
    aligned_man = align_images(
        apple,
        man,
        [(210, 222), (300, 222)],
        [(300, 342), (443, 333)],
    )
    face_mask = _ellipse_union_mask(
        apple.shape[:2],
        [
            ((255, 255), (101, 116), -3),
            ((255, 184), (78, 55), -3),
            ((255, 327), (58, 37), 0),
        ],
    )
    _run_blend_example(
        "face-on-apple",
        apple,
        aligned_man,
        face_mask,
        "The portrait's eyes align across the apple; three overlapping ellipses "
        "make an irregular face-shaped mask that selects the portrait.",
    )

    print("Saved Part 2.4 examples under: outputs/part2_4")
    return True


def run_part_3_1():
    """Remove 30% of the Broadway Tower image's width with minimum-energy seams."""
    print("Running Part 3.1 seam-carving tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_seam_carving.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = CODE_DIR.parent / "images" / "part3_seams" / "broadway_tower.png"
    image = load_image(image_path)
    height, width = image.shape[:2]
    seam_count = int(round(width * 0.30))
    carved, seams = carve_vertical(image, seam_count)
    energy = compute_energy(image)
    seam_overlay = draw_seams(image, seams)
    energy_maximum = float(energy.max())
    energy_display = energy / energy_maximum if energy_maximum > 0 else energy

    output_directory = CODE_DIR.parent / "outputs" / "part3_1"
    output_directory.mkdir(parents=True, exist_ok=True)
    save_image(output_directory / "original.png", image)
    save_image(output_directory / "energy.png", energy_display)
    save_image(output_directory / "removed_seams.png", seam_overlay)
    save_image(output_directory / "carved.png", carved)
    np.save(output_directory / "energy.npy", energy)
    np.save(output_directory / "removed_seams_original_columns.npy", seams)
    np.save(output_directory / "carved.npy", carved)

    figure, axes = plt.subplots(1, 4, figsize=(16, 5))
    for axis, displayed, title in zip(
        axes,
        (image, energy_display, seam_overlay, carved),
        ("Original", "Gradient-magnitude energy", "Removed seams (red)", "Carved image"),
    ):
        axis.imshow(
            displayed,
            cmap="gray" if displayed.ndim == 2 else None,
            vmin=0 if displayed.ndim == 2 else None,
            vmax=1 if displayed.ndim == 2 else None,
        )
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_directory / "comparison.png", dpi=150)
    plt.show()
    plt.close(figure)

    summary = (
        "Part 3.1: Finding and Removing Seams\n"
        f"Input: {image_path}\n"
        f"Original dimensions: {width} x {height}\n"
        "Requested reduction: 30% of original width\n"
        "Rounding: nearest whole seam count; round(0.30 * width)\n"
        f"Seams removed: {seam_count}\n"
        f"Final dimensions: {carved.shape[1]} x {carved.shape[0]}\n"
        "Energy: Part 1.2 finite-difference gradient magnitude on grayscale intensity.\n"
        "The dynamic program stores the cheapest cumulative cost ending at every "
        "pixel and its predecessor, then backtracks from the cheapest bottom-row "
        "pixel. This solves each seam in O(height * width) time. Trying every "
        "connected path would take exponential time because each row can branch "
        "to up to three columns below. Energy is recomputed after every removal.\n"
        "Original-column coordinates are tracked through each removal so the red "
        "overlay shows each seam at its location in the original image.\n"
        "Horizontal seam removal is available through carve_horizontal(), which "
        "transposes the image and reuses vertical seam carving.\n"
    )
    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved Part 3.1 results under: {output_directory}")
    return True


def _normalized_energy_display(energy):
    maximum = float(energy.max())
    return energy / maximum if maximum > 0 else energy


def _save_part_3_2_comparison(name, image_path, image, results, output_root):
    """Save the four same-sized width-reduction variants for one image."""
    output_directory = output_root / name
    output_directory.mkdir(parents=True, exist_ok=True)
    normal_width = results["target_width"]
    images = (
        ("original", image),
        ("normal_resize", results["resized"]),
        ("center_crop", results["center_crop"]),
        ("seam_carved", results["seam_carved"]),
    )
    for filename, result_image in images:
        save_image(output_directory / f"{filename}.png", result_image)
    np.save(output_directory / "removed_seams_original_columns.npy", results["seams"])

    figure, axes = plt.subplots(1, 4, figsize=(16, 5))
    titles = (
        f"Original ({image.shape[1]}x{image.shape[0]})",
        f"Normal resize ({normal_width}x{image.shape[0]})",
        f"Center crop ({normal_width}x{image.shape[0]})",
        f"Seam carved ({normal_width}x{image.shape[0]})",
    )
    for axis, (_, result_image), title in zip(axes, images, titles):
        axis.imshow(np.clip(result_image, 0, 1))
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_directory / "comparison.png", dpi=150)
    report_line = (
        f"{name}: source={image_path}; working={image.shape[1]}x{image.shape[0]}; "
        f"target={normal_width}x{image.shape[0]}; "
        f"seams={results['seam_count']}\n"
    )
    return report_line, figure


def _save_energy_comparison(name, image, finite_results, output_directory):
    """Compare Part 1.2 and 1.3 energy maps and the seams they select."""
    sigma = 2.0
    dog_results = compare_width_reduction(
        image,
        energy_function=lambda current: compute_dog_energy(current, sigma=sigma),
    )
    finite_seams = finite_results["seams"]
    dog_seams = dog_results["seams"]
    finite_energy = compute_energy(image)
    dog_energy = compute_dog_energy(image, sigma=sigma)
    finite_overlay = draw_seams(image, finite_seams, color=(1.0, 0.0, 0.0))
    dog_overlay = draw_seams(image, dog_seams, color=(0.0, 1.0, 1.0))
    combined_overlay = draw_seams(
        finite_overlay, dog_seams, color=(0.0, 1.0, 1.0)
    )
    save_image(output_directory / "energy_finite_difference.png",
               _normalized_energy_display(finite_energy))
    save_image(output_directory / f"energy_dog_sigma_{sigma:g}.png",
               _normalized_energy_display(dog_energy))
    save_image(output_directory / "seams_finite_difference_red.png", finite_overlay)
    save_image(output_directory / "seams_dog_cyan.png", dog_overlay)
    save_image(output_directory / "seams_energy_comparison.png", combined_overlay)
    save_image(output_directory / "seam_carved_dog.png", dog_results["seam_carved"])
    np.save(output_directory / "energy_finite_difference.npy", finite_energy)
    np.save(output_directory / "energy_dog.npy", dog_energy)
    np.save(output_directory / "seams_finite_difference.npy", finite_seams)
    np.save(output_directory / "seams_dog.npy", dog_seams)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = (
        (image, "Original"),
        (_normalized_energy_display(finite_energy), "Finite-difference energy"),
        (finite_overlay, "Finite-difference seams (red)"),
        (_normalized_energy_display(dog_energy), f"DoG energy (sigma={sigma:g})"),
        (dog_overlay, "DoG seams (cyan)"),
        (combined_overlay, "Both seam sets (red / cyan)"),
    )
    for axis, (panel, title) in zip(axes.flat, panels):
        axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout(h_pad=2.0)
    figure.savefig(output_directory / "energy_comparison.png", dpi=150)
    row_indices = np.arange(image.shape[0])[np.newaxis, :]
    finite_removed = np.zeros(image.shape[:2], dtype=bool)
    dog_removed = np.zeros(image.shape[:2], dtype=bool)
    finite_removed[row_indices, finite_seams] = True
    dog_removed[row_indices, dog_seams] = True
    intersection = int(np.count_nonzero(finite_removed & dog_removed))
    union = int(np.count_nonzero(finite_removed | dog_removed))
    return dog_results, (intersection, union), figure


def run_part_3_2(personal_images=None):
    """Compare seam carving, resizing, cropping, and energies on six images."""
    print("Running Part 3.2 seam-carving comparison tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_part3_2.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    using_default_images = personal_images is None
    image_directory = CODE_DIR.parent / "images" / "part3_seams"
    if using_default_images:
        personal_images = (
            image_directory / "person_outdoors.jpg",
            image_directory / "personal_lake.jpg",
            image_directory / "group_failure_case.jpg",
        )
    if len(personal_images) != 3:
        print("Provide exactly three --personal-image paths, or omit them for the discovered defaults.")
        return False

    personal_names = (
        ("person_outdoors", "lake", "group_failure_case")
        if using_default_images
        else ("personal_01", "personal_02", "personal_03")
    )
    image_sources = [
        ("rocket", image_directory / "rocket.jpg"),
        ("home", image_directory / "home.jpg"),
        ("starry_night", image_directory / "starry_night.jpg"),
        (personal_names[0], Path(personal_images[0]).expanduser()),
        (personal_names[1], Path(personal_images[1]).expanduser()),
        (personal_names[2], Path(personal_images[2]).expanduser()),
    ]
    missing = [str(path) for _, path in image_sources if not path.is_file()]
    if missing:
        print("Could not find required images:\n  " + "\n  ".join(missing))
        return False

    output_root = CODE_DIR.parent / "outputs" / "part3_2"
    output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    comparison_figures = []
    finite_results_by_name = {}
    working_images = {}
    for name, image_path in image_sources:
        image = _limit_image_size(load_image(image_path))
        results = compare_width_reduction(image, energy_function=compute_energy)
        report_line, comparison_figure = (
            _save_part_3_2_comparison(
                name, image_path, image, results, output_root
            )
        )
        reports.append(report_line)
        comparison_figures.append(comparison_figure)
        finite_results_by_name[name] = results
        working_images[name] = image

    failure_name = personal_names[2]
    failure_output = output_root / failure_name
    dog_results, mask_overlap, energy_figure = _save_energy_comparison(
        failure_name,
        working_images[failure_name],
        finite_results_by_name[failure_name],
        failure_output,
    )
    intersection, union = mask_overlap
    overlap_fraction = intersection / union if union else 1.0
    dog_report = (
        f"Energy comparison image: {failure_name}; finite-difference vs DoG "
        f"(sigma=2.0) removed-pixel mask Jaccard overlap: "
        f"|A\u2229B| / |A\u222aB| = {intersection} / {union} "
        f"= {overlap_fraction:.1%}.\n"
        f"DoG carved dimensions: {dog_results['seam_carved'].shape[1]} x "
        f"{dog_results['seam_carved'].shape[0]}.\n"
    )
    summary = (
        "Part 3.2: Results and Comparisons\n"
        "Each working image is capped at 1000 pixels on its longest side before "
        "comparison, following the assignment's medium-size image guidance.\n"
        "The target width is the working width minus round(0.30 * working width); "
        "the PDF says about 30% and gives no further rounding rule. All three "
        "narrowed methods preserve height and use that same target width.\n"
        "Normal resizing uses OpenCV INTER_AREA. Center cropping removes equal "
        "horizontal margins. Seam carving uses Part 1.2 finite-difference magnitude "
        "for the six standard comparisons.\n"
        + "".join(reports)
        + dog_report
        + "The finite-difference and DoG energy maps differ in smoothing and edge "
        "response; their distinct seam sets are shown in the energy-comparison "
        "figure, with finite-difference paths in red and DoG paths in cyan.\n"
        + "Each image folder contains the four-way comparison, and the group-photo "
        "folder additionally contains both energy maps, both carved outputs, and "
        "the overlaid seam comparison.\n"
    )
    (output_root / "summary.txt").write_text(summary, encoding="utf-8")
    plt.show()
    for figure in comparison_figures + [energy_figure]:
        plt.close(figure)
    print(summary)
    print(f"Saved Part 3.2 results under: {output_root}")
    return True


def run_bonus(interactive_mask=False):
    """Remove the standing person from a personal photo and restore its width."""
    print("Running Bonus seam-carving tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_bonus.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = Path.home() / "Downloads" / "person3.jpg"
    if not image_path.is_file():
        print(f"Personal bonus image was not found: {image_path}")
        return False
    image = _limit_image_size(load_image(image_path))
    if interactive_mask:
        mask = select_polygon_mask(image)
        polygon_vertices = None
    else:
        normalized_vertices = np.array(
            [
                (0.482, 0.695), (0.489, 0.682), (0.499, 0.683),
                (0.507, 0.697), (0.509, 0.721), (0.512, 0.741),
                (0.520, 0.758), (0.519, 0.790), (0.514, 0.824),
                (0.516, 0.882), (0.519, 0.999), (0.470, 0.999),
                (0.474, 0.890), (0.476, 0.842), (0.468, 0.808),
                (0.463, 0.781), (0.469, 0.758), (0.478, 0.744),
                (0.479, 0.720),
            ],
            dtype=np.float64,
        )
        polygon_vertices = normalized_vertices * np.array(
            [image.shape[1] - 1, image.shape[0] - 1]
        )
        mask = polygon_mask(image.shape, polygon_vertices)

    output_directory = CODE_DIR.parent / "outputs" / "bonus"
    output_directory.mkdir(parents=True, exist_ok=True)
    np.save(output_directory / "initial_energy.npy", compute_energy(image))
    reduced, seam_count, remaining_mask = remove_masked_object(image, mask)
    remaining_mask_count = int(np.count_nonzero(remaining_mask))
    if remaining_mask_count:
        raise RuntimeError("object removal finished with masked pixels remaining")
    restored, reduced_locations, inserted_positions = restore_width_by_insertion(
        reduced, seam_count
    )
    if restored.shape != image.shape:
        raise RuntimeError(
            f"width restoration returned {restored.shape}, expected {image.shape}"
        )
    restored_overlay = draw_inserted_seams(restored, inserted_positions)
    figure = save_bonus_outputs(
        output_directory,
        image,
        mask,
        reduced,
        restored,
        restored_overlay,
        inserted_positions,
    )
    np.save(output_directory / "seams_found_on_reduced_copy.npy", reduced_locations)
    if polygon_vertices is not None:
        (output_directory / "mask_polygon.json").write_text(
            json.dumps(
                {
                    "source_image": str(image_path),
                    "vertices_xy": polygon_vertices.tolist(),
                    "working_dimensions_wh": [image.shape[1], image.shape[0]],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    original_height, original_width = image.shape[:2]
    summary = (
        "Bonus: Seam-Carving Object Removal and Seam Insertion\n"
        f"Source photo: {image_path}\n"
        f"Mask source: {'interactive polygon selection' if interactive_mask else 'saved polygon around the person in person3.jpg'}\n"
        f"Initial dimensions: {original_width} x {original_height}\n"
        f"Final reduced dimensions: {reduced.shape[1]} x {reduced.shape[0]}\n"
        f"Seams removed (k): {seam_count}\n"
        f"Restored dimensions: {restored.shape[1]} x {restored.shape[0]}\n"
        f"Masked pixels remaining: {remaining_mask_count}\n"
        "Object removal uses float64 gradient energy and sets masked pixels to -1e6 "
        "before each minimum-seam search. The seam is removed from both the image "
        "and mask until the mask is empty.\n"
        "To restore width, k seams are found by removing seams from a copy of the "
        "reduced image while tracking original reduced-image columns. Those distinct "
        "locations are then inserted into the actual reduced image; each seam pixel "
        "is duplicated as left- and right-neighbor averages, using itself at borders. "
        "Finding seams on a copy avoids repeatedly choosing the same path after each "
        "insertion changes the image.\n"
        "The binary mask and selected polygon are saved for reproducibility. Use "
        "--select-bonus-mask to redraw the polygon interactively.\n"
    )
    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    plt.show()
    plt.close(figure)
    print(summary)
    print(f"Saved Bonus results under: {output_directory}")
    return True


def run_section(
    section, threshold=0.2, sigma=2.0, hybrid_options=None, dog_threshold=None
):
    """Run an implemented section, or report that it is not ready yet."""
    if section == "1.2":
        return run_part_1_2(threshold)
    if section == "1.3":
        return run_part_1_3(
            threshold if dog_threshold is None else dog_threshold,
            sigma,
            finite_difference_threshold=threshold,
        )
    if section == "2.1":
        return run_part_2_1(hybrid_options)
    if section == "2.2":
        return run_part_2_2(hybrid_options.aliasing_sigma)
    if section == "2.3":
        return run_part_2_3()
    if section == "2.4":
        return run_part_2_4()
    if section == "3.1":
        return run_part_3_1()
    if section == "3.2":
        return run_part_3_2(getattr(hybrid_options, "personal_image", None))
    if section == "bonus":
        return run_bonus(getattr(hybrid_options, "select_bonus_mask", False))
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


def run_selection(
    selection, threshold=None, sigma=2.0, hybrid_options=None, dog_threshold=None
):
    """Run one section, every section in a part, or all known sections."""
    if threshold is None:
        threshold = 0.2
        if dog_threshold is None:
            dog_threshold = 0.08
    elif dog_threshold is None:
        dog_threshold = threshold

    if selection == "all":
        sections = ALL_SECTIONS
    elif selection in PART_SECTIONS:
        sections = PART_SECTIONS[selection]
    else:
        sections = (selection,)

    success = True
    for section in sections:
        if not run_section(
            section, threshold, sigma, hybrid_options, dog_threshold
        ):
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
        default=None,
        help=(
            "Part 1.2 edge threshold (default: 0.2); when supplied, also sets "
            "Part 1.3 unless --dog-threshold is given"
        ),
    )
    parser.add_argument(
        "--dog-threshold",
        type=float,
        default=None,
        help="Part 1.3 edge threshold (default: 0.08)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Part 1.3 Gaussian standard deviation (default: 2.0)",
    )
    parser.add_argument(
        "--sigma-low",
        type=float,
        default=4.0,
        help="Part 2.1 low-pass Gaussian sigma (default: 4.0)",
    )
    parser.add_argument(
        "--sigma-high",
        type=float,
        default=4.0,
        help="Part 2.1 high-pass Gaussian sigma (default: 4.0)",
    )
    parser.add_argument(
        "--aliasing-sigma",
        type=float,
        default=2.0,
        help="Part 2.2 Gaussian prefilter sigma (default: 2.0)",
    )
    parser.add_argument(
        "--custom-pair",
        nargs=2,
        action="append",
        metavar=("FIRST_IMAGE", "SECOND_IMAGE"),
        help="optionally replace both default custom pairs (repeat this option twice)",
    )
    parser.add_argument(
        "--favorite",
        choices=("man-cat", "cat-tiger", "dodge-motorcycle", "custom-1", "custom-2"),
        default="man-cat",
        help="pair to include in the image/Fourier comparison (default: man-cat)",
    )
    parser.add_argument(
        "--alignment-points",
        action="append",
        nargs=9,
        metavar=("PAIR", "X1", "Y1", "X2", "Y2", "U1", "V1", "U2", "V2"),
        help=(
            "provide points in original-image pixels: pair, two points for the "
            "first image, then two points for the second; repeat per pair"
        ),
    )
    parser.add_argument(
        "--reselect-points",
        action="append",
        choices=("man-cat", "cat-tiger", "dodge-motorcycle", "custom-1", "custom-2"),
        help="interactively choose this pair's landmarks again; repeat to reselect multiple",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/part2_1",
        help="result directory (default: outputs/part2_1)",
    )
    parser.add_argument(
        "--personal-image",
        action="append",
        metavar="IMAGE",
        help=(
            "Part 3.2 personal image path; repeat exactly three times to replace "
            "the discovered local defaults"
        ),
    )
    parser.add_argument(
        "--select-bonus-mask",
        action="store_true",
        help="interactively outline the object for the optional seam-carving Bonus",
    )
    arguments = parser.parse_args()
    return (
        0
        if run_selection(
            arguments.part,
            arguments.threshold,
            arguments.sigma,
            arguments,
            arguments.dog_threshold,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
