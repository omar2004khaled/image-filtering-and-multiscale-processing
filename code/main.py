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
from filters import derivative_of_gaussian_edges, finite_difference_edges
from hybrid_images import fourier_log_magnitude, make_hybrid
from utils import load_image, save_image, show


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


def run_part_1_3(threshold, sigma):
    """Test and display derivative-of-Gaussian edges on cameraman."""
    print("Running Part 1.3 correctness tests...")
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(CODE_DIR),
        pattern="test_dog.py",
        top_level_dir=str(CODE_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return False

    image_path = CODE_DIR.parent / "images" / "part1_filters" / "cameraman.png"
    image = load_image(image_path, gray=True)
    gaussian, dog_x, dog_y, ix, iy, magnitude, edges = (
        derivative_of_gaussian_edges(image, sigma=sigma, threshold=threshold)
    )
    _, _, _, finite_difference_edge_map = finite_difference_edges(
        image, threshold=threshold
    )

    magnitude_maximum = float(magnitude.max())
    magnitude_display = magnitude / magnitude_maximum if magnitude_maximum > 0 else magnitude
    print(f"DoG sigma: {sigma:g}; edge threshold: {threshold:g}.")
    print(
        "DoG combines smoothing and differentiation, generally reducing "
        "sensitivity to small-scale noise."
    )
    print(
        "Edge pixels: Part 1.2 = "
        f"{np.count_nonzero(finite_difference_edge_map)}, Part 1.3 = "
        f"{np.count_nonzero(edges)}."
    )
    show(
        image,
        gaussian,
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


def run_section(section, threshold=0.2, sigma=2.0, hybrid_options=None):
    """Run an implemented section, or report that it is not ready yet."""
    if section == "1.2":
        return run_part_1_2(threshold)
    if section == "1.3":
        return run_part_1_3(threshold, sigma)
    if section == "2.1":
        return run_part_2_1(hybrid_options)
    if section == "2.2":
        return run_part_2_2(hybrid_options.aliasing_sigma)
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


def run_selection(selection, threshold=0.2, sigma=2.0, hybrid_options=None):
    """Run one section, every section in a part, or all known sections."""
    if selection == "all":
        sections = ALL_SECTIONS
    elif selection in PART_SECTIONS:
        sections = PART_SECTIONS[selection]
    else:
        sections = (selection,)

    success = True
    for section in sections:
        if not run_section(section, threshold, sigma, hybrid_options):
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
        help="edge threshold for Parts 1.2 and 1.3 (default: 0.2)",
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
    arguments = parser.parse_args()
    return (
        0
        if run_selection(
            arguments.part,
            arguments.threshold,
            arguments.sigma,
            arguments,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
