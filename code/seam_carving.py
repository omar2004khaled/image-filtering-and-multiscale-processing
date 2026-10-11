"""Seam carving operations for Part 3.1."""
import numpy as np

from filters import derivative_of_gaussian_edges, finite_difference_edges
from hybrid_images import to_grayscale


def _as_image(image):
    """Validate an image and convert it to float64 without changing its shape."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or RGB array")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color image must have three RGB channels")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return image


def compute_energy(image):
    """Return Part 1.2's gradient magnitude as a grayscale pixel-energy map."""
    image = _as_image(image)
    gray_image = to_grayscale(image)
    _, _, gradient_magnitude, _ = finite_difference_edges(gray_image)
    return gradient_magnitude


def compute_dog_energy(image, sigma=2.0):
    """Return the Part 1.3 DoG gradient magnitude as a pixel-energy map."""
    image = _as_image(image)
    gray_image = to_grayscale(image)
    return derivative_of_gaussian_edges(gray_image, sigma=sigma)[5]


def find_vertical_seam(energy):
    """Find a minimum-energy top-to-bottom seam by dynamic programming."""
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 2 or energy.size == 0:
        raise ValueError("energy must be a non-empty 2D array")
    if not np.isfinite(energy).all():
        raise ValueError("energy must contain only finite values")

    height, width = energy.shape
    cumulative = energy.copy()
    predecessors = np.zeros((height, width), dtype=np.int64)
    columns = np.arange(width)

    for row in range(1, height):
        previous = cumulative[row - 1]
        from_left = np.full(width, np.inf)
        from_left[1:] = previous[:-1]
        from_above = previous
        from_right = np.full(width, np.inf)
        from_right[:-1] = previous[1:]
        candidates = np.stack((from_left, from_above, from_right))
        best_predecessor = np.argmin(candidates, axis=0)
        cumulative[row] += candidates[best_predecessor, columns]
        predecessors[row] = columns + best_predecessor - 1

    seam = np.empty(height, dtype=np.int64)
    seam[-1] = int(np.argmin(cumulative[-1]))
    for row in range(height - 1, 0, -1):
        seam[row - 1] = predecessors[row, seam[row]]
    return seam


def remove_vertical_seam(image, seam):
    """Remove one connected vertical seam from a grayscale or RGB image."""
    image = _as_image(image)
    height, width = image.shape[:2]
    seam = np.asarray(seam)
    if seam.shape != (height,) or not np.issubdtype(seam.dtype, np.integer):
        raise ValueError("seam must contain one integer column per image row")
    if width <= 1 or np.any(seam < 0) or np.any(seam >= width):
        raise ValueError("seam columns must be in range and image width must exceed one")
    if np.any(np.abs(np.diff(seam)) > 1):
        raise ValueError("seam columns must be connected between adjacent rows")

    keep = np.ones((height, width), dtype=bool)
    keep[np.arange(height), seam] = False
    remaining_shape = (height, width - 1) + image.shape[2:]
    return image[keep].reshape(remaining_shape)


def carve_vertical(image, seam_count, energy_function=compute_energy):
    """Remove vertical seams using energy_function and track original columns."""
    image = _as_image(image)
    if not isinstance(seam_count, (int, np.integer)) or seam_count < 0:
        raise ValueError("seam_count must be a non-negative integer")
    if not callable(energy_function):
        raise ValueError("energy_function must be callable")
    height, width = image.shape[:2]
    if seam_count >= width:
        raise ValueError("seam_count must be smaller than the image width")

    carved = image.copy()
    original_columns = np.tile(np.arange(width, dtype=np.int64), (height, 1))
    removed_seams = []
    rows = np.arange(height)
    for _ in range(seam_count):
        energy = np.asarray(energy_function(carved), dtype=np.float64)
        if energy.shape != carved.shape[:2] or not np.isfinite(energy).all():
            raise ValueError("energy_function must return a finite map matching the image")
        seam = find_vertical_seam(energy)
        removed_seams.append(original_columns[rows, seam].copy())
        carved = remove_vertical_seam(carved, seam)
        original_columns = remove_vertical_seam(original_columns, seam)
    seams = np.asarray(removed_seams, dtype=np.int64).reshape(seam_count, height)
    return carved, seams


def carve_horizontal(image, seam_count, energy_function=compute_energy):
    """Remove horizontal seams by transposing and reusing vertical carving."""
    image = _as_image(image)
    transposed = np.swapaxes(image, 0, 1)
    carved, seams = carve_vertical(
        transposed, seam_count, energy_function=energy_function
    )
    return np.swapaxes(carved, 0, 1), seams


def draw_seams(image, seams, orientation="vertical", color=(1.0, 0.0, 0.0)):
    """Draw removed seam coordinates on a copy of the original image in red."""
    image = _as_image(image)
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if image.ndim == 2:
        overlay = np.repeat(image[..., np.newaxis], 3, axis=2)
    else:
        overlay = image.copy()

    seams = np.asarray(seams)
    if seams.ndim != 2 or not np.issubdtype(seams.dtype, np.integer):
        raise ValueError("seams must be a 2D array of integer coordinates")
    height, width = image.shape[:2]
    expected_length = height if orientation == "vertical" else width
    bound = width if orientation == "vertical" else height
    if seams.shape[1] != expected_length:
        raise ValueError("seam coordinate arrays do not match the image orientation")
    if np.any(seams < 0) or np.any(seams >= bound):
        raise ValueError("seam coordinates fall outside the image")
    color = np.asarray(color, dtype=np.float64)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError("color must contain three finite RGB values")

    if orientation == "vertical":
        for seam in seams:
            overlay[np.arange(height), seam] = color
    else:
        for seam in seams:
            overlay[seam, np.arange(width)] = color
    return overlay