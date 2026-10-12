"""Object removal and width restoration for the seam-carving Bonus."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from seam_carving import compute_energy, find_vertical_seam, remove_vertical_seam
from utils import save_image


def polygon_mask(shape, vertices):
    """Rasterize polygon vertices into a boolean mask matching (height, width)."""
    height, width = shape[:2]
    points = np.asarray(vertices, dtype=np.float64)
    if height < 1 or width < 1:
        raise ValueError("mask dimensions must be positive")
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("vertices must contain at least three (x, y) points")
    if not np.isfinite(points).all():
        raise ValueError("polygon vertices must be finite")
    if (
        np.any(points[:, 0] < 0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("polygon vertices must lie inside the image")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def select_polygon_mask(image):
    """Let a user click polygon vertices; right-click finishes the selection."""
    image = np.asarray(image)
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(np.clip(image, 0, 1))
    axis.set_title("Click around the object; right-click to finish (3+ points)")
    axis.axis("off")
    try:
        vertices = plt.ginput(n=-1, timeout=0, show_clicks=True)
    finally:
        plt.close(figure)
    if len(vertices) < 3:
        raise ValueError("object mask selection needs at least three polygon points")
    return polygon_mask(image.shape, vertices)


def _as_image(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or RGB image")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("color image must have three RGB channels")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return image


def remove_masked_object(image, mask, energy_function=compute_energy,
                         mask_energy=-1e6):
    """Remove masked pixels with negative-energy seams until the mask is empty."""
    image = _as_image(image)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("mask dimensions must match the image")
    if not np.any(mask):
        raise ValueError("mask must cover at least one pixel")
    if not np.isfinite(mask_energy) or mask_energy >= 0:
        raise ValueError("mask_energy must be a finite negative value")
    if not callable(energy_function):
        raise ValueError("energy_function must be callable")

    reduced = image.copy()
    remaining_mask = mask.copy()
    removed_seams = 0
    while np.any(remaining_mask):
        if reduced.shape[1] <= 1:
            raise ValueError("cannot remove remaining mask from a one-column image")
        energy = np.asarray(energy_function(reduced), dtype=np.float64).copy()
        if energy.shape != remaining_mask.shape or not np.isfinite(energy).all():
            raise ValueError("energy_function must return a finite image-sized map")
        energy[remaining_mask] = mask_energy
        seam = find_vertical_seam(energy)
        reduced = remove_vertical_seam(reduced, seam)
        remaining_mask = remove_vertical_seam(remaining_mask, seam).astype(bool)
        removed_seams += 1
    return reduced, removed_seams, remaining_mask


def find_insertion_locations(image, seam_count, energy_function=compute_energy):
    """Find distinct seams on a shrinking copy and map them to source columns."""
    image = _as_image(image)
    height, width = image.shape[:2]
    if not isinstance(seam_count, (int, np.integer)) or seam_count < 0:
        raise ValueError("seam_count must be a non-negative integer")
    if seam_count >= width:
        raise ValueError("seam_count must be smaller than the image width")
    if not callable(energy_function):
        raise ValueError("energy_function must be callable")

    working_copy = image.copy()
    coordinate_map = np.tile(np.arange(width, dtype=np.int64), (height, 1))
    rows = np.arange(height)
    locations = []
    for _ in range(seam_count):
        energy = np.asarray(energy_function(working_copy), dtype=np.float64)
        if energy.shape != working_copy.shape[:2] or not np.isfinite(energy).all():
            raise ValueError("energy_function must return a finite image-sized map")
        seam = find_vertical_seam(energy)
        locations.append(coordinate_map[rows, seam].copy())
        working_copy = remove_vertical_seam(working_copy, seam)
        coordinate_map = remove_vertical_seam(coordinate_map, seam).astype(np.int64)
    if not locations:
        return np.empty((0, height), dtype=np.int64)
    return np.stack(locations)


def insert_vertical_seams(image, locations):
    """Insert two neighbor-averaged copies at each unique source seam pixel."""
    image = _as_image(image)
    height, width = image.shape[:2]
    locations = np.asarray(locations)
    if locations.ndim != 2 or not np.issubdtype(locations.dtype, np.integer):
        raise ValueError("locations must be a 2D integer array with one row per seam")
    if locations.shape[1] != height:
        raise ValueError("each seam must contain one source column per image row")
    if np.any(locations < 0) or np.any(locations >= width):
        raise ValueError("seam locations must lie inside the reduced image")

    seam_count = locations.shape[0]
    restored_shape = (height, width + seam_count) + image.shape[2:]
    restored = np.empty(restored_shape, dtype=np.float64)
    inserted_positions = np.empty((seam_count, height, 2), dtype=np.int64)

    for row in range(height):
        row_seam_order = np.argsort(locations[:, row], kind="stable")
        seam_columns = locations[row_seam_order, row]
        if np.unique(seam_columns).size != seam_count:
            raise ValueError("seam locations must be unique within each row")
        insertion_map = {
            int(column): int(seam_index)
            for column, seam_index in zip(seam_columns, row_seam_order)
        }
        output_column = 0
        for source_column in range(width):
            pixel = image[row, source_column]
            if source_column in insertion_map:
                seam_index = insertion_map[source_column]
                left = image[row, max(0, source_column - 1)]
                right = image[row, min(width - 1, source_column + 1)]
                inserted_positions[seam_index, row] = (
                    output_column,
                    output_column + 1,
                )
                restored[row, output_column] = 0.5 * (pixel + left)
                restored[row, output_column + 1] = 0.5 * (pixel + right)
                output_column += 2
            else:
                restored[row, output_column] = pixel
                output_column += 1
        if output_column != width + seam_count:
            raise RuntimeError("seam insertion produced an incorrect row width")
    return restored, inserted_positions


def restore_width_by_insertion(image, seam_count, energy_function=compute_energy):
    """Find seams on a copy, then insert them into the original reduced image."""
    image = _as_image(image)
    locations = find_insertion_locations(image, seam_count, energy_function)
    restored, inserted_positions = insert_vertical_seams(image, locations)
    return restored, locations, inserted_positions


def draw_inserted_seams(image, inserted_positions, color=(0.0, 1.0, 0.0)):
    """Draw both interpolated columns of every inserted seam in green."""
    image = _as_image(image)
    if image.ndim == 2:
        overlay = np.repeat(image[..., np.newaxis], 3, axis=2)
    else:
        overlay = image.copy()
    positions = np.asarray(inserted_positions)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("inserted_positions must have shape (seams, height, 2)")
    if positions.shape[1] != image.shape[0]:
        raise ValueError("inserted seam positions must match image height")
    if np.any(positions < 0) or np.any(positions >= image.shape[1]):
        raise ValueError("inserted seam positions must lie inside the image")
    for seam in positions:
        overlay[np.arange(image.shape[0]), seam[:, 0]] = color
        overlay[np.arange(image.shape[0]), seam[:, 1]] = color
    return overlay


def mask_preview(image, mask):
    """Tint masked object pixels red for the saved and displayed preview."""
    image = _as_image(image)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("mask dimensions must match the image")
    if image.ndim == 2:
        preview = np.repeat(image[..., np.newaxis], 3, axis=2)
    else:
        preview = image.copy()
    red = np.array([1.0, 0.0, 0.0])
    preview[mask] = 0.35 * preview[mask] + 0.65 * red
    return preview


def save_bonus_figure(path, original, mask, reduced, restored_overlay):
    """Save the required three-panel Bonus report figure and return its figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        (mask_preview(original, mask), "Original with object mask (red)"),
        (reduced, f"Object removed ({reduced.shape[1]}x{reduced.shape[0]})"),
        (restored_overlay, "Restored width; inserted seams (green)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(np.clip(panel, 0, 1), cmap="gray" if panel.ndim == 2 else None)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def save_bonus_outputs(output_directory, original, mask, reduced, restored,
                       restored_overlay, inserted_locations):
    """Save the reproducible mask, images, coordinate data, and report figure."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    save_image(output_directory / "original.png", original)
    save_image(output_directory / "mask_preview.png", mask_preview(original, mask))
    save_image(output_directory / "object_mask.png", np.asarray(mask, dtype=float))
    save_image(output_directory / "object_removed.png", reduced)
    save_image(output_directory / "restored.png", restored)
    save_image(output_directory / "restored_inserted_seams_green.png", restored_overlay)
    np.save(output_directory / "object_mask.npy", np.asarray(mask, dtype=bool))
    np.save(output_directory / "object_removed.npy", reduced)
    np.save(output_directory / "restored.npy", restored)
    np.save(output_directory / "inserted_seam_output_positions.npy", inserted_locations)
    figure = save_bonus_figure(
        output_directory / "bonus_comparison.png",
        original,
        mask,
        reduced,
        restored_overlay,
    )
    return figure