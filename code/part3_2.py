"""Comparable width-reduction methods for Part 3.2."""
import cv2
import numpy as np

from seam_carving import carve_vertical, compute_energy


def target_width(original_width, reduction_fraction=0.30):
    """Return width after rounding the requested seam count to nearest integer."""
    if not isinstance(original_width, (int, np.integer)) or original_width <= 1:
        raise ValueError("original_width must be an integer greater than one")
    if not np.isfinite(reduction_fraction) or not 0 < reduction_fraction < 1:
        raise ValueError("reduction_fraction must be between zero and one")
    seam_count = int(round(original_width * reduction_fraction))
    if seam_count >= original_width:
        raise ValueError("reduction_fraction removes the entire image width")
    return original_width - seam_count


def compare_width_reduction(image, energy_function=compute_energy,
                            reduction_fraction=0.30):
    """Return original, normal resize, centered crop, and seam-carved images."""
    image = np.asarray(image)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or RGB array")
    height, width = image.shape[:2]
    reduced_width = target_width(width, reduction_fraction)
    seam_count = width - reduced_width
    resized = cv2.resize(
        image,
        (reduced_width, height),
        interpolation=cv2.INTER_AREA,
    )
    crop_start = (width - reduced_width) // 2
    center_crop = image[:, crop_start:crop_start + reduced_width].copy()
    seam_carved, seams = carve_vertical(
        image, seam_count, energy_function=energy_function
    )
    expected_shape = (height, reduced_width) + image.shape[2:]
    results = (resized, center_crop, seam_carved)
    if any(result.shape != expected_shape for result in results):
        raise RuntimeError("width-reduction methods returned mismatched dimensions")
    return {
        "original": image.copy(),
        "resized": resized,
        "center_crop": center_crop,
        "seam_carved": seam_carved,
        "seams": seams,
        "target_width": reduced_width,
        "seam_count": seam_count,
    }