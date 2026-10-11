"""Multiresolution image blending for Part 2.4."""
import numpy as np

from pyramids import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_laplacian,
)


def build_blended_laplacian_pyramid(
    first_image, second_image, mask, min_size=16
):
    """Build source Laplacians, a Gaussian mask, and their blended Laplacian."""
    first_image = np.asarray(first_image, dtype=np.float64)
    second_image = np.asarray(second_image, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if first_image.ndim not in (2, 3) or first_image.size == 0:
        raise ValueError("first_image must be a non-empty grayscale or RGB image")
    if second_image.shape != first_image.shape:
        raise ValueError("source images must have matching dimensions and channels")
    if not np.isfinite(first_image).all() or not np.isfinite(second_image).all():
        raise ValueError("source images must contain only finite values")
    if mask.shape != first_image.shape[:2]:
        raise ValueError("mask must be a 2D image matching the source dimensions")
    if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1):
        raise ValueError("mask values must be finite and within [0, 1]")

    first_gaussian = build_gaussian_pyramid(first_image, min_size=min_size)
    second_gaussian = build_gaussian_pyramid(second_image, min_size=min_size)
    mask_gaussian = build_gaussian_pyramid(mask, min_size=min_size)
    first_laplacian = build_laplacian_pyramid(first_gaussian)
    second_laplacian = build_laplacian_pyramid(second_gaussian)
    if not (
        len(first_laplacian) == len(second_laplacian) == len(mask_gaussian)
    ):
        raise ValueError("source and mask pyramids must have matching levels")

    blended_laplacian = []
    for first_level, second_level, mask_level in zip(
        first_laplacian, second_laplacian, mask_gaussian
    ):
        if first_level.shape != second_level.shape:
            raise ValueError("source pyramid levels must have matching dimensions")
        if first_level.ndim == 3:
            mask_level = mask_level[..., np.newaxis]
        blended_laplacian.append(
            mask_level * first_level + (1.0 - mask_level) * second_level
        )
    return blended_laplacian, first_laplacian, second_laplacian, mask_gaussian


def multiresolution_blend(first_image, second_image, mask, min_size=16):
    """Blend two same-sized images by their Laplacian bands and a mask pyramid."""
    blended_laplacian, _, _, _ = build_blended_laplacian_pyramid(
        first_image, second_image, mask, min_size=min_size
    )
    return reconstruct_from_laplacian(blended_laplacian)