"""Part 2.2 direct and Gaussian-prefiltered image downsampling."""
import numpy as np

from hybrid_images import gaussian_blur


DEFAULT_SIGMA_SWEEP = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.25)


def downsample_by_factor(image, factor=4):
    """Keep every factor-th row and column without applying a filter."""
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    return image[::factor, ::factor]


def aliasing_comparison(image, sigma=2.0, factor=4):
    """Return blurred input, direct sample, and Gaussian-prefiltered sample."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive number")
    blurred = gaussian_blur(image, sigma)
    direct_downsampled = downsample_by_factor(image, factor)
    prefiltered_downsampled = downsample_by_factor(blurred, factor)
    return blurred, direct_downsampled, prefiltered_downsampled


def sigma_sweep(image, sigmas=DEFAULT_SIGMA_SWEEP, factor=4):
    """Return (sigma, downsampled image) entries for visual sigma comparison."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")
    return [
        (float(sigma), downsample_by_factor(gaussian_blur(image, sigma), factor))
        for sigma in sigmas
    ]
