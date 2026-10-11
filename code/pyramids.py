"""Gaussian and Laplacian pyramid operations for Part 2.3."""
import numpy as np

from filters import convolve_separable


BINOMIAL_FILTER = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0


def _as_float_image(image):
    """Validate a grayscale or channel-last image and return float64 data."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or channel-last array")
    if image.ndim == 3 and image.shape[2] == 0:
        raise ValueError("image must have at least one channel")
    return image


def _blur_binomial(image):
    """Blur horizontally and vertically with the assignment's 1D filter."""
    if image.ndim == 2:
        horizontal = convolve_separable(
            image, BINOMIAL_FILTER[np.newaxis, :], padding_mode="reflect"
        )
        return convolve_separable(
            horizontal, BINOMIAL_FILTER[:, np.newaxis], padding_mode="reflect"
        )

    blurred_channels = []
    for channel in range(image.shape[2]):
        horizontal = convolve_separable(
            image[..., channel],
            BINOMIAL_FILTER[np.newaxis, :],
            padding_mode="reflect",
        )
        blurred_channels.append(
            convolve_separable(
                horizontal,
                BINOMIAL_FILTER[:, np.newaxis],
                padding_mode="reflect",
            )
        )
    return np.stack(blurred_channels, axis=-1)


def reduce(image):
    """Blur an image with the binomial filter, then keep every other sample."""
    image = _as_float_image(image)
    return _blur_binomial(image)[::2, ::2]


def expand(image):
    """Zero-insert, blur with the binomial filter, and multiply by four."""
    image = _as_float_image(image)
    expanded_shape = (image.shape[0] * 2, image.shape[1] * 2) + image.shape[2:]
    expanded = np.zeros(expanded_shape, dtype=np.float64)
    expanded[::2, ::2] = image
    return 4.0 * _blur_binomial(expanded)


def build_gaussian_pyramid(image, min_size=16):
    """Return largest-to-smallest levels, reducing until a side is at most min_size."""
    image = _as_float_image(image)
    if not isinstance(min_size, (int, np.integer)) or min_size < 1:
        raise ValueError("min_size must be a positive integer")

    levels = [image.copy()]
    while min(levels[-1].shape[:2]) > min_size:
        levels.append(reduce(levels[-1]))
    return levels


def _crop_expanded(image, target_shape):
    """Crop only trailing rows and columns needed to match a pyramid level."""
    target_height, target_width = target_shape[:2]
    if image.shape[0] < target_height or image.shape[1] < target_width:
        raise ValueError("expanded image is smaller than the target pyramid level")
    if image.ndim != len(target_shape) or image.shape[2:] != target_shape[2:]:
        raise ValueError("pyramid levels must have matching channel dimensions")
    return image[:target_height, :target_width]


def build_laplacian_pyramid(gaussian_levels):
    """Store each Gaussian level's difference from the expanded next level."""
    if not gaussian_levels:
        raise ValueError("gaussian_levels must contain at least one image")
    gaussian_levels = [_as_float_image(level) for level in gaussian_levels]

    laplacian_levels = []
    for current, smaller in zip(gaussian_levels, gaussian_levels[1:]):
        expanded = _crop_expanded(expand(smaller), current.shape)
        laplacian_levels.append(current - expanded)
    laplacian_levels.append(gaussian_levels[-1].copy())
    return laplacian_levels


def reconstruct_from_laplacian(laplacian_levels):
    """Rebuild an image from largest-to-smallest Laplacian levels."""
    if not laplacian_levels:
        raise ValueError("laplacian_levels must contain at least one image")
    laplacian_levels = [_as_float_image(level) for level in laplacian_levels]
    reconstruction = laplacian_levels[-1].copy()

    for level in reversed(laplacian_levels[:-1]):
        expanded = _crop_expanded(expand(reconstruction), level.shape)
        reconstruction = expanded + level
    return reconstruction