"""Core filtering and spectrum functions for Assignment 2.1 hybrid images."""
import numpy as np

from filters import convolve_separable, gaussian_kernel


def gaussian_blur(image, sigma):
    """Blur a grayscale or RGB image using the assignment's reflected padding."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty grayscale or RGB array")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError("color image must have one or three channels")

    kernel = gaussian_kernel(sigma)
    if image.ndim == 2:
        return convolve_separable(image, kernel, padding_mode="reflect")

    blurred_channels = [
        convolve_separable(image[:, :, channel], kernel, padding_mode="reflect")
        for channel in range(image.shape[2])
    ]
    return np.stack(blurred_channels, axis=2)


def make_hybrid(first_image, second_image, sigma_low=4.0, sigma_high=4.0):
    """Combine a low-pass first image with the high-pass of the second image."""
    first_image = np.asarray(first_image, dtype=np.float64)
    second_image = np.asarray(second_image, dtype=np.float64)
    if first_image.shape != second_image.shape:
        raise ValueError("images must have matching dimensions and channels")
    if first_image.ndim not in (2, 3) or first_image.size == 0:
        raise ValueError("images must be non-empty grayscale or RGB arrays")

    low_pass_first = gaussian_blur(first_image, sigma_low)
    blurred_second = gaussian_blur(second_image, sigma_high)
    high_pass_second = second_image - blurred_second
    hybrid = low_pass_first + high_pass_second
    return low_pass_first, high_pass_second, hybrid


def to_grayscale(image):
    """Convert a grayscale or RGB floating-point image to grayscale."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        )
    raise ValueError("image must be grayscale or have three RGB channels")


def fourier_log_magnitude(image):
    """Return the assignment's centered log-magnitude spectrum safely."""
    gray_image = to_grayscale(image)
    if gray_image.size == 0:
        raise ValueError("image must not be empty")
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))
    return np.log(np.maximum(magnitude, np.finfo(np.float64).tiny))