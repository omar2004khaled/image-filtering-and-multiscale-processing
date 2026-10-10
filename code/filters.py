"""Part 1.1: 2D convolution implementations for grayscale images."""
import numpy as np


def _as_float_inputs(image, kernel):
    """Validate grayscale shapes and return floating-point arrays."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays")
    if image.size == 0 or kernel.size == 0:
        raise ValueError("image and kernel must not be empty")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("kernel dimensions must be odd")
    return image, kernel


def convolve_loop(image, kernel):
    """Convolve a grayscale image with loops over pixels and kernel entries."""
    image, kernel = _as_float_inputs(image, kernel)
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_y, pad_x = kernel_height // 2, kernel_width // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")

    # Convolution flips the kernel in both dimensions before applying it.
    flipped_kernel = np.flip(kernel, axis=(0, 1))
    output = np.zeros((height, width), dtype=np.float64)
    for image_y in range(height):
        for image_x in range(width):
            for kernel_y in range(kernel_height):
                for kernel_x in range(kernel_width):
                    output[image_y, image_x] += (
                        padded[image_y + kernel_y, image_x + kernel_x]
                        * flipped_kernel[kernel_y, kernel_x]
                    )
    return output


def convolve_vectorized(image, kernel, padding_mode="constant"):
    """Convolve using NumPy operations without loops over image pixels."""
    image, kernel = _as_float_inputs(image, kernel)
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode=padding_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    flipped_kernel = np.flip(kernel, axis=(0, 1))
    return np.einsum("ijkl,kl->ij", windows, flipped_kernel)


def convolve_separable(
    image, kernel, rank_tolerance=1e-10, padding_mode="constant"
):
    """Use two 1D convolutions for rank-1 kernels; otherwise use vectorized."""
    image, kernel = _as_float_inputs(image, kernel)
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be non-negative")

    left, singular_values, right = np.linalg.svd(kernel, full_matrices=False)
    is_rank_one = (
        singular_values.size == 1
        or singular_values[1] <= rank_tolerance * singular_values[0]
    )
    if not is_rank_one:
        return convolve_vectorized(image, kernel, padding_mode=padding_mode)

    # SVD writes a rank-1 kernel as column @ row; convolve each factor in turn.
    column = left[:, 0] * singular_values[0]
    row = right[0, :]
    result = convolve_vectorized(
        image, column[:, np.newaxis], padding_mode=padding_mode
    )
    return convolve_vectorized(
        result, row[np.newaxis, :], padding_mode=padding_mode
    )


def finite_difference_edges(image, threshold=0.2):
    """Return reflected finite differences, gradient magnitude, and edge mask."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative value")

    derivative_x = np.array([[1.0, 0.0, -1.0]])
    derivative_y = derivative_x.T
    dx = convolve_separable(image, derivative_x, padding_mode="reflect")
    dy = convolve_separable(image, derivative_y, padding_mode="reflect")
    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    edges = gradient_magnitude >= threshold
    return dx, dy, gradient_magnitude, edges