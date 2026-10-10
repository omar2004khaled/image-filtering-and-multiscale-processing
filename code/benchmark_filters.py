"""Benchmark Part 1.1 convolution methods on generated Gaussian kernels."""
import time

import numpy as np

from filters import convolve_loop, convolve_separable, convolve_vectorized


def gaussian_kernel(size, sigma=None):
    """Build a normalized square Gaussian kernel of an odd size."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if sigma is None:
        sigma = size / 6.0

    coordinates = np.arange(size, dtype=np.float64) - size // 2
    one_dimensional = np.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    one_dimensional /= one_dimensional.sum()
    return np.outer(one_dimensional, one_dimensional)


def main():
    random = np.random.default_rng(2026)
    image = random.random((256, 256))
    methods = (
        ("Loop", convolve_loop),
        ("Vectorized", convolve_vectorized),
        ("Separable", convolve_separable),
    )

    print("256 x 256 image; one run per method and kernel")
    print(f"{'Kernel':>8} | {'Loop (ms)':>12} | {'Vectorized (ms)':>16} | {'Separable (ms)':>15}")
    print("-" * 62)
    for size in (3, 5, 7, 15, 31):
        kernel = gaussian_kernel(size)
        elapsed_ms = []
        for _, method in methods:
            start = time.perf_counter()
            result = method(image, kernel)
            elapsed_ms.append((time.perf_counter() - start) * 1000)
            assert result.shape == image.shape

        print(
            f"{size:>2} x {size:<2} | {elapsed_ms[0]:>12.2f} | "
            f"{elapsed_ms[1]:>16.2f} | {elapsed_ms[2]:>15.2f}"
        )


if __name__ == "__main__":
    main()