"""Correctness tests for the Part 1.1 convolution implementations."""
import unittest

import numpy as np
from scipy.signal import convolve2d

from filters import convolve_loop, convolve_separable, convolve_vectorized


class ConvolutionTests(unittest.TestCase):
    def test_random_odd_kernels_match_scipy(self):
        random = np.random.default_rng(2026)
        methods = (convolve_loop, convolve_vectorized, convolve_separable)
        image_shapes = ((3, 4), (7, 6))
        kernel_shapes = ((1, 1), (1, 5), (5, 1), (3, 5), (5, 3), (3, 3))

        for image_shape in image_shapes:
            image = random.normal(size=image_shape)
            for kernel_shape in kernel_shapes:
                kernels = [random.normal(size=kernel_shape)]
                column = random.normal(size=kernel_shape[0])
                row = random.normal(size=kernel_shape[1])
                kernels.append(np.outer(column, row))

                for kernel in kernels:
                    expected = convolve2d(
                        image,
                        kernel,
                        mode="same",
                        boundary="fill",
                        fillvalue=0,
                    )
                    for method in methods:
                        with self.subTest(
                            method=method.__name__,
                            image_shape=image_shape,
                            kernel_shape=kernel_shape,
                        ):
                            actual = method(image, kernel)
                            self.assertEqual(actual.shape, image.shape)
                            np.testing.assert_allclose(
                                actual, expected, rtol=1e-10, atol=1e-10
                            )

    def test_inputs_are_not_modified(self):
        image = np.arange(20, dtype=np.float64).reshape(4, 5)
        kernel = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [2.0, 1.0, 0.0]])
        original_image = image.copy()
        original_kernel = kernel.copy()

        for method in (convolve_loop, convolve_vectorized, convolve_separable):
            method(image, kernel)

        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(kernel, original_kernel)

    def test_even_sized_kernel_is_rejected(self):
        image = np.ones((3, 4))
        even_kernel = np.ones((2, 3))
        for method in (convolve_loop, convolve_vectorized, convolve_separable):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method(image, even_kernel)


if __name__ == "__main__":
    unittest.main()