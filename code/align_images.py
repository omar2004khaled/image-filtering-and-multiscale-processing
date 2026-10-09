"""Align two images, for the hybrid images of 2.1 and the blends of 2.4.

Pick the same two points in both images (for faces, the centers of the two eyes
work well). The second image is then rotated, scaled and shifted so its points
land on the first image's points, and returned at the same size as the first.

Two ways to get the points:

  1. Interactive (running Python locally, not in Colab):
         pts1 = pick_points(im1)
         pts2 = pick_points(im2)

  2. By hand (works anywhere, including Colab and Jupyter):
         show_with_grid(im1)   # read the coordinates off the grid
         pts1 = [(x1, y1), (x2, y2)]

Then:
     im2_aligned = align_images(im1, im2, pts1, pts2)

Example with the starter images (eye centers, in (x, y) pixel coordinates):
     im1 = load_image("../images/part2_frequencies/hybrid_man.jpg")
     im2 = load_image("../images/part2_frequencies/hybrid_cat.jpg")
     im2_aligned = align_images(im1, im2, [(299, 341), (443, 332)], [(605, 290), (760, 372)])
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def similarity_from_two_points(src, dst):
    """Return the 2x3 matrix of the rotation + uniform scale + shift that maps src -> dst.

    src and dst each hold two (x, y) points. Points are treated as complex numbers
    z = x + iy, so the transform is z' = a*z + b.
    """
    s = np.array([complex(x, y) for x, y in src])
    d = np.array([complex(x, y) for x, y in dst])
    if abs(s[1] - s[0]) < 1e-9:
        raise ValueError("The two source points must be different.")
    a = (d[1] - d[0]) / (s[1] - s[0])
    b = d[0] - a * s[0]
    return np.array([[a.real, -a.imag, b.real],
                     [a.imag, a.real, b.imag]], dtype=np.float64)


def align_images(im1, im2, pts1, pts2):
    """Warp im2 so that pts2 lands on pts1. Returns an image the same size as im1."""
    M = similarity_from_two_points(pts2, pts1)
    h, w = im1.shape[:2]
    return cv2.warpAffine(im2, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def pick_points(im, n=2):
    """Click n points on the image (local Python only). Returns a list of (x, y)."""
    plt.figure(figsize=(8, 8))
    plt.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None)
    plt.title(f"Click {n} points, in the same order for both images")
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return [(float(x), float(y)) for x, y in pts]


def show_with_grid(im, step=20):
    """Show the image with a pixel grid, to read point coordinates by eye."""
    h, w = im.shape[:2]
    plt.figure(figsize=(9, 9 * h / w))
    plt.imshow(np.clip(im, 0, 1), cmap="gray" if im.ndim == 2 else None)
    plt.xticks(range(0, w, step), fontsize=6, rotation=90)
    plt.yticks(range(0, h, step), fontsize=6)
    plt.grid(True, alpha=0.4)
    plt.show()
