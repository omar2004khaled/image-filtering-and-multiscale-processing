"""Small helpers for loading, saving and showing images.

All images are handled as float32 NumPy arrays with values in [0, 1].
Color images have shape (H, W, 3) in RGB order; grayscale images have shape (H, W).
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(path, gray=False):
    """Read an image file as a float32 array in [0, 1] (RGB, or grayscale if gray=True)."""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    im = cv2.imread(str(path), flag)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if not gray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im.astype(np.float32) / 255.0


def save_image(path, im):
    """Save a float image in [0, 1] (values outside are clipped)."""
    out = (np.clip(im, 0, 1) * 255).round().astype(np.uint8)
    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), out)


def show(*images, titles=None, cmap="gray", size=4):
    """Show one or more images side by side.

    Grayscale images are shown with a fixed [0, 1] range, so a filtered image that
    has negative values (e.g. a derivative) should be shifted or rescaled first.
    """
    titles = titles or [""] * len(images)
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    axes = np.atleast_1d(axes)
    for ax, im, title in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
