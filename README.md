# Assignment 1 Starter Pack

Everything you need to start *Assignment 1: Fine Detail and the Big Picture*.

## What's inside

```
images/
  part1_filters/
    cameraman.png          an image for 1.2 and 1.3 (or use one of your own)
    facade.png             the photo behind the handout's figures in 1.2, 1.3 and 2.3
  part2_frequencies/
    hybrid_man.jpg         the man-and-cat pair for 2.1 (not aligned yet:
    hybrid_cat.jpg           use code/align_images.py)
    rings.png              for the aliasing experiment in 2.2 (view it at 100% zoom)
    brick.png              a real repeating texture, to compare with rings.png in 2.2
    apple.jpg, orange.jpg  for the oraple in 2.4
    hand.jpg, eye.jpg      a pair to practice irregular masks on in 2.4
  part3_seams/
    rocket.jpg, home.jpg, starry_night.jpg
                           the three images to narrow in 3.2
    broadway_tower.png     the photo behind the handout's figures in Part 3 and the Bonus
code/
  utils.py                 load_image, save_image, show
  align_images.py          align two images using two matching points
requirements.txt
CREDITS.md
```

`facade.png` and `broadway_tower.png` let you compare your results with the handout's
figures. The tower is at the size those figures used (714 × 484), and the walker on its
left is the object the Bonus figure removes. Like the hand and eye, these two are for
practice: they do not count as images of your own in 3.2 or the Bonus.

## Setup

```
pip install -r requirements.txt
```

All helpers work with float images in [0, 1]:

```python
from utils import load_image, show
im = load_image("../images/part1_filters/cameraman.png", gray=True)
show(im, titles=["cameraman"])
```

## Aligning images for hybrids and blends

Pick the same two points in both images (for faces, the centers of the eyes),
then warp the second image onto the first:

```python
from utils import load_image
from align_images import align_images, show_with_grid

im1 = load_image("../images/part2_frequencies/hybrid_man.jpg")
im2 = load_image("../images/part2_frequencies/hybrid_cat.jpg")

show_with_grid(im1)   # read point coordinates off the grid
show_with_grid(im2)

im2_aligned = align_images(im1, im2,
                           pts1=[(299, 341), (443, 332)],  # the man's eyes, as (x, y)
                           pts2=[(605, 290), (760, 372)])  # the cat's eyes, as (x, y)
```

To practice for 2.4, the same call places the eye on the palm, with `im1` the hand
and `im2` the eye: `pts1=[(170, 400), (390, 410)]` across the palm, and
`pts2=[(165, 255), (470, 265)]` at the corners of the eye.

Running Python locally (not in Colab)? You can also click the points with
`pick_points(im)`.

## A reminder

The helpers here are only for loading, showing and aligning images. Filtering,
pyramids, blending and seam carving are yours to write.
