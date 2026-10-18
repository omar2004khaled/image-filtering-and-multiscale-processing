# Assignment 1: Results and Discussion

## Part 1: Filters and Edges

### 1.1 Convolution from Scratch

All three implementations return same-sized results using zero padding, as required. The loop, vectorized, and separable implementations were compared on a generated 256 x 256 image with Gaussian kernels. The recorded times are in milliseconds:

| Kernel | Loop | Vectorized | Separable |
| --- | ---: | ---: | ---: |
| 3 x 3 | 354.42 | 2.32 | 1.71 |
| 5 x 5 | 976.75 | 3.31 | 1.72 |
| 7 x 7 | 3483.22 | 21.09 | 6.36 |
| 15 x 15 | 29288.91 | 65.54 | 6.88 |
| 31 x 31 | 97831.67 | 55.06 | 3.72 |

The loop implementation performs the work for every pixel and every filter entry in Python, so its runtime increases rapidly with kernel area. The vectorized implementation still performs the kernel operations at every pixel, but moves those operations into NumPy rather than Python loops; this accounts for its large speedup. A rank-one Gaussian can be factored into a column and a row, so the separable implementation performs two one-dimensional convolutions instead of one full two-dimensional convolution. Its work therefore grows roughly with kernel width rather than kernel area, making it especially effective for larger Gaussian kernels.

These are one-run measurements from the current machine, not stable hardware-independent benchmarks. The vectorized timings are not perfectly monotonic because short single runs are affected by startup, cache, and system-load variation. The trend, especially the loop cost and the separable advantage for large kernels, is more meaningful than small differences between individual measurements.

Benchmark environment: Python 3.13.5, NumPy 2.3.1, SciPy 1.16.0. The benchmark uses one random 256 x 256 image and one timing per method and kernel size.

### 1.2 Finite Difference Operator

**Image:** `images/part1_filters/facade.png`, grayscale, 736 x 670 pixels, with intensities in [0, 1].

**Threshold:** 0.20. The gradient magnitude is measured on the normalized image, so this threshold keeps substantial architectural transitions while excluding many weaker intensity changes. On this image it marks 13.1% of pixels as edges. Nearby values illustrate the tradeoff: threshold 0.15 marks 18.6%, while 0.25 marks 9.4%. The selected value gives a useful balance of visible facade detail and a readable binary map; it is an image-dependent choice, not a universal threshold.

**Lines along the image frame:** The convolution treats pixels outside the image as zero. At the frame, the derivative compares real image intensities with those zeros, creating an artificial intensity jump. That jump produces strong responses in the derivative and gradient-magnitude maps, which thresholding turns into lines around the frame. These are padding artifacts, not edges of objects in the scene. They are expected with the zero padding required for Part 1.1; Part 2 onward explicitly changes the padding rule to reflection.

### 1.3 Derivative of Gaussian

**Image:** `images/part1_filters/facade.png`, the same grayscale image as Part 1.2.

**Parameters:** Gaussian sigma = 2.0; edge threshold = 0.08. The handout specifies sigma 2.0. At that sigma, threshold 0.08 marks 11.7% of pixels as edges on this image. Nearby thresholds show the sensitivity: 0.06 marks 19.1%, while 0.10 marks 7.3%. The selected threshold gives a contour density close to the handout's example while suppressing many weak responses. It should be reconsidered if the input image or its intensity scaling changes.

Blurring before differentiation suppresses small-scale intensity changes, so the DoG edge map is substantially less noisy than the finite-difference map. The same zero-padding boundary rule still creates a response near the frame; smoothing spreads that boundary effect over the DoG filter's support. The Gaussian kernel is normalized only for display in the figure so it is visible at the plotting scale; the filter values used for processing are unchanged.

## Part 2: Frequencies, Pyramids, and Blending

### 2.1 Hybrid Images

_To complete after the final pairs and parameters are selected._

For each of the three pairs, record the input filenames, alignment points, low-pass and high-pass sigma values, and what is recognizable close up and from farther away. For the favorite pair, include the five spatial images and their corresponding Fourier log-magnitude images, then explain how the spectra relate to the hybrid's appearance.

### 2.2 Aliasing

_To complete after selecting the final sigma._

Record the smallest tested Gaussian sigma that removes visible false circles, the input and output dimensions, and the visual difference between direct sampling and blur-then-sample. Explain why filtering before sampling prevents high frequencies from folding into false lower-frequency patterns, and connect this to Gaussian-pyramid reduction.

### 2.3 Gaussian and Laplacian Pyramids

_To complete after choosing the final non-power-of-two image._

Record the image dimensions, pyramid levels, total Gaussian-pyramid pixel count and ratio to the original, and reconstruction error. Explain why subtracting the expanded next Gaussian level makes reconstruction exact, why the lecture's blur-residual version is not generally exact, and how odd dimensions are handled.

### 2.4 Multiresolution Blending

_To complete after the three blends are finalized._

For each pair, record the source images, mask construction, and number of pyramid levels. Show the two inputs, mask, hard cut, and blend. For the favorite, include the per-level weighted Laplacian components and their sum. Explain how blending frequency bands over different spatial widths hides the seam.

## Part 3: Seam Carving

### 3.1 Finding and Removing Seams

_To complete after selecting the report image and seam count._

Record the image, energy map, requested width reduction, and seams drawn on the original. Explain the cumulative-energy dynamic program and backtracking procedure, and why enumerating all possible seams is impractical.

### 3.2 Results and Comparisons

The person-outdoors comparison shows the difference between uniform scaling and content-aware reduction: normal resizing narrows the person along with the landscape, while seam carving removes more of the surrounding sky and field and leaves the upright figure closer to its original shape. Center cropping also preserves the figure's shape, but discards content at both sides.

The group photo is a failure case for both energy choices. Finite-difference and DoG seams cross the people, including clothing and body regions, rather than removing only background. The tight framing leaves few expendable background columns, and gradient magnitude does not represent the semantic importance of a person; the carved image therefore crowds and locally distorts the group. The DoG energy map is smoother than the finite-difference map, but this smoothing does not prevent seams from crossing the subjects.

The figures and measured dimensions for all six comparisons are saved under `outputs/part3_2/`. The group-photo energy figure overlays both seam sets, and `summary.txt` reports their removed-pixel mask Jaccard overlap rather than comparing seam order.

## Bonus: Object Removal

_Optional section. Add the selected object and mask, initial and inserted seams, restored result, and any limitations observed._
