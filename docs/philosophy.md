# The Augmax Philosophy

Existing Data Augmentation frameworks like [imgaug](https://imgaug.readthedocs.io/en/latest/) or [albumentations](https://albumentations.ai/)
work by applying the random transformations specified by the user step-by-step.
In contrast to this, augmax tries to fuse as many operations as possible,
and in doing so, improve the performance and memory footprint of the augmentation pipeline.

To understand why this matters, let us start with a simple example. Here is an example data augmentation pipeline:
1. Random horizontal flip
2. Rotate by a random angle in $[-15^\circ, 15^\circ]$
3. Crop the image to a size of $128 \times 128$ pixels
4. Randomly adjust the contrast of the image
5. Randomly adjust the brightness of the image
6. Randomly apply grayscale to the image

In albumentations, we might write this chain of transformations like this:

```python
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(15),
    A.RandomCrop(width=256, height=256),
    A.RandomBrightnessContrast(),
])
```

Generally, `albumentation.Compose` acts more or less like a for-loop that applies the specified operations one after another to the given input image. So an example run including intermediate results might look like this:

![augmentation_chain](imgs/augmentation_chain.png)

That is a lot of intermediate data which we don't really care about.
One design goal of augmax is to
reduce the number of operations that are carried out on your data.
This is motivated by the assumption that less operations will require less memory,
compute faster, and lead to more accurate results as less interpolation is done in-between.
Note how we used `RandomBrightnessContrast` instead of `RandomBrightness` and `RandomContrast`? Augmax tries to do this sort of operation fusion for you automatically, without the user needing to worry about the possible combinations of operations that can be fused.

## Dynamic Fusion of Augmentations

Dynamically fusing of augmentations into more efficient operations is achieved by tracing the specified operations and automatically generating quicker, equivalent operations. Augmax categorizes its augmentation operations into 3 groups:

1. **Geometric Transformations** are transformations that change the pixel-layout of the image, like crops, flips or rotations.
2. **Colorspace Transformations** operate on the color values of each pixel separately, like contrast or hue adjustments. 
3. **Imagelevel Transformations** is an umbrella term for any transformation that doesn't fit into either one of the previous categories, for example blurring operations.

Consecutive geometric and colorspace transformations can be fused into a single augmax operation, which will speed up performance especially for long augmentation pipelines.

### Fusing Geometric Transformations

> TLDR: Augmax does not transform pixels, but coordinates. Coordinate transformations can be fused more efficiently.

When we think about geometric transformations, we usually think about where certain parts of the image will end up after the transformation. For example when we translate an image by an offset of $(d_x, d_y)$, the pixel at location $(x, y)$ will end up at $(x+d_x, y+d_y)$. When we formulate geometric image transformations like this, we can easily chain them by applying them in order. 

![forward_augmentation](imgs/forward_augmentation.png)

To better optimize geometric transformations,
augmax takes a different approach.
Instead of following the forward path of an input pixel through the transformation chain,
it does the whole process backwards.
For each output pixel,
it determines where the original location of that pixel is in the input image.
Going back to the translation example,
the output pixel $(x', y')$ originates from the input pixel $(x' - d_x, y' - d_y)$.
Mathematically speaking, we are calculating the *preimage* of each pixel's location under the augmentation chain.
Using these transformed coordinates,
a simple texture sampling step is then used to sample the output pixels directly from the input image
via [`jax.scipy.ndimage.map_coordinates`](jax.scipy.ndimage.map_coordinates).
The important difference to note with this approach is that augmax will
need to internally reverse the order of operations,
as we start from output coordinates and work our way back to the pixels of the input image.

![backward_augmentation](imgs/backward_augmentation.png)

### Fusing Colorspace Transformations

** TODO **
