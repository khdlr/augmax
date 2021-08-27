from abc import abstractmethod
from typing import Union

import math
import jax
import jax.numpy as jnp
from einops import rearrange

from .base import Transformation


class ImageLevelTransformation(Transformation):
    pass


class GridShuffle(ImageLevelTransformation):
    """Divides the image into grid cells and shuffles them randomly.

    Args:
        grid_size (int, int): Tuple of `(gridcells_x, gridcells_y)` that specifies into how many
            cells the image is to be divided along each axis.
            If only a single number is given, that value will be used along both axes.
            Currently requires that each image dimension is a multiple of the corresponding value.
        p (float): Probability of applying the transformation
    """
    def __init__(self, grid_size: Union[tuple[int, int], int] = (4, 4), p: float = 0.5):
        if hasattr(grid_size, '__iter__'):
            self.grid_size = tuple(grid_size)
        else:
            self.grid_size = (self.grid_size, self.grid_size)
        self.grid_size = grid_size
        self.probability = p

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        raw_image = image

        H, W, C = image.shape
        gx, gy = self.grid_size

        if H % self.grid_size[0] != 0:
            raise ValueError(f"Image height ({H}) needs to be a multiple of gridcells_y ({gy})")
        if W % self.grid_size[1] != 0:
            raise ValueError(f"Image width ({W}) needs to be a multiple of gridcells_x ({gx})")

        image = rearrange(image, '(gy h) (gx w) c -> (gy gx) h w c', gx=gx, gy=gy)
        image = jax.random.permutation(rng, image)
        image = rearrange(image, '(gy gx) h w c -> (gy h) (gx w) c', gx=gx, gy=gy)

        do_apply = jax.random.bernoulli(rng, self.probability)
        
        return jnp.where(do_apply, image, raw_image)


class _ConvolutionalBlur(ImageLevelTransformation):
    @abstractmethod
    def __init__(self, p: float = 0.5):
        self.probability = p
        self.kernel = None
        self.kernelsize = -1

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        do_apply = jax.random.bernoulli(rng, self.probability)
        p0 = self.kernelsize // 2
        p1 = self.kernelsize - p0 - 1
        image_padded = jnp.pad(image, [(p0, p1), (p0, p1), (0, 0)], mode='edge')
        image_padded = rearrange(image_padded, 'h w (c c2) -> c c2 h w', c2=1)
        convolved = jax.lax.conv(image_padded, self.kernel, [1, 1], 'valid')
        convolved = rearrange(convolved, 'c c2 h w -> h w (c c2)', c2=1)
        image = jnp.where(do_apply, convolved, image)
        return image


class Blur(_ConvolutionalBlur):
    def __init__(self, size: int = 5, p: float = 0.5):
        super().__init__(p)
        self.kernel = jnp.ones([1, 1, size, size])
        self.kernel = self.kernel / self.kernel.sum()
        self.kernelsize = size


class GaussianBlur(_ConvolutionalBlur):
    def __init__(self, sigma: int = 3, p: float = 0.5):
        super().__init__(p)
        N = int(math.ceil(2 * sigma))
        rng = jnp.linspace(-2.0, 2.0, N)
        x = rng.reshape(1, -1)
        y = rng.reshape(-1, 1)

        self.kernel = jnp.exp((-0.5/sigma) * (x*x + y*y))
        self.kernel = self.kernel / self.kernel.sum()
        self.kernel = self.kernel.reshape(1, 1, N, N)
        self.kernelsize = N
