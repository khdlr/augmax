from abc import abstractmethod
from typing import Union

import jax
import jax.numpy as jnp
from einops import rearrange

from .base import Transformation


class ImageLevelTransformation(Transformation):
    pass


class GridShuffle(Transformation):
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
