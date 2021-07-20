from typing import Union
from collections.abc import Sequence
from abc import abstractmethod
import math

import numpy as np
import jax
import jax.numpy as jnp

from .base import Transformation
from . import utils


class LazyCoordinates:
    _current_transform: jnp.ndarray = jnp.eye(3)
    _current_pullback: jnp.ndarray = jnp.eye(3)
    _dirty: bool = False
    _coordinates: jnp.ndarray
    _shape: Sequence[int]

    def __init__(self, shape: Sequence[int]):
        self._shape = shape
        H, W = self._shape
        self._coordinates = jnp.mgrid[0:H, 0:W] - jnp.array([H/2, W/2]).reshape(2, 1, 1)

    def get(self) -> jnp.ndarray:
        if self._dirty:
            coords = jnp.concatenate([self._coordinates, jnp.ones([1, *self._shape])])
            transform = self._current_pullback @ self._current_transform
            transformed = jnp.tensordot(transform, coords, axes=1)
            yx, z = jnp.split(transformed, [2])
            self._coordinates = yx / z
            self._dirty = False
            self._current_transform = jnp.eye(3)
            self._current_pullback = jnp.eye(3)
        return self._coordinates

    def crop(self, x0: float, y0: float, w: int, h: int):
        H, W = self._shape
        print(f'cropping rect at ({x0},{y0}) with w={w}, h={h}')

        y0 = int(math.floor(y0 + H/2))
        x0 = int(math.floor(x0 + W/2))

        self._coordinates = self._coordinates[:, y0:y0+h, x0:x0+w]
        self._shape = self._coordinates.shape[1:]

    def push_transform(self, M: jnp.ndarray):
        assert M.shape == (3, 3)
        self._current_transform = M @ self._current_transform
        self._dirty = True

    # def push_pullback(self, M: jnp.ndarray):
    #     assert M.shape == (3, 3)
    #     self._current_pullback = self._current_pullback @ M
    #     self._dirty = True

    def update_coordinates(self, coordinates: jnp.ndarray):
        assert not self._dirty
        self._coordinates = coordinates
        self._shape = coordinates.shape[1:]


class GeometricTransformation(Transformation):
    @abstractmethod
    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray) -> LazyCoordinates:
        return coordinates

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        coordinates = LazyCoordinates(image.shape[:2])
        self.transform_coordinates(coordinates, rng)

        H, W, _ = image.shape
        coordinates = coordinates.get() + jnp.array([H/2, W/2]).reshape(2, 1, 1)
        return utils.resample_image(image, coordinates, order=1)


class GeometricChain(GeometricTransformation):
    def __init__(self, *transforms: GeometricTransformation):
        self.transforms = transforms

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        for transform in reversed(self.transforms):
            rng, subkey = jax.random.split(rng)
            transform.transform_coordinates(coordinates, subkey)
        return coordinates


class HorizontalFlip(GeometricTransformation):
    def __init__(self, p: float = 0.5):
        self.probability = p

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        f = 1. - 2. * jax.random.bernoulli(rng, self.probability)
        transform = jnp.array([
            [1, 0, 0],
            [0, f, 0],
            [0, 0, 1]
        ])
        coordinates.push_transform(transform)


class VerticalFlip(GeometricTransformation):
    def __init__(self, p: float = 0.5):
        self.probability = p

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        f = 1. - 2. * jax.random.bernoulli(rng, self.probability)
        transform = jnp.array([
            [f, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        coordinates.push_transform(transform)


class Rotate(GeometricTransformation):
    def __init__(self,
            angle_range: Union[tuple[float, float], float]=(-30, 30),
            p: float = 1.0):
        if isinstance(angle_range, float):
            self.theta_min, self.theta_max = np.deg2rad([-angle_range, angle_range])
        else:
            self.theta_min, self.theta_max = np.deg2rad(angle_range)
        self.probability = p

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        do_apply = jax.random.bernoulli(rng, self.probability)
        theta = do_apply * jax.random.uniform(rng, minval=self.theta_min, maxval=self.theta_max)
        transform = jnp.array([
            [ jnp.cos(theta), jnp.sin(theta), 0],
            [-jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1]
        ])
        coordinates.push_transform(transform)


class CenterCrop(GeometricTransformation):
    def __init__(self, output_shape: Union[tuple[int, int], int]):
        if isinstance(output_shape, int):
            self.output_shape = (output_shape, output_shape)
        else:
            self.output_shape = output_shape

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H_, W_ = self.output_shape
        y0 = -H_ / 2
        x0 = -W_ / 2
        coordinates.crop(x0, y0, W_, H_)


class Translate(GeometricTransformation):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        transform = jnp.array([
            [1, 0, -self.dy],
            [0, 1, -self.dx],
            [0, 0,        1]
        ])
        coordinates.push_transform(transform)


class Crop(GeometricTransformation):
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.w  = w
        self.h  = h

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates._shape
        coordinates.crop(-self.w / 2, -self.h / 2, self.w, self.h)

        center_x = self.x0 + self.w / 2 - W / 2
        center_y = self.y0 + self.h / 2 - H / 2

        # self.dx/dy is in (0,0) -- (H,W) reference frame
        # => push it to (-H/2, -W/2) -- (H/2, W/2) reference frame

        # Forward transform: Translate by (dx, dy)
        transform = jnp.array([
            [1, 0,  center_y],
            [0, 1,  center_x],
            [0, 0,          1]
        ])
        coordinates.push_transform(transform)


class Warp(GeometricTransformation):
    def __init__(self, strength: int=5, coarseness: int=32):
        self.strength = strength
        self.coarseness = coarseness

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        coords = coordinates.get()
        _, H, W = coords.shape
        H_, W_ = H // self.coarseness, W // self.coarseness
        coordshift_coarse = self.strength * jax.random.normal(rng, [2, H_, W_]) 
        coordshift = jax.image.resize(coordshift_coarse, coords.shape, method='bicubic')
        coordinates.update_coordinates(coords + coordshift)
