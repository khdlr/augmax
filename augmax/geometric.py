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
    _dirty: bool = False
    _coordinates: jnp.ndarray
    shape: tuple[int, int]

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape

    def get(self) -> jnp.ndarray:
        H, W = self.shape
        if not hasattr(self, '_coordinates'):
            self._coordinates = jnp.mgrid[0:H, 0:W] - jnp.array([H/2, W/2]).reshape(2, 1, 1)
        if self._dirty:
            coords = jnp.concatenate([self._coordinates, jnp.ones([1, *self.shape])])
            transformed = jnp.tensordot(self._current_transform, coords, axes=1)
            yx, z = jnp.split(transformed, [2])
            self._coordinates = yx / z
            self._dirty = False
            self._current_transform = jnp.eye(3)
        return self._coordinates

    def crop(self, h: int, w: int):
        if hasattr(self, '_coordinates'):
            H, W = self.shape
            y0 = int(math.floor((H-h)/2))
            x0 = int(math.floor((W-w)/2))
            self._coordinates = self._coordinates[:, y0:y0+h, x0:x0+w]
            print('y0:', y0, 'x0:', x0)
        # self.shape = self._coordinates.shape[1:]

        self.shape = (h, w)

    def push_transform(self, M: jnp.ndarray):
        assert M.shape == (3, 3)
        self._current_transform = M @ self._current_transform
        self._dirty = True

    def update_coordinates(self, coordinates: jnp.ndarray):
        assert not self._dirty
        self._coordinates = coordinates
        _, H, W = coordinates.shape
        self.shape = (H, W)


class GeometricTransformation(Transformation):
    @abstractmethod
    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray) -> LazyCoordinates:
        return coordinates

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        H, W, _ = image.shape
        coordinates = LazyCoordinates((H, W))
        self.transform_coordinates(coordinates, rng)

        H, W, _ = image.shape
        coordinates = coordinates.get() + jnp.array([H/2, W/2]).reshape(2, 1, 1)
        return utils.resample_image(image, coordinates, order=1)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return input_shape


class GeometricChain(GeometricTransformation):
    def __init__(self, *transforms: GeometricTransformation):
        self.transforms = transforms

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        shape_chain = [coordinates.shape]
        for transform in self.transforms[:-1]:
            shape_chain.append(transform.output_shape(shape_chain[-1]))

        subkeys = jax.random.split(rng, len(self.transforms))
        for transform, input_shape, subkey in zip(reversed(self.transforms), reversed(shape_chain), subkeys):
            print('input_shape:', input_shape)
            coordinates.shape = input_shape
            transform.transform_coordinates(coordinates, subkey)

        return coordinates

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        shape = input_shape
        for transform in self.transforms:
            shape = transform.output_shape(shape)
        return shape


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


class Rotate90(GeometricTransformation):
    def __init__(self):
        pass

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        params = jax.random.bernoulli(rng, 0.5, [2])
        flip = 1. - 2. * params[0] 
        rot = params[1]

        transform = jnp.array([
            [flip * rot,       flip * (1.-rot), 0],
            [flip * (-1.+rot), flip * rot,      0],
            [0,                0,               1]
        ])
        coordinates.push_transform(transform)


class Rotate(GeometricTransformation):
    def __init__(self,
            angle_range: Union[tuple[float, float], float]=(-30, 30),
            p: float = 1.0):
        if hasattr(angle_range, '__iter__'):
            self.theta_min, self.theta_max = np.deg2rad(angle_range)
        else:
            self.theta_min, self.theta_max = np.deg2rad([-angle_range, angle_range])
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
    width: int
    height: int

    def __init__(self, width: int, height: int = None):
        self.width = width
        self.height = width if height is None else height

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        coordinates.crop(self.height, self.width)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)


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
        self.width = w
        self.height = h

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.shape
        coordinates.crop(self.height, self.width)

        center_x = self.x0 + self.width / 2 - W / 2
        center_y = self.y0 + self.height / 2 - H / 2

        # self.dx/dy is in (0,0) -- (H,W) reference frame
        # => push it to (-H/2, -W/2) -- (H/2, W/2) reference frame

        # Forward transform: Translate by (dx, dy)
        transform = jnp.array([
            [1, 0,  center_y],
            [0, 1,  center_x],
            [0, 0,          1]
        ])
        coordinates.push_transform(transform)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)


class RandomCrop(GeometricTransformation):
    width: int
    height: int

    def __init__(self, width: int, height: int = None):
        self.width = width
        self.height = width if height is None else height

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.shape

        limit_y = (H - self.height) / 2
        limit_x = (W - self.width) / 2

        center_y, center_x = jax.random.uniform(rng, [2],
                minval=jnp.array([-limit_y, -limit_x]),
                maxval=jnp.array([limit_y, limit_x]))

        coordinates.crop(self.height, self.width)

        transform = jnp.array([
            [1, 0,  center_y],
            [0, 1,  center_x],
            [0, 0,          1]
        ])
        coordinates.push_transform(transform)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)


class RandomSizedCrop(GeometricTransformation):
    width: int
    height: int
    min_logzoom: float
    max_logzoom: float

    def __init__(self,
            width: int, height: int = None, zoom_range: tuple[float, float] = (0.5, 2.0),
            prevent_underzoom: bool = False):
        self.width = width
        self.height = width if height is None else height
        self.min_logzoom = math.log(zoom_range[0])
        self.max_logzoom = math.log(zoom_range[1])
        self.prevent_underzoom = prevent_underzoom

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.shape
        key1, key2 = jax.random.split(rng)

        if self.prevent_underzoom:
            min_logzoom = max(self.min_logzoom, math.log(self.height / H), math.log(self.width / W))
            max_logzoom = max(self.max_logzoom, min_logzoom)
        else:
            min_logzoom = self.min_logzoom
            max_logzoom = self.max_logzoom

        log_zoom  = jax.random.uniform(key1, minval=min_logzoom, maxval=max_logzoom)
        zoom = jnp.exp(log_zoom)

        limit_y = ((H/zoom) - self.height) / 2
        limit_x = ((W/zoom) - self.width) / 2

        center_y, center_x, = jax.random.uniform(key2, [2],
                minval=jnp.array([-limit_y, -limit_x]),
                maxval=jnp.array([limit_y, limit_x]))

        coordinates.crop(self.height, self.width)

        transform = jnp.array([
            [1/zoom,      0,  center_y],
            [     0, 1/zoom,  center_x],
            [     0,      0,         1]
        ])
        coordinates.push_transform(transform)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)


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
