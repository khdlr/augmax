from typing import Union
from abc import abstractmethod
import math

import numpy as np
import jax
import jax.numpy as jnp

from .base import Transformation, Chain
from . import utils


def _apply_perspective(xy: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    _, H, W = xy.shape
    xyz = jnp.concatenate([xy, jnp.ones([1, H, W])])
    xyz = jnp.tensordot(M, xyz, axes=1)
    yx, z = jnp.split(xyz, [2])
    return yx / z


class LazyCoordinates:
    _current_transform: jnp.ndarray = jnp.eye(3)
    _offsets: Union[jnp.ndarray, None] = None
    input_shape: tuple[int, int]
    final_shape: tuple[int, int]

    def __init__(self, shape: tuple[int, int]):
        self.input_shape = shape
        self.final_shape = shape

    def get(self) -> jnp.ndarray:
        H, W = self.final_shape
        coordinates = jnp.mgrid[0:H, 0:W] - jnp.array([H/2-0.5, W/2-0.5]).reshape(2, 1, 1)
        coordinates = _apply_perspective(coordinates, self._current_transform)

        if self._offsets is not None:
            coordinates = coordinates + self._offsets

        return coordinates

    def push_transform(self, M: jnp.ndarray):
        assert M.shape == (3, 3)
        self._current_transform = M @ self._current_transform
        self._dirty = True

    def apply_pixelwise_offsets(self, offsets):
        assert offsets.shape[1:] == self.final_shape
        if self._offsets == None:
            self._offsets = offsets
        else:
            self._offsets = self._offsets + offsets


class GeometricTransformation(Transformation):
    @abstractmethod
    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray) -> LazyCoordinates:
        return coordinates

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        H, W, _ = image.shape
        coordinates = LazyCoordinates((H, W))
        coordinates.final_shape = self.output_shape((H, W))

        self.transform_coordinates(coordinates, rng)

        coordinates = coordinates.get() + jnp.array([H/2-0.5, W/2-0.5]).reshape(2, 1, 1)
        return utils.resample_image(image, coordinates, order=1)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return input_shape


class GeometricChain(GeometricTransformation, Chain):
    def __init__(self, *transforms: GeometricTransformation):
        super().__init__()
        for transform in transforms:
            # TODO: Re-enable this, autoreload breaks this...
            pass
            # assert isinstance(transform, GeometricTransformation), f"{transform} is not a GeometricTransformation!"
        self.transforms = transforms

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        shape_chain = [coordinates.input_shape]
        for transform in self.transforms[:-1]:
            shape_chain.append(transform.output_shape(shape_chain[-1]))

        subkeys = jax.random.split(rng, len(self.transforms))
        for transform, input_shape, subkey in zip(reversed(self.transforms), reversed(shape_chain), subkeys):
            coordinates.input_shape = input_shape
            transform.transform_coordinates(coordinates, subkey)

        return coordinates

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        shape = input_shape
        for transform in self.transforms:
            shape = transform.output_shape(shape)
        return shape


class HorizontalFlip(GeometricTransformation):
    """Randomly flips an image horizontally.

    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
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
    """Randomly flips an image vertically.

    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
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
    """Randomly rotates the image by a multiple of 90 degrees.
    """
    def __init__(self):
        super().__init__()

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
    """Rotates the image by a random arbitrary angle.

    Args:
        angle_range (float, float): Tuple of `(min_angle, max_angle)` to sample from.
            If only a single number is given, angles will be sampled from `(-angle_range, angle_range)`.
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            angle_range: Union[tuple[float, float], float]=(-30, 30),
            p: float = 1.0):
        super().__init__()
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


class Translate(GeometricTransformation):
    def __init__(self, dx, dy):
        super().__init__()
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
    """Crop the image at the specified x0 and y0 with given width and height

    Args:
        x0 (float): x-coordinate of the crop's top-left corner
        y0 (float): y-coordinate of the crop's top-left corner
        w  (float): width of the crop
        h  (float): height of the crop
    """
    def __init__(self, x0, y0, w, h):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.input_shape

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


class CenterCrop(GeometricTransformation):
    """Extracts a central crop from the image with given width and height.

    Args:
        w  (float): width of the crop
        h  (float): height of the crop
    """
    width: int
    height: int

    def __init__(self, width: int, height: int = None):
        super().__init__()
        self.width = width
        self.height = width if height is None else height

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        # Cropping is done implicitly via output_shape
        pass

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)

    def __repr__(self):
        return f'CenterCrop({self.width}, {self.height})'


class RandomCrop(GeometricTransformation):
    """Extracts a random crop from the image with given width and height.

    Args:
        w  (float): width of the crop
        h  (float): height of the crop
    """
    width: int
    height: int

    def __init__(self, width: int, height: int = None):
        super().__init__()
        self.width = width
        self.height = width if height is None else height

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.input_shape

        limit_y = (H - self.height) / 2
        limit_x = (W - self.width) / 2

        center_y, center_x = jax.random.uniform(rng, [2],
                minval=jnp.array([-limit_y, -limit_x]),
                maxval=jnp.array([limit_y, limit_x]))

        transform = jnp.array([
            [1, 0,  center_y],
            [0, 1,  center_x],
            [0, 0,         1]
        ])
        coordinates.push_transform(transform)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)


class RandomSizedCrop(GeometricTransformation):
    """Extracts a randomly sized crop from the image and rescales it to the given width and height.

    Args:
        w  (float): width of the crop
        h  (float): height of the crop
        zoom_range (float, float): minimum and maximum zoom level for the transformation
        prevent_underzoom (bool): whether to prevent zooming beyond the image size
    """
    width: int
    height: int
    min_zoom: float
    max_zoom: float

    def __init__(self,
            width: int, height: int = None, zoom_range: tuple[float, float] = (0.5, 2.0),
            prevent_underzoom: bool = True):
        super().__init__()
        self.width = width
        self.height = width if height is None else height
        self.min_zoom = zoom_range[0]
        self.max_zoom = zoom_range[1]
        self.prevent_underzoom = prevent_underzoom

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.input_shape
        key1, key2 = jax.random.split(rng)

        if self.prevent_underzoom:
            min_zoom = max(self.min_zoom, math.log(self.height / H), math.log(self.width / W))
            max_zoom = max(self.max_zoom, min_zoom)
        else:
            min_zoom = self.min_zoom
            max_zoom = self.max_zoom

        zoom = utils.log_uniform(key1, minval=min_zoom, maxval=max_zoom)

        limit_y = ((H*zoom) - self.height) / 2
        limit_x = ((W*zoom) - self.width) / 2

        center = jax.random.uniform(key2, [2],
            minval=jnp.array([-limit_y, -limit_x]),
            maxval=jnp.array([limit_y, limit_x]))

        # Out matrix:
        # [ 1/zoom    0   1/c_y ]
        # [   0    1/zoom 1/c_x ]
        # [   0       0     1   ]
        transform = jnp.concatenate([
            jnp.concatenate([jnp.eye(2), center.reshape(2, 1)], axis=1) / zoom,
            jnp.array([[0, 0, 1]])
        ], axis=0)

        coordinates.push_transform(transform)

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return (self.height, self.width)


class Warp(GeometricTransformation):
    """
    Warp an image (similar to ElasticTransform).

    Args:
        strength (float): How strong the transformation is, corresponds to the standard deviation of
            deformation values.
        coarseness (float): Size of the initial deformation grid cells. Lower values lead to a more noisy deformation.
    """
    def __init__(self, strength: int=5, coarseness: int=32):
        super().__init__()
        self.strength = strength
        self.coarseness = coarseness

    def transform_coordinates(self, coordinates: LazyCoordinates, rng: jnp.ndarray):
        H, W = coordinates.final_shape

        H_, W_ = H // self.coarseness, W // self.coarseness
        coordshift_coarse = self.strength * jax.random.normal(rng, [2, H_, W_]) 
        # Note: This is not 100% correct as it ignores possible perspective conmponents of
        #       the current transform. Also, interchanging resize and transform application
        #       is a speed hack, but this shouldn't diminish the quality.
        coordshift = jnp.tensordot(coordinates._current_transform[:2, :2], coordshift_coarse, axes=1)
        coordshift = jax.image.resize(coordshift, (2, H, W), method='bicubic')
        coordinates.apply_pixelwise_offsets(coordshift)
