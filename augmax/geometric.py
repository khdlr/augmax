# Copyright 2021 Konrad Heidler
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List, Tuple, Any
from abc import abstractmethod
import math
import warnings

import jax
import jax.numpy as jnp
from einops import rearrange

from .base import Transformation, BaseChain, InputType, same_type
from . import utils
from .geometric_utils import CoordinateTracer


class GeometricTransformation(Transformation):
    @abstractmethod
    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False) -> CoordinateTracer:
        return coordinates

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType]=None, invert=False) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        input_shape  = inputs[0].shape[:2]
        output_shape = self.output_shape(input_shape)
        if invert:
            if not self.size_changing():
                output_shape = input_shape
            elif hasattr(self, 'shape_full'):
                output_shape = self.shape_full
            else:
                raise ValueError("Can't invert a size-changing transformation without running it forward once.")
        else:
            self.shape_full = input_shape

        coordinates = CoordinateTracer(input_shape)
        coordinates.final_shape = output_shape

        if invert:
            coordinates.current_shape = output_shape

        self.transform_coordinates(rng, coordinates, invert)
        sampling_coords = coordinates.get_coordinate_grid()

        val = []
        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE) or same_type(type, InputType.DENSE):
                # Linear Interpolation for Images
                current = utils.resample_image(input, sampling_coords, order=1, mode='constant')
            elif same_type(type, InputType.MASK):
                # Nearest Interpolation for Masks
                current = utils.resample_image(input, sampling_coords, order=0, mode='constant')
            elif same_type(type, InputType.KEYPOINTS):
                current = coordinates.apply_to_points(input)
            elif same_type(type, InputType.CONTOUR):
                current = coordinates.apply_to_points(input)
                current = jnp.where(jnp.linalg.det(coordinates._current_transform) < 0,
                    current[::-1],
                    current
                )

            if current is None:
                raise NotImplementedError(f"Cannot transform input of type {type} with {self.__class__.__name__}")
            val.append(current)
        return val

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return input_shape

    def size_changing(self):
        return False
        if invert:
            if hasattr(self, 'shape_full'):
                output_shape = self.shape_full
            elif self.size_changing():
                raise ValueError("Can't invert a size-changing transformation without running it forward once.")


class SizeChangingGeometricTransformation(GeometricTransformation):
    def size_changing(self):
        return True


class GeometricChain(GeometricTransformation, BaseChain):
    def __init__(self, *transforms: GeometricTransformation):
        super().__init__()
        for transform in transforms:
            assert isinstance(transform, GeometricTransformation), f"{transform} is not a GeometricTransformation!"
        self.transforms = transforms

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        shape_chain = [coordinates.input_shape]

        for transform in self.transforms:
            shape_chain.append(transform.output_shape(shape_chain[-1]))

        N = len(self.transforms)
        subkeys = [None]*N if rng is None else jax.random.split(rng, N)

        transforms = self.transforms
        if not invert:
            # Reverse the transformations iff not inverting!
            transforms = reversed(transforms)
            subkeys = reversed(subkeys)
            shape_chain = reversed(shape_chain[:-1])

        for transform, current_shape, subkey in zip(transforms, shape_chain, subkeys):
            coordinates.current_shape = current_shape
            transform.transform_coordinates(subkey, coordinates, invert=invert)

        return coordinates

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        shape = input_shape
        for transform in self.transforms:
            shape = transform.output_shape(shape)
        return shape

    def size_changing(self):
        return any(t.size_changing() for t in self.transforms)


class HorizontalFlip(GeometricTransformation):
    """Randomly flips an image horizontally.

    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.probability = p

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        params = jax.random.bernoulli(rng, 0.5, [2])
        flip = 1. - 2. * params[0] 
        rot = params[1]

        if invert:
            flip = (2. * rot - 1.) * flip

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
            angle_range: Union[Tuple[float, float], float]=(-30, 30),
            p: float = 1.0):
        super().__init__()
        if not hasattr(angle_range, '__iter__'):
            angle_range = (-angle_range, angle_range)
        self.theta_min, self.theta_max = map(math.radians, angle_range)
        self.probability = p

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        do_apply = jax.random.bernoulli(rng, self.probability)
        theta = do_apply * jax.random.uniform(rng, minval=self.theta_min, maxval=self.theta_max)

        if invert:
            theta = -theta

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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        dy = self.dy
        dx = self.dx
        if invert:
            dy = -dy
            dx = -dx
        transform = jnp.array([
            [1, 0, -dy],
            [0, 1, -dx],
            [0, 0,   1]
        ])
        coordinates.push_transform(transform)


class Crop(SizeChangingGeometricTransformation):
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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        H, W = coordinates.current_shape

        center_x = self.x0 + self.width / 2 - W / 2
        center_y = self.y0 + self.height / 2 - H / 2

        # self.dx/dy is in (0,0) -- (H,W) reference frame
        # => push it to (-H/2, -W/2) -- (H/2, W/2) reference frame

        # Forward transform: Translate by (dx, dy)
        if invert:
            center_y = -center_y
            center_x = -center_x

        transform = jnp.array([
            [1, 0,  center_y],
            [0, 1,  center_x],
            [0, 0,          1]
        ])
        coordinates.push_transform(transform)

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return (self.height, self.width)


class Resize(SizeChangingGeometricTransformation):
    def __init__(self, width: int, height: int = None):
        super().__init__()
        self.width = width
        self.height = width if height is None else height

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return (self.height, self.width)

    def __repr__(self):
        return f'Resize({self.width}, {self.height})'

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        H, W = coordinates.current_shape
        H_, W_ = self.height, self.width

        sy = H / H_
        sx = W / W_
        if invert:
            sy = 1 / sy
            sx = 1 / sx

        transform = jnp.array([
            [sy,  0, 0],
            [ 0, sx, 0],
            [ 0,  0, 1],
        ])

        coordinates.push_transform(transform)


class CenterCrop(SizeChangingGeometricTransformation):
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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        # Cropping is done implicitly via output_shape
        pass

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return (self.height, self.width)

    def __repr__(self):
        return f'CenterCrop({self.width}, {self.height})'


class RandomCrop(SizeChangingGeometricTransformation):
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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        H, W = coordinates.current_shape

        limit_y = (H - self.height) / 2
        limit_x = (W - self.width) / 2

        center_y, center_x = jax.random.uniform(rng, [2],
                minval=jnp.array([-limit_y, -limit_x]),
                maxval=jnp.array([limit_y, limit_x]))

        if invert:
            center_y = -center_y
            center_x = -center_x

        transform = jnp.array([
            [1, 0,  center_y],
            [0, 1,  center_x],
            [0, 0,         1]
        ])
        coordinates.push_transform(transform)

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        return (self.height, self.width)


class RandomSizedCrop(SizeChangingGeometricTransformation):
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
            width: int, height: int = None, zoom_range: Tuple[float, float] = (0.5, 2.0),
            prevent_underzoom: bool = True):
        super().__init__()
        self.width = width
        self.height = width if height is None else height
        self.min_zoom = zoom_range[0]
        self.max_zoom = zoom_range[1]
        self.prevent_underzoom = prevent_underzoom

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        H, W = coordinates.current_shape
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
        if not invert:
            transform = jnp.concatenate([
                jnp.concatenate([jnp.eye(2), center.reshape(2, 1)], axis=1) / zoom,
                jnp.array([[0, 0, 1]])
            ], axis=0)
        else:
            transform = jnp.concatenate([
                jnp.concatenate([jnp.eye(2) * zoom, -center.reshape(2, 1)], axis=1),
                jnp.array([[0, 0, 1]])
            ], axis=0)

        coordinates.push_transform(transform)

    def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
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

    def transform_coordinates(self, rng: jnp.ndarray, coordinates: CoordinateTracer, invert=False):
        if invert:
            warnings.warn("Inverting a Warp transform not yet implemented. Returning warped image as is.")
            return

        H, W = coordinates.final_shape

        H_, W_ = H // self.coarseness, W // self.coarseness
        coordshift_coarse = self.strength * jax.random.normal(rng, [2, H_, W_]) 
        # Note: This is not 100% correct as it ignores possible perspective conmponents of
        #       the current transform. Also, interchanging resize and transform application
        #       is a speed hack, but this shouldn't diminish the quality.
        coordshift = jnp.tensordot(coordinates._current_transform[:2, :2], coordshift_coarse, axes=1)
        coordshift = jax.image.resize(coordshift, (2, H, W), method='bicubic')
        coordinates.apply_pixelwise_offsets(coordshift)
