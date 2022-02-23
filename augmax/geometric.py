# Copyright 2022 Konrad Heidler
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, List, Tuple, Optional, Any
from abc import abstractmethod
import math
import warnings
from functools import partial

import jax
import jax.numpy as jnp

from .base import Transformation, BaseChain, InputType, same_type
from . import utils
from .functional import geometric as F
from ._types import RNGKey, PyTree


class TracedGeometricTransformation(Transformation):
  def __init__(self, tracer: F.CoordinateTracer):
    self.tracer = tracer

  def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree, invert=False) -> PyTree:
    if invert:
      H, W, *_ = utils.tree_first(inputs).shape
      tracer = self.tracer.invert((H, W))
    else:
      tracer = self.tracer
    coords = tracer.get()

    def inner(input, input_type):
      if same_type(input_type, InputType.IMAGE) or same_type(input_type, InputType.DENSE):
        # Linear Interpolation for Images
        return utils.resample_image(input, coords, order=1, mode='constant')
      elif same_type(input_type, InputType.MASK):
        # Nearest Interpolation for Masks
        return utils.resample_image(input, coords, order=0, mode='constant', cval=-1)
      elif same_type(input_type, InputType.KEYPOINTS):
        return tracer.invert().get()
      ## This is currently broken:
      ## elif same_type(type, InputType.CONTOUR):
      ##   current = coordinates.invert(input).evaluate()
      ##   current = jnp.where(jnp.linalg.det(coordinates._current_transform) < 0,
      ##     current[::-1],
      ##     current
      ##   )
      else:
        raise NotImplementedError(
            f"Cannot transform inputof type {type} with {self.__class__.__name__}")

    return jax.tree_multimap(inner, inputs, input_types)


class GeometricTransformation(Transformation):
  def __init__(self):
    self.traced = None

  @abstractmethod
  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    return coordinates

  def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree, invert=False) -> PyTree:
    traced = self.compile(rng, inputs)
    return traced.apply(rng, inputs, input_types, invert)

  def compile(self, rng: RNGKey, inputs: PyTree) -> TracedGeometricTransformation:
    if self.traced is None:
      H, W, *_ = utils.tree_first(inputs).shape
      coordinates = F.CoordinateTracer((H, W))
      coordinates = self.transform_coordinates(rng, coordinates)
      self.traced = TracedGeometricTransformation(coordinates)
    return self.traced


class GeometricChain(GeometricTransformation):
  def __init__(self, *transforms: GeometricTransformation):
    super().__init__()
    for transform in transforms:
      assert isinstance(transform, GeometricTransformation), f"{transform} is not a GeometricTransformation!"
    self.transforms = transforms

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    N = len(self.transforms)
    subkeys = [None]*N if rng is None else jax.random.split(rng, N)

    transforms = self.transforms
    for transform, subkey in zip(transforms, subkeys):
      coordinates = transform.transform_coordinates(subkey, coordinates)

    return coordinates


class HorizontalFlip(GeometricTransformation):
  """Randomly flips an image horizontally.

  Args:
    p (float): Probability of applying the transformation
  """
  def __init__(self, p: float = 0.5):
    super().__init__()
    self.probability = p

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    f = 1. - 2. * jax.random.bernoulli(rng, self.probability)
    return coordinates * jnp.array([1, f])


class VerticalFlip(GeometricTransformation):
  """Randomly flips an image vertically.

  Args:
    p (float): Probability of applying the transformation
  """
  def __init__(self, p: float = 0.5):
    super().__init__()
    self.probability = p

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    f = 1. - 2. * jax.random.bernoulli(rng, self.probability)
    return coordinates * jnp.array([f, 1])


class Rotate90(GeometricTransformation):
  """Randomly rotates the image by a multiple of 90 degrees.
  """
  def __init__(self):
    super().__init__()

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    params = jax.random.bernoulli(rng, 0.5, [2])
    flip = 1. - 2. * params[0] 
    rot = params[1]

    T = jnp.array([
      [flip * rot,       flip * (1.-rot)],
      [flip * (-1.+rot), flip * rot     ],
    ])

    return T @ coordinates


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

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    do_apply = jax.random.bernoulli(rng, self.probability)
    theta = do_apply * jax.random.uniform(rng, minval=self.theta_min, maxval=self.theta_max)
    return F.rotate(coordinates, theta)


class Translate(GeometricTransformation):
  def __init__(self, dx, dy):
    super().__init__()
    self.dx = dx
    self.dy = dy

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    return coordinates + jnp.array([self.dy, self.dx])


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

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    H, W = coordinates.shape
    center_x = self.x0 + self.width / 2 - W / 2
    center_y = self.y0 + self.height / 2 - H / 2

    # self.dx/dy is in (0,0) -- (H,W) reference frame
    # => push it to (-H/2, -W/2) -- (H/2, W/2) reference frame
    coordinates += jnp.array([center_y, center_x])
    coordinates = coordinates.crop(self.height, self.width)
    return coordinates


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

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    # Cropping is done implicitly via output_shape
    return coordinates.crop(self.height, self.width)

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

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    H, W = coordinates.shape

    limit_y = (H - self.height) / 2
    limit_x = (W - self.width) / 2

    translation = jax.random.uniform(rng, [2],
        minval=jnp.array([-limit_y, -limit_x]),
        maxval=jnp.array([limit_y, limit_x]))

    coordinates += translation
    coordinates = coordinates.crop(self.height, self.width)
    return coordinates

  def output_shape(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
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
      width: int, height: int = None, zoom_range: Tuple[float, float] = (0.5, 2.0),
      prevent_underzoom: bool = True):
    super().__init__()
    self.width = width
    self.height = width if height is None else height
    self.min_zoom = zoom_range[0]
    self.max_zoom = zoom_range[1]
    self.prevent_underzoom = prevent_underzoom

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    H, W = coordinates.shape
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

    # TODO: these might be switched
    coordinates *= zoom
    coordinates += center
    coordinates = coordinates.crop(self.height, self.width)
    return coordinates


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

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    H, W = coordinates.shape
    H_, W_ = H // self.coarseness, W // self.coarseness
    coordshift_coarse = self.strength * jax.random.normal(rng, [2, H_, W_]) 
    # Note: This is not 100% correct as it ignores possible perspective conmponents of
    #       the current transform. Also, interchanging resize and transform application
    #       is a speed hack, but this shouldn't diminish the quality.
    def inner(coords):
      _, h, w = coords.shape
      coordshift = jax.image.resize(coordshift_coarse, (2, h, w), method='bicubic')
      return coords + coordshift
    return coordinates.with_append(F.LambdaOp(inner))


class PiecewiseLinearWarp(GeometricTransformation):
  def __init__(self, grid_size, strength=1.0):
    if isinstance(grid_size, int):
      grid_size = (grid_size, grid_size)
    self.grid_size = grid_size
    self.strength = strength

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    H, W = coordinates.shape
    Gh, Gw = self.grid_size
    G = jnp.array([Gh, Gw])
    GS = jnp.array([H / Gh, W / Gw])

    grid_y = self.strength * jnp.exp(0.8 * jax.random.normal(rng, [self.grid_size[0]]))
    grid_x = self.strength * jnp.exp(0.8 * jax.random.normal(rng, [self.grid_size[1]]))
    # grid = jnp.ones([H // Gh, W // Gw])

    def coord_to_grid(coord):
      return (1.0 + coord) / 2.0 * G

    def grid_to_coord(coord):
      return coord * (2.0 / G) - 1.0

    def make_warp(backward=False):
      def inner(coord):
        coord = coord_to_grid(coord)
        floor, frac = jnp.divmod(coord, 1.0)
        floor_y, floor_x = jnp.split(floor, 2)
        k_y = jax.scipy.ndimage.map_coordinates(grid_y, floor_y, order=0, mode='wrap')
        k_x = jax.scipy.ndimage.map_coordinates(grid_x, floor_x, order=0, mode='wrap')
        k = jnp.stack([k_y, k_x])
        if backward:
          k = 1./k
        return grid_to_coord(floor + jnp.where(frac > k/(1+k),
            k * frac + 1 - k,
            frac / k
        ))
      return jax.vmap(jax.vmap(inner, in_axes=1, out_axes=1), in_axes=2, out_axes=2)

    forward  = make_warp(False)
    backward = make_warp(True)
    return coordinates.with_append(F.LambdaOp(forward, backward))


class InvertibleWarp(GeometricTransformation):
  def __init__(self, grid_size, strength=1.0):
    self.warp1 = PiecewiseLinearWarp(grid_size, strength=strength/2)
    self.warp2 = PiecewiseLinearWarp(grid_size, strength=strength/2)

  def transform_coordinates(self, rng: RNGKey, coordinates: F.CoordinateTracer) -> F.CoordinateTracer:
    subkeys = jax.random.split(rng, 3)
    angle = jax.random.uniform(subkeys[0], 0, 2*math.pi)
    coordinates = self.warp1.transform_coordinates(subkeys[1], coordinates)
    coordinates = F.rotate(coordinates, angle)
    coordinates = self.warp2.transform_coordinates(subkeys[2], coordinates)
    coordinates = F.rotate(coordinates, -angle)
    return coordinates
 

def shape_snoop(fun):
  def inner(x):
    print(x.shape)
    return fun(x)
  return inner
