# Copyright 2022 Konrad Heidler
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
from abc import ABC, abstractmethod
from functools import partial
from .._types import Tuple, List, Any, Mapping, Callable
from .. import flags
from ..utils import centered_linspace

import jax
import jax.numpy as jnp


class CoordinateOp(ABC):
  @abstractmethod
  def apply(self, coordinates: jnp.ndarray) -> jnp.ndarray:
    pass

  @abstractmethod
  def invert(self, scale) -> 'CoordinateOp':
    pass

  def update_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
    return shape

  def __repr__(self):
    return self.__class__.__name__


class PerspectiveOp(CoordinateOp):
  def __init__(self, transformation: jnp.ndarray):
    assert transformation.shape == (3, 3), "Perspective Transforms need to be 3x3"
    self.transformation = transformation

  def apply(self, coordinates: jnp.ndarray) -> jnp.ndarray:
    ones = jnp.ones([1, *coordinates.shape[1:]])
    expanded = jnp.concatenate([coordinates, ones], axis=0)
    xyz = jnp.einsum('ij,j...->i...', self.transformation, expanded)
    xy, z = jnp.split(xyz, [2], axis=0)
    return xy / z

  def invert(self, scale) -> CoordinateOp:
    # TODO: Using the adjunct matrix here might be better numerically
    return PerspectiveOp(jnp.linalg.inv(self.transformation))


class LinearOp(CoordinateOp):
  def __init__(self, transformation: jnp.ndarray):
    assert transformation.shape == (2, 2), "Linear Transforms need to be 2x2"
    self.transformation = transformation

  def apply(self, coordinates: jnp.ndarray):
    return jnp.einsum('ij,j...->i...', self.transformation, coordinates)

  def invert(self, scale) -> CoordinateOp:
    return LinearOp(jnp.linalg.inv(self.transformation))


class Translation(CoordinateOp):
  def __init__(self, translation: jnp.ndarray):
    assert translation.shape == (2, ), \
        f"Coordinate Translations need to have shape [2], not {translation.shape}"
    self.translation = translation

  def apply(self, coordinates: jnp.ndarray):
    return (coordinates.T + self.translation).T

  def invert(self, scale) -> CoordinateOp:
    return Translation(-scale * self.translation)


class LambdaOp(CoordinateOp):
  def __init__(self, function, inverse=None):
    self.function = function
    self.inverse = inverse

  def apply(self, coordinates: jnp.ndarray):
    return self.function(coordinates)

  def invert(self, scale) -> CoordinateOp:
    if self.inverse is None:
      raise ValueError("Cannot invert this operation")
    return LambdaOp(self.inverse, self.function)


class RelativeCropOp(CoordinateOp):
  def __init__(self, relative_height, relative_width):
    assert relative_height <= 1 and relative_width <= 1, \
        f"Cannot crop image larger than it is"
    self.relative_height = relative_height
    self.relative_width = relative_width

  def _crops(self, H, W):
    H_ = round(H * self.relative_height)
    W_ = round(W * self.relative_width)
    y0 = (H - H_) // 2
    y1 = y0 + H_
    x0 = (W - W_) // 2
    x1 = x0 + W_
    return y0, x0, y1, x1

  def apply(self, coordinates: jnp.ndarray):
    _, H, W = coordinates.shape
    y0, x0, y1, x1 = self._crops(H, W)
    print(f'cropping {coordinates.shape} to [:, {y0}:{y1}, {x0}:{x1}]')
    return coordinates[:, y0:y1, x0:x1]

  def invert(self, scale) -> CoordinateOp:
    return RelativePadOp(1.0 / self.relative_height, 1.0 / self.relative_width)

  def transform_frame(self, tl, br):
    scale = jnp.array([self.relative_height, self.relative_width])
    center = (br + tl) / 2
    half_span = (br - tl) / 2
    tl_ = center - scale * half_span
    br_ = center + scale * half_span
    return tl_, br_

  def update_shape(self, shape):
    y0, x0, y1, x1 = self._crops(*shape)
    return y1-y0, x1-x0


class RelativePadOp(CoordinateOp):
  def __init__(self, relative_height, relative_width):
    assert relative_height >= 1 and relative_width >= 1, \
        f"Cannot pad image smaller than it is"
    self.relative_height = relative_height
    self.relative_width = relative_width

  def _pads(self, H, W):
    H_ = round(H * self.relative_height)
    W_ = round(W * self.relative_width)
    y0 = (H_ - H) // 2
    y1 = (H_ - H) - y0
    x0 = (W_ - W) // 2
    x1 = (W_ - W) - x0
    return ((y0, y1), (x0, x1))

  def apply(self, coordinates: jnp.ndarray):
    _, H, W = coordinates.shape
    pad = partial(jnp.pad, pad_width=self._pads(H, W), mode='constant', constant_values=jnp.inf)
    return jax.vmap(pad)(coordinates)

  def invert(self, scale) -> CoordinateOp:
    return RelativeCropOp(1.0 / self.relative_height, 1.0 / self.relative_width)

  def update_shape(self, shape):
    H, W = shape
    (y0, y1), (x0, x1) = self._pads(H, W)
    return H+y1+y0, W+x1+x0

  def transform_frame(self, tl, br):
    scale = jnp.array([self.relative_height, self.relative_width])
    center = (br + tl) / 2
    half_span = (br - tl) / 2
    tl_ = center - scale * half_span
    br_ = center + scale * half_span
    return tl_, br_


class CoordinateTracer:
  """
  Tracer for lazily aggreagating coordinate transformations.

  """
  _ops: List[Any]

  def __init__(self, initial_shape, ops=[], top_left=None, bottom_right=None):
    self._ops = ops
    self.initial_shape = initial_shape
    self.shape = initial_shape
    for op in ops:
      self.shape = op.update_shape(self.shape)

    H, W = initial_shape
    xscl = W / H
    self.top_left     = (top_left     or jnp.array([-1., -xscl]))
    self.bottom_right = (bottom_right or jnp.array([ 1.,  xscl]))

  def with_append(self, transform: CoordinateOp):
    return CoordinateTracer(self.initial_shape, [*self._ops, transform])

  def with_prepend(self, transform: CoordinateOp):
    return CoordinateTracer(self.initial_shape, [transform, *self._ops])

  def __add__(self, translation: jnp.ndarray):
    return self.with_append(Translation(2 * translation / self.shape[0]))

  def __radd__(self, translation: jnp.ndarray):
    return self + translation

  def __mul__(self, scale: jnp.ndarray):
    if jnp.asarray(scale).shape == ():
      return self.with_append(_op_from_matrix(jnp.diag(jnp.array([scale, scale]))))
    else:
      return self.with_append(_op_from_matrix(jnp.diag(scale)))

  def __rmul__(self, scale: jnp.ndarray):
    return self * scale

  def __matmul__(self, matrix: jnp.ndarray):
    return self.with_prepend(_op_from_matrix(matrix))

  def __rmatmul__(self, matrix: jnp.ndarray):
    return self.with_append(_op_from_matrix(matrix))

  def pad(self, height, width):
    H, W = self.shape
    relative_height = height / H
    relative_width = width / W
    return self.with_append(RelativePadOp(relative_height, relative_width))

  def crop(self, height, width):
    H, W = self.shape
    relative_height = height / H
    relative_width = width / W
    return self.with_append(RelativeCropOp(relative_height, relative_width))

  def invert(self, initial_shape=None):
    if initial_shape is None:
      initial_shape = self.shape
    scale = self.initial_shape[0] / self.shape[0]
    ops = [op.invert(scale) for op in reversed(self._ops)]
    return CoordinateTracer(initial_shape, ops)

  def get(self) -> jnp.ndarray:
    if flags.NAIVE_COORDINATE_OPS:
      return self._naive_get()

    optimized_ops = _optimize_ops(self._ops)

    tl, br = self.top_left, self.bottom_right
    if isinstance(optimized_ops[0], (RelativeCropOp, RelativePadOp)):
      crop, *optimized_ops = optimized_ops
      tl, br = crop.transform_frame(tl, br)

    changes_shape = any(map(
      lambda x: isinstance(x, (RelativeCropOp, RelativePadOp)),
      optimized_ops))

    if changes_shape:
      H, W = self.initial_shape
      xscl = W / H

      x, y = jnp.meshgrid(centered_linspace(-xscl, xscl, W), centered_linspace(-1, 1, H))
      coordinate_grid = jnp.stack([y, x])
    else:
      H, W = self.shape
      tl = tl.reshape(2, 1, 1)
      br = br.reshape(2, 1, 1)

      x, y = jnp.meshgrid(centered_linspace(0, 1, W), centered_linspace(0, 1, H))
      grid  = jnp.stack([y, x])
      coordinate_grid = (1-grid) * tl + grid * br

    return _apply_ops(optimized_ops, coordinate_grid)

  def _naive_get(self) -> jnp.ndarray:
    # Naive implementation
    H, W = self.initial_shape
    xscl = W / H

    x, y = jnp.meshgrid(centered_linspace(-xscl, xscl, W), centered_linspace(-1, 1, H))
    coordinate_grid = jnp.stack([y, x])
    return _apply_ops(self._ops, coordinate_grid)


def rotate(coordinates: CoordinateTracer, theta: jnp.ndarray) -> CoordinateTracer:
    T = jnp.array([
      [ jnp.cos(theta), jnp.sin(theta)],
      [-jnp.sin(theta), jnp.cos(theta)],
    ])
    return T @ coordinates


# Operation Type promotions and fusions
PROMOTIONS = {}
def register_promotion(source, target):
  def register_inner(fun):
    global PROMOTIONS
    source_promotions = PROMOTIONS.get(source, {})
    if target in source_promotions:
      raise ValueError(f'Promotion from {source} to {target} has already been registered')
    source_promotions[target] = fun
    PROMOTIONS[source] = source_promotions
    return fun
  return register_inner


FUSIONS = {}
def register_fusion(left, right):
  if not isinstance(left, list):
    left = [left]
  if not isinstance(right, list):
    right = [right]
  def register_inner(fun):
    global FUSIONS
    for l in left:
      for r in right:
        if (l, r) in FUSIONS:
          raise ValueError(f'Fusion for {l} and {r} has already been registered')
        FUSIONS[(l, r)] = fun
    return fun
  return register_inner


@register_promotion(Translation, PerspectiveOp)
def _translation_to_perspective(translation):
  cat = jnp.concatenate
  lastcol = cat([translation.translation, jnp.ones([1])]).reshape(3, 1)
  M = cat([jnp.eye(3, 2), lastcol], axis=1)
  return PerspectiveOp(M)


@register_promotion(LinearOp, PerspectiveOp)
def _linear_to_perspective(linear):
  cat = jnp.concatenate
  M = cat([
    cat([linear.transformation, jnp.zeros([2, 1])], axis=1),
    jnp.array([[0, 0, 1]])
  ], axis=0)
  return PerspectiveOp(M)


@register_fusion(Translation, Translation)
def _translation_fusion(left, right):
  return Translation(left.translation + right.translation)


@register_fusion(LinearOp, LinearOp)
def _linear_fusion(left, right):
  return LinearOp(right.transformation @ left.transformation)


@register_fusion(PerspectiveOp, PerspectiveOp)
def _perspective_fusion(left, right):
  return PerspectiveOp(right.transformation @ left.transformation)


@register_fusion([Translation, LinearOp, PerspectiveOp], [RelativeCropOp, RelativePadOp])
def _push_crops_left(left, right):
  return [right, left]


# @register_fusion(RelativePadOp, [Translation, LinearOp, PerspectiveOp])
# def _push_pads_right(left, right):
#   return [right, left]


# @register_fusion(LambdaOp, LambdaOp)
# def _fuse_lambda(left, right):
#   forward  = lambda c: right.function(left.function(c))
#   backward = lambda c: left.inverse(right.inverse(c))
#   return LambdaOp(forward, backward)


def _promotions(source: type) -> Tuple[Mapping[type, Callable[[CoordinateOp], CoordinateOp]], Mapping[type, int]]:
  """
  A BFS to find all possible promotion targets from a given source promotion
  """
  queue = [source]
  transforms = {source: lambda x: x}
  steps = {source: int(0)}
  while queue:
    current = queue.pop()
    for tgt in PROMOTIONS.get(current, {}):
      if tgt in transforms:
        continue
      current_transform = transforms[current]
      current_steps = steps[current]
      next_transform = PROMOTIONS[current][tgt]
      assert PROMOTIONS[current][tgt] is not None
      transforms[tgt] = lambda x: next_transform(current_transform(x))
      steps[tgt] = current_steps + 1
      queue.append(tgt)
  return transforms, steps


def _fuse(left: CoordinateOp, right: CoordinateOp) -> List[CoordinateOp]:
  fused = FUSIONS[(type(left), type(right))](left, right)
  if not isinstance(fused, list):
    fused = [fused]
  return fused


# Utility functions for CoordinateTracer
def _try_fuse_ops(left: CoordinateOp, right: CoordinateOp) -> List[CoordinateOp]:
  fns_l, steps_l = _promotions(type(left))
  fns_r, steps_r = _promotions(type(right))

  candidates = []
  for l in steps_l:
    for r in steps_r:
      if (l, r) in FUSIONS:
        candidates.append((l, r))
  candidates = sorted(candidates, key=lambda x: steps_l[x[0]] + steps_r[x[1]])

  if not candidates:
    raise NotImplementedError(f"Can't fuse {type(left)} and {type(right)}")
  else:
    type_l, type_r = candidates[0]
    left  = fns_l[type_l](left)
    right = fns_r[type_r](right)
    return _fuse(left, right)


def _optimize_ops(ops: List[CoordinateOp]) -> List[CoordinateOp]:
  did_optimize = True
  while did_optimize:
    did_optimize = False
    i = 0
    while i+1 < len(ops):
      try:
        fused = _try_fuse_ops(ops[i], ops[i+1])
        ops = ops[:i] + fused + ops[i+2:]
        did_optimize = True
      except NotImplementedError as e:
        i += 1
  return ops


def _apply_ops(ops: List[CoordinateOp], coordinates: jnp.ndarray):
    for op in ops:
        coordinates = op.apply(coordinates)
    return coordinates


def _op_from_matrix(matrix: jnp.ndarray):
  if matrix.shape == (2, 2):
    return LinearOp(matrix)
  elif matrix.shape == (3, 3):
    return PerspectiveOp(matrix)
  else:
    raise ValueError(f"Matrix with shape {matrix.shape} cannot be converted to a coordinate operation.")

