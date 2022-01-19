from typing import List, Tuple, Any, Mapping, Callable
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp


class CoordinateOp(ABC):
  @abstractmethod
  def apply(self, coordinates: jnp.ndarray) -> jnp.ndarray:
    pass

  @abstractmethod
  def invert(self) -> 'CoordinateOp':
    pass

  @abstractmethod
  def fuse(self, other) -> 'CoordinateOp':
    pass

  @staticmethod
  def promotes_to() -> List[type]:
    return []


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

  def invert(self) -> CoordinateOp:
    # TODO: Using the adjunct matrix here might be better numerically
      return PerspectiveOp(jnp.linalg.inv(self.transformation))

  def fuse(self, other: CoordinateOp):
    if isinstance(other, PerspectiveOp):
      return PerspectiveOp(other.transformation @ self.transformation)
    else:
      raise NotImplementedError(f"Cannot fuse {type(self)} with {type(other)}")

  @staticmethod
  def promotes_to() -> List[type]:
    return []


class LinearOp(CoordinateOp):
  def __init__(self, transformation: jnp.ndarray):
    assert transformation.shape == (2, 2), "Linear Transforms need to be 2x2"
    self.transformation = transformation

  def apply(self, coordinates: jnp.ndarray):
    return jnp.einsum('ij,j...->i...', self.transformation, coordinates)

  def invert(self) -> CoordinateOp:
    return LinearOp(jnp.linalg.inv(self.transformation))

  def fuse(self, other: CoordinateOp):
    if isinstance(other, LinearOp):
      return LinearOp(other.transformation @ self.transformation)
    else:
      raise NotImplementedError(f"Cannot fuse {type(self)} with {type(other)}")


class Translation(CoordinateOp):
  def __init__(self, translation: jnp.ndarray):
    assert translation.shape == (2, ), \
        f"Coordinate Translations need to have shape [2], not {translation.shape}"
    self.translation = translation

  def apply(self, coordinates: jnp.ndarray):
    return (coordinates.T + self.translation).T

  def invert(self) -> CoordinateOp:
    return Translation(-self.translation)

  def fuse(self, other: CoordinateOp):
    if isinstance(other, Translation):
      return Translation(other.translation + self.translation)
    else:
      raise NotImplementedError(f"Cannot fuse {type(self)} with {type(other)}")


class LambdaOp(CoordinateOp):
  def __init__(self, function, inverse=None):
    self.function = function
    self.inverse = inverse

  def apply(self, coordinates: jnp.ndarray):
    return self.function(coordinates)

  def invert(self) -> CoordinateOp:
    if self.inverse is None:
      raise ValueError("Cannot invert this operation")
    return LambdaOp(self.inverse, self.function)

  def fuse(self, other: CoordinateOp):
    if isinstance(other, LambdaOp):
      forward  = lambda c: other.function(self.function(c))
      backward = lambda c: self.inverse(other.inverse(c))
      return LambdaOp(forward, backward)
    else:
      raise NotImplementedError(f"Cannot fuse {type(self)} with {type(other)}")


# Operation Type promotions
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


def promotions(source: type) -> Mapping[type, Tuple[int, Callable[[CoordinateOp], CoordinateOp]]]:
  """
  A BFS to find all possible promotion targets from a given source promotion
  """
  queue = [source]
  transforms = {source: (int(0), lambda x: x)}
  while queue:
    current = queue.pop()
    for tgt in PROMOTIONS.get(current, {}):
      if tgt in transforms:
        continue
      steps, current_transform = transforms[current]
      next_transform = PROMOTIONS[current][tgt]
      assert PROMOTIONS[current][tgt] is not None
      transforms[tgt] = (steps+1, lambda x: next_transform(current_transform(x)))
      queue.append(tgt)
  return transforms


# Utility functions for CoordinateTracer
def _fuse_ops(left: CoordinateOp, right: CoordinateOp) -> CoordinateOp:
  fns_left  = promotions(type(left))
  fns_right = promotions(type(right))

  candidates = set(fns_left) & set(fns_right)
  candidates = sorted(candidates, key=lambda x: fns_left[x][0] + fns_right[x][0])

  if not candidates:
    raise NotImplementedError(f"Can't fuse {type(left)} and {type(right)}")
  else:
    tgt_type = candidates[0]
    left  = fns_left[tgt_type][1](left)
    right = fns_right[tgt_type][1](right)
    return left.fuse(right)


def _optimize_ops(ops: List[CoordinateOp]) -> List[CoordinateOp]:
  did_optimize = True
  while did_optimize:
      did_optimize = False
      i = 0
      while i+1 < len(ops):
        try:
          fused = _fuse_ops(ops[i], ops[i+1])
          ops = ops[:i] + [fused] + ops[i+2:]
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


class CoordinateTracer:
    """
    Tracer for lazily aggreagating coordinate transformations.

    """
    _ops: List[Any]

    def __init__(self, ops=[]):
        self._ops = ops

    def __add__(self, translation: jnp.ndarray):
        return CoordinateTracer([*self._ops, Translation(translation)])

    def __radd__(self, translation: jnp.ndarray):
        return CoordinateTracer([*self._ops, Translation(translation)])

    def __matmul__(self, matrix: jnp.ndarray):
        return CoordinateTracer([_op_from_matrix(matrix), *self._ops])

    def __rmatmul__(self, matrix: jnp.ndarray):
        return CoordinateTracer([*self._ops, _op_from_matrix(matrix)])

    def evaluate(self, coordinates: jnp.ndarray) -> jnp.ndarray:
        optimized_ops = _optimize_ops(self._ops)
        return _apply_ops(optimized_ops, coordinates)

    def _naive_evaluate(self, coordinates: jnp.ndarray) -> jnp.ndarray:
        # Naive implementation
        return _apply_ops(self._ops, coordinates)
