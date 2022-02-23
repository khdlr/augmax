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
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial

from ._types import RNGKey, PyTree

 
class InputType(Enum):
  IMAGE = 'image'
  MASK = 'mask'
  DENSE = 'dense'
  CONTOUR = 'contour'
  KEYPOINTS = 'keypoints'


def same_type(left_type, right_type):
  if isinstance(left_type, InputType):
    left_type = left_type.value
  if isinstance(right_type, InputType):
    right_type = right_type.value
  return left_type.lower() == right_type.lower()


class Transformation(ABC):
  def __init__(self, input_types=None):
    self.input_types = input_types

  @abstractmethod
  def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree=None, invert=False) -> PyTree:
    pass

  def __call__(self, rng: RNGKey, inputs: PyTree, input_types: PyTree=None, invert=False) -> PyTree:
    return self.apply(rng, inputs, input_types, invert)

  def invert(self, rng: RNGKey, inputs: PyTree, input_types: PyTree=None) -> PyTree:
    return self.apply(rng, inputs, self.input_types, invert=True)


class StopOptimization(Transformation):
  def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree=None, invert=False) -> PyTree:
    return inputs


class BaseChain(Transformation):
  def __init__(self, *transforms: Transformation, input_types=None):
    super().__init__(input_types)
    self.transforms = transforms

  def apply(self, rng: RNGKey, inputs: PyTree, input_types: PyTree=None, invert=False) -> PyTree:
    input_types = input_types or self.input_types
    if input_types is None:
      input_types = jax.tree_map(lambda x: InputType.IMAGE, inputs)

    N = len(self.transforms)
    subkeys = [None]*N if rng is None else list(jax.random.split(rng, N))

    transforms = self.transforms
    if invert:
      transforms = reversed(transforms)
      subkeys = reversed(subkeys)

    transformed = inputs
    for transform, subkey in zip(transforms, subkeys):
      fun = partial(transform.apply, subkey, invert=invert)
      transformed = jax.tree_multimap(fun, transformed, input_types)
    return transformed 

  def __repr__(self):
    members_repr = ",\n".join(str(t) for t in self.transforms)
    members_repr = '\n'.join(['\t'+line for line in members_repr.split('\n')])
    return f'{self.__class__.__name__}(\n{members_repr}\n)'
