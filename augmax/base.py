import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Sequence
from enum import Enum

from .utils import unpack_list_if_singleton

 
class InputType(Enum):
    IMAGE = 'image'
    MASK = 'mask'
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
            if input_types is None:
                self.input_types = [InputType.IMAGE]
            else:
                self.input_types = input_types

    def __call__(self, rng: jnp.ndarray, *inputs: jnp.ndarray) -> Union[jnp.ndarray, Sequence[jnp.ndarray]]:
        if len(self.input_types) != len(inputs):
            raise ValueError(f"List of input types (length {len(self.input_types)}) must match inputs to Augmentation (length {len(inputs)})")
        augmented = self.apply(rng, inputs, self.input_types)
        return unpack_list_if_singleton(augmented)

    @abstractmethod
    def apply(self, rng: jnp.ndarray, inputs: Sequence[jnp.ndarray], input_types: Sequence[InputType]=None) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types
        val = []
        for input, type in zip(inputs, input_types):
            val.append(input)
        return val


class BaseChain(Transformation):
    def __init__(self, *transforms: Transformation, input_types=[InputType.IMAGE]):
        super().__init__(input_types)
        self.transforms = transforms

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: Sequence[InputType]=None) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types
        subkeys = jax.random.split(rng, len(self.transforms))

        images = list(inputs)
        for transform, subkey in zip(self.transforms, subkeys):
            images = transform.apply(subkey, images, input_types)
        return images 

    def __repr__(self):
        members_repr = ",\n".join(str(t) for t in self.transforms)
        members_repr = '\n'.join(['\t'+line for line in members_repr.split('\n')])
        return f'{self.__class__.__name__}(\n{members_repr}\n)'
