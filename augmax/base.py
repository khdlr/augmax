import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class Transformation(ABC):
    def __call__(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return self.apply(image, rng)

    @abstractmethod
    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return image


class Chain(Transformation):
    def __init__(self, *transforms: Transformation):
        self.transforms = transforms

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        subkeys = jax.random.split(rng, len(self.transforms))
        for transform, subkey in zip(self.transforms, subkeys):
            image = transform.apply(image, subkey)
        return image

    def __repr__(self):
        members_repr = ",\n".join(str(t) for t in self.transforms)
        members_repr = '\n'.join(['\t'+line for line in members_repr.split('\n')])
        return f'{self.__class__.__name__}(\n{members_repr}\n)'
