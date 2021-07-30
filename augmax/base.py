import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class Transformation(ABC):
    def __call__(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return self.apply(image, rng)

    @abstractmethod
    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return image


class TransformationChain(Transformation):
    def __init__(self, *transforms: Transformation):
        self.transforms = transforms

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        subkeys = jax.random.split(rng, len(self.transforms))
        for transform, subkey in zip(self.transforms, subkeys):
            image = transform.apply(image, subkey)
        return image
