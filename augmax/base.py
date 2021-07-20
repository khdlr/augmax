import jax.numpy as jnp
from abc import ABC, abstractmethod

class Transformation(ABC):
    def __call__(self, images: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return self.apply(images, rng)

    @abstractmethod
    def apply(self, images: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return images


