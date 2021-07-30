from abc import abstractmethod

import jax
import jax.numpy as jnp

from .base import Transformation


class ColorspaceTransformation(Transformation):
    @abstractmethod
    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return pixel

    def apply(self, image: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        full_op = jax.jit(jax.vmap(jax.vmap(self.pixelwise, [0, None], 0), [1, None], 1))
        image_out = full_op(image, rng)
        return image_out


class ColorspaceChain(ColorspaceTransformation):
    def __init__(self, *transforms: ColorspaceTransformation):
        self.transforms = transforms

    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        subkeys = jax.random.split(rng, len(self.transforms))
        for transform, subkey in zip(self.transforms, subkeys):
            pixel = transform.pixelwise(pixel, subkey)
        return pixel

class ToFloat(ColorspaceTransformation):
    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return pixel / 255.0


class Normalize(ColorspaceTransformation):
    def __init__(self,
            mean: jnp.ndarray = jnp.array([0.485, 0.456, 0.406]),
            std: jnp.ndarray = jnp.array([0.229, 0.224, 0.225])):
        self.mean = mean
        self.std = std

    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return (pixel - self.mean) / self.std

