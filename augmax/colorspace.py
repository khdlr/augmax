from abc import abstractmethod

import jax
import jax.numpy as jnp

from .base import Transformation
from .utils import log_uniform


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


class ByteToFloat(ColorspaceTransformation):
    """Transforms images from uint8 representation (values 0-255)
    to normalized float representation (values 0.0-1.0)
    """
    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return pixel.astype(jnp.float32) / 255.0


class Normalize(ColorspaceTransformation):
    """Normalizes images using given coefficients

    Args:
        mean (jnp.ndarray): Mean values for each channel
        std (jnp.ndarray): Standard deviation for each channel
    """
    def __init__(self,
            mean: jnp.ndarray = jnp.array([0.485, 0.456, 0.406]),
            std: jnp.ndarray = jnp.array([0.229, 0.224, 0.225])):
        self.mean = jnp.asarray(mean)
        self.std = jnp.asarray(std)

    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return (pixel - self.mean) / self.std


class ChannelShuffle(ColorspaceTransformation):
    """Randomly shuffles an images channels.

    Args:
        p (float): Probability of applying the transformation
    """

    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        return jax.random.permutation(rng, pixel)


class RandomGamma(ColorspaceTransformation):
    """Randomly adjusts the image gamma.

    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self, range: tuple[float, float]=(0.75, 1.33), p: float = 0.5):
        self.range = range
        self.probability = p

    def pixelwise(self, pixel: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomGamma can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_gamma = log_uniform(k1, minval=self.range[0], maxval=self.range[1])
        gamma = jnp.where(jax.random.bernoulli(k2, self.probability), random_gamma, 1.0)

        return jnp.power(pixel, gamma)
