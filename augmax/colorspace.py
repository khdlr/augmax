from abc import abstractmethod
from typing import List

import jax
import jax.numpy as jnp

from .base import Transformation, BaseChain, InputType, same_type
from .utils import log_uniform, rgb_to_hsv, hsv_to_rgb


class ColorspaceTransformation(Transformation):
    @abstractmethod
    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        return pixel

    def apply(self, rng: jnp.ndarray, inputs: jnp.ndarray, input_types: List[InputType]=None) -> List[jnp.ndarray]:
        if input_types is None:
            input_types = self.input_types

        full_op = jax.jit(jax.vmap(jax.vmap(self.pixelwise, [None, 0], 0), [None, 1], 1))

        val = []
        for input, type in zip(inputs, input_types):
            current = None
            if same_type(type, InputType.IMAGE):
                # Linear Interpolation for Images
                current = full_op(rng, input)
            else:
                current = input
            val.append(current)
        return val


class ColorspaceChain(ColorspaceTransformation, BaseChain):
    def __init__(self, *transforms: ColorspaceTransformation, input_types=None):
        super().__init__(input_types)
        self.transforms = transforms

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        subkeys = jax.random.split(rng, len(self.transforms))
        for transform, subkey in zip(self.transforms, subkeys):
            pixel = transform.pixelwise(subkey, pixel)
        return pixel


class ByteToFloat(ColorspaceTransformation):
    """Transforms images from uint8 representation (values 0-255)
    to normalized float representation (values 0.0-1.0)
    """
    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        return pixel.astype(jnp.float32) / 255.0


class Normalize(ColorspaceTransformation):
    """Normalizes images using given coefficients

    Args:
        mean (jnp.ndarray): Mean values for each channel
        std (jnp.ndarray): Standard deviation for each channel
    """
    def __init__(self,
            mean: jnp.ndarray = jnp.array([0.485, 0.456, 0.406]),
            std: jnp.ndarray = jnp.array([0.229, 0.224, 0.225]),
            input_types=None
    ):
        super().__init__(input_types)
        self.mean = jnp.asarray(mean)
        self.std = jnp.asarray(std)

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        return (pixel - self.mean) / self.std


class ChannelShuffle(ColorspaceTransformation):
    """Randomly shuffles an images channels.

    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        k1, k2 = jax.random.split(rng)
        return jnp.where(jax.random.bernoulli(k2, self.probability),
            jax.random.permutation(k1, pixel),
            pixel
        )


class RandomGamma(ColorspaceTransformation):
    """Randomly adjusts the image gamma.

    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: tuple[float, float]=(0.75, 1.33),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.range = range
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomGamma can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_gamma = log_uniform(k1, minval=self.range[0], maxval=self.range[1])
        gamma = jnp.where(jax.random.bernoulli(k2, self.probability), random_gamma, 1.0)

        return jnp.power(pixel, gamma)


class RandomBrightness(ColorspaceTransformation):
    """Randomly adjusts the image brightness.
   
    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: tuple[float, float] = (-1.0, 1.0),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.minval = range[0] / 2.0
        self.maxval = range[1] / 2.0
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomContrast can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_brightness = jax.random.uniform(k1, minval=self.minval, maxval=self.maxval)
        brightness = jnp.where(jax.random.bernoulli(k2, self.probability), random_brightness, 1.0)
        # cf. https://gitlab.gnome.org/GNOME/gimp/-/blob/master/app/operations/gimpoperationbrightnesscontrast.c
        pixel = jnp.where(brightness < 0.0,
            pixel * (1.0 + brightness),
            pixel + ((1.0 - pixel) * brightness)
        )

        return pixel


class RandomContrast(ColorspaceTransformation):
    """Randomly adjusts the image contrast.
   
    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            range: tuple[float, float] = (-1.0, 1.0),
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.minval = range[0] / 2.0
        self.maxval = range[1] / 2.0
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomContrast can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        k1, k2 = jax.random.split(rng)
        random_contrast = jax.random.uniform(k1, minval=self.minval, maxval=self.maxval)
        contrast = jnp.where(jax.random.bernoulli(k2, self.probability), random_contrast, 0.0)
        slant = jnp.tan((contrast + 1.0) * (jnp.pi / 4))
        # cf. https://gitlab.gnome.org/GNOME/gimp/-/blob/master/app/operations/gimpoperationbrightnesscontrast.c
        # pixel = (pixel - 0.5) * slant + 0.5
        offset = 0.5 * (1 - slant)
        pixel = jnp.clip(pixel * slant + offset, 0.0, 1.0)
        return pixel


class ColorJitter(ColorspaceTransformation):
    """Randomly jitter the image colors.
   
    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            brightness: float = 0.1,
            contrast: float = 0.1,
            saturation: float = 0.1,
            hue: float = 0.1,
            p: float=0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.probability = p
        
        self.keys_needed = sum(1 if val > 0 else 0
                for val in [brightness, contrast, saturation, hue])
        if p < 1:
            self.keys_needed += 1

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        if pixel.shape != (3, ):
            raise ValueError(f"ColorJitter only supports RGB imagery for now, got {pixel.shape}")
        if pixel.dtype != jnp.float32:
            raise ValueError(f"ColorJitter can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")
        keys = iter(jax.random.split(rng, self.keys_needed))
        hue, saturation, value = rgb_to_hsv(pixel)

        if self.hue > 0:
            hue = hue + jax.random.uniform(next(keys), minval=-self.hue, maxval=self.hue)
        if self.saturation > 0:
            saturation = jnp.clip(
                saturation + jax.random.uniform(next(keys), minval=-self.saturation, maxval=self.saturation),
                0, 1)
        if self.brightness > 0:
            value = value + jax.random.uniform(next(keys), minval=-self.brightness, maxval=self.brightness)
        if self.contrast > 0:
            contrast = jax.random.uniform(next(keys), minval=-self.contrast, maxval=self.contrast)
            slant = jnp.tan((contrast + 1.0) * (jnp.pi / 4))
            # cf. https://gitlab.gnome.org/GNOME/gimp/-/blob/master/app/operations/gimpoperationbrightnesscontrast.c
            # pixel = (pixel - 0.5) * slant + 0.5
            offset = 0.5 * (1 - slant)
            value = value * slant + offset
        if self.contrast > 0 or self.brightness > 0:
            value = jnp.clip(value, 0, 1)

        transformed = hsv_to_rgb(hue, saturation, value)

        if self.probability < 1:
            do_apply = jax.random.bernoulli(rng, self.probability)
            transformed = jnp.where(do_apply, transformed, pixel)

        return transformed


class RandomGrayscale(ColorspaceTransformation):
    """Randomly converts the image to grayscale.
   
    Args:
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"RandomContrast can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        do_apply = jax.random.bernoulli(rng, self.probability)
        grayscale = jnp.broadcast_to(pixel.mean(axis=-1, keepdims=True), pixel.shape)
        return jnp.where(do_apply,
                grayscale,
                pixel
        )


class Solarization(ColorspaceTransformation):
    """Randomly solarizes the image.

    Args:
        range (float, float): 
        p (float): Probability of applying the transformation
    """
    def __init__(self,
            threshold: float = 0.5,
            p: float = 0.5,
            input_types=None
    ):
        super().__init__(input_types)
        self.range = range
        self.threshold = threshold
        self.probability = p

    def pixelwise(self, rng: jnp.ndarray, pixel: jnp.ndarray) -> jnp.ndarray:
        if pixel.dtype != jnp.float32:
            raise ValueError(f"Solarization can only be applied to float images, but the input is {pixel.dtype}. "
                    "Please call ByteToFloat first.")

        do_apply = jax.random.bernoulli(rng, self.probability)
        solarized = jnp.where((pixel > self.threshold) & do_apply,
                1.0 - pixel,
                pixel
        )
        return solarized

