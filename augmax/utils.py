from typing import Union, Any, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

Tensor = Union[np.ndarray, jnp.ndarray]

def resample_image(image: Tensor, coordinates: Tensor, order: int=1, mode: str='constant', cval: Any=0):
    H, W, C = image.shape
    D, H_, W_ = coordinates.shape
    assert D == 2, f'Expected last dimension of coordinates array to have size 2, got {coordinates.shape}'
    coordinates = coordinates.reshape(2, -1)

    def resample_channel(channel: Tensor):
        return jnd.map_coordinates(channel, coordinates, order=order, mode=mode, cval=cval)

    resampled = jax.vmap(resample_channel, in_axes=-1, out_axes=-1)(image)
    resampled = resampled.reshape(H_, W_, C)

    return resampled


def log_uniform(key, shape=(), dtype=jnp.float32, minval=0.5, maxval=2.0):
    logmin = jnp.log(minval)
    logmax = jnp.log(maxval)

    sample = jax.random.uniform(key, minval=logmin, maxval=logmax)

    return  jnp.exp(sample)


def rgb_to_hsv(pixel: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    cf. https://en.wikipedia.org/wiki/HSL_and_HSV#Color_conversion_formulae
    """
    value = jnp.max(pixel)
    range = value - jnp.min(pixel)
    argmax = jnp.argmax(pixel)
    second = jnp.mod(argmax + 1, 3)
    hue = jnp.where(range == 0.0,
        0.0,
        (2 * argmax + (pixel[argmax] - pixel[second])/range) / 6
    )
    saturation = jnp.where(value == 0,
        0.0,
        range / value
    )

    return hue, saturation, value


def hsv_to_rgb(hue: jnp.ndarray, saturation: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """
    cf. https://en.wikipedia.org/wiki/HSL_and_HSV#Color_conversion_formulae
    """
    n = jnp.array([5, 3, 1])
    k = jnp.mod(n + hue * 6, 6)
    f = value - value * saturation * jnp.maximum(0, jnp.minimum(jnp.minimum(k, 4-k), 1))
    return f
