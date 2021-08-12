from typing import Union, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

from einops import rearrange

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
