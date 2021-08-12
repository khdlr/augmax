```{toctree}
:hidden:
:caption: Getting Started
:maxdepth: 2

installation
getting_started
```

```{toctree}
:hidden:
:caption: List of Augmentations
augmentations/geometric
augmentations/colorspace
augmentations/imagelevel
```

# Augmax

Augmax is an image data augmentation framework supporting efficiently-composable transformations
with support for JAX function transformations.
Its strengths are efficient execution of complex augmentation pipelines and batched data augmentation on the GPU/TPU via the use of [`jax.jit`](jax:jax-jit) and [`jax.vmap`](jax:jax-vmap).

In existing data augmentation frameworks,
each transformation is executed separately,
leading to many unnecessary memory reads and writes that could be avoided.
In contrast, Augmax tries its best to fuse transformations together,
so that these data-intensive operations are be minimized.

## Getting Started

Augmax aims to implement an API similar to that of [Albumentations](https://albumentations.ai).
An augmentation pipeline is defined as a sequence of transformations,
which are then randomly applied to the input images.

```python
import jax
import augmax

transform = augmax.Compose(
  augmax.RandomCrop(256, 256),
  augmax.HorizontalFlip(),
  augmax.Rotate(),
)

image = ...

rng = jax.random.PRNGKey(27)

transformed_image = transform(image, rng)
```

## Batch-wise Augmentation on the GPU

Leveraging the JAX infrastructure,
it is possible to greatly accelerate data augmentation by using Just-in-Time compilation (`jax.jit`),
which can execute the code on the GPU, as well as automated batching (`jax.vmap`).

### Augmenting a single image on the GPU
```python
jit_transform = jax.jit(transform)
transformed_image = jit_transform(image, rng)
```

### 


## Benchmarks

