# Change log

## augmax 0.3.0
* Changes:
  * Arguments to the augment function can now be arbitrary [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html), this includes `list`s and `dict`s.
  * The `input_types` argument must now be a PyTree that matches the structure of the inputs.
  * Default gamma range for the `RandomGamma` augmentation was changed from `[0.75, 1.33]` to `[0.25, 4.0]`
* New Augmentations:
  * [ChannelDrop](https://augmax.readthedocs.io/en/latest/augmentations/colorspace.html#augmax.ChannelDrop): Drop a random channel from the image.
  * [RandomChannelGamma](https://augmax.readthedocs.io/en/latest/augmentations/colorspace.html#augmax.RandomChannelGamma): Apply a separate Gamma transform to each channel of the image.

