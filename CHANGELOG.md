# Change log

## augmax 0.3.2
* Fixes:
  * Fixed: Passing differently sized inputs to geometric transforms used to produce weird behaviour. Now this is correctly detected and a more helpful error message is thrown.
  * Fixed: Typing hints updated to be compatible with jax >=0.4.24


## augmax 0.3.1
* Bugfixes:
  * Fixed: Inverting a transformation without given input types
  * Fixed: [RandomChannelGamma](https://augmax.readthedocs.io/en/latest/augmentations/colorspace.html#augmax.RandomChannelGamma) now correctly applies different gamma to each channel.


## augmax 0.3.0
* Changes:
  * Arguments to the augment function can now be arbitrary [PyTrees](https://jax.readthedocs.io/en/latest/pytrees.html), this includes `list`s and `dict`s.
  * The `input_types` argument must now be a PyTree that matches the structure of the inputs.
  * Default gamma range for the `RandomGamma` augmentation was changed from `[0.75, 1.33]` to `[0.25, 4.0]`
* New Augmentations:
  * [ChannelDrop](https://augmax.readthedocs.io/en/latest/augmentations/colorspace.html#augmax.ChannelDrop): Drop a random channel from the image.
  * [RandomChannelGamma](https://augmax.readthedocs.io/en/latest/augmentations/colorspace.html#augmax.RandomChannelGamma): Apply a separate Gamma transform to each channel of the image.

