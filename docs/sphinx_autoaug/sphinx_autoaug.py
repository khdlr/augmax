"""
Automatically document Augmax augmentations
including sample outputs
"""

from docutils import nodes
from sphinx.ext.autodoc.directive import AutodocDirective

import jax
import jax.numpy as jnp
import json
import augmax
from imageio import imread, imwrite
from pathlib import Path

jax.config.update("jax_platform_name", "cpu")

import inspect

SEED = 42
N_IMGS = 3


def generate_images(augmentation, args, kwargs={}, to_float: bool = False):
  augname = augmentation.__name__
  basedir = Path(__file__).parent.parent
  image = imread(basedir / "teddy.png")
  keys = jax.random.split(jax.random.PRNGKey(SEED), N_IMGS)

  transform = augmentation(*args, **kwargs)
  if to_float:
    transform = augmax.Chain(augmax.ByteToFloat(), transform)

  transform = jax.jit(jax.vmap(transform, (0, None)))
  images = transform(keys, image)

  if images.dtype == jnp.float32:
    # if augname == 'ByteToFloat' or to_float:
    images = (images * 255.0).astype(jnp.uint8)

  imgdir = Path(basedir / "generated_imgs").absolute()
  imgdir.mkdir(exist_ok=True)
  imgnames = []
  for i in range(N_IMGS):
    imgname = str(imgdir / f"{augname}_{i}.png")
    imwrite(imgname, images[i])
    imgnames.append("/" + imgname)
  return imgnames


class AutoAugmentation(AutodocDirective):
  required_arguments = 1
  optional_arguments = 10

  def run(self):
    # Leverage autoclass for the base documentation.
    # To do so, we have to go incognito and change our name... ;)
    self.name = "autoclass"
    augname, *args = self.arguments
    args = json.loads("[" + ", ".join(args) + "]")
    augmentation = getattr(augmax, augname)
    self.arguments = [augname]
    cls_result = super().run()

    to_float = False
    if args and args[0] == "flt":
      args = args[1:]
      to_float = True

    kwargs = {}
    if "p" in inspect.getfullargspec(augmentation)[0]:
      kwargs["p"] = 1.0

    images = generate_images(augmentation, args, kwargs, to_float=to_float)

    figure = nodes.figure(align="center")
    for img in images:
      figure += nodes.image(uri=img)

    argstrings = []
    for arg in args:
      argstrings.append(str(arg))
    for argname, argval in kwargs.items():
      argstrings.append(f"{argname}={argval}")
    caption = f"Augmentation Examples for {augname}({', '.join(argstrings)})"
    figure += nodes.caption(text=caption, align="center")

    cls_result.insert(0, nodes.title(text=augname))
    cls_result.append(figure)

    entry = cls_result[2].children[0]
    section = nodes.section("", *cls_result, ids=entry["ids"], names=[augname])
    entry["ids"] = []

    return [section]


def setup(app):
  app.add_directive("autoaug", AutoAugmentation)
