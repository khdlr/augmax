"""
Automatically document Augmax augmentations
including sample outputs
"""
from docutils import nodes
from sphinx.ext.autodoc.directive import AutodocDirective

import jax
import json
import augmax
from imageio import imread, imwrite
from pathlib import Path

import inspect

SEED = 42
N_IMGS = 3

def generate_images(augmentation, args, kwargs={}, to_float: bool=False):
    basedir = Path(__file__).parent.parent.parent
    image = imread(basedir / 'docs' / 'teddy.png')
    keys = jax.random.split(jax.random.PRNGKey(SEED), N_IMGS)

    transform = augmentation(*args, **kwargs)
    if to_float:
        transform = augmax.Chain(augmax.ByteToFloat(), transform)

    transform = jax.jit(jax.vmap(transform.apply, (None, 0)))
    images = transform(image, keys)

    imgdir = Path(basedir / 'docs' / 'generated_imgs').relative_to(basedir).absolute()
    imgnames = []
    for i in range(N_IMGS):
        imgname = str(imgdir / f'{augmentation.__name__}_{i}.png')
        imwrite(imgname, images[i])
        imgnames.append('/' + imgname)
    return imgnames


class AutoAugmentation(AutodocDirective):
    required_arguments = 1
    optional_arguments = 10

    def run(self):
        # Leverage autoclass for the base documentation.
        # To do so, we have to go incognito and change our name... ;)
        self.name = 'autoclass'
        augname, *args = self.arguments
        args = json.loads('[' + ', '.join(args) + ']')
        augmentation = getattr(augmax, augname)
        self.arguments = [augname]
        cls_result = super().run()

        to_float = False
        if args and args[0] == 'flt':
            args = args[1:]
            to_float = True

        kwargs = {}
        if 'p' in inspect.getfullargspec(augmentation)[0]:
            kwargs['p'] = 1.0

        images = generate_images(augmentation, args, kwargs, to_float=to_float)

        figure = nodes.figure(align='center')
        for img in images:
            figure += nodes.image(uri=img)

        argstrings = []
        for arg in args:
            argstrings.append(str(arg))
        for argname, argval in kwargs.items():
            argstrings.append(f'{argname}={argval}')
        caption = f'Augmentation Examples for {augname}({", ".join(argstrings)})'
        figure += nodes.caption(text=caption, align='center')

        entry = cls_result[1].children[0]
        if 'ids' not in entry:
            print(entry.pformat())
        cls_result.insert(1, nodes.title(text=augname, ids=entry['ids']))
        entry['ids'] = []
        cls_result.append(figure)

        section = nodes.section('', *cls_result, ids=[augname], names=[augname])

        return [section]


def setup(app):
    app.add_directive('autoaug', AutoAugmentation)
