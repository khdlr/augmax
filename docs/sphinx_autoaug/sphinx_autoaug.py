"""
Automatically document Augmax augmentations
including sample outputs
"""
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.ext.autodoc.directive import AutodocDirective

import jax
import json
import augmax
from imageio import imread, imwrite
from pathlib import Path

SEED = 42
N_IMGS = 3

def generate_images(augmentation, args, to_float: bool=False):
    basedir = Path(__file__).parent.parent.parent
    image = imread(basedir / 'docs' / 'teddy.png')
    keys = jax.random.split(jax.random.PRNGKey(SEED), N_IMGS)

    transform = augmentation(*args)
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

        images = generate_images(augmentation, args, to_float=to_float)

        figure = nodes.figure(align='center')
        for img in images:
            figure += nodes.image(uri=img)
        caption = f'Augmentation Examples for {augname}'
        if args:
            caption += '(' + ', '.join(map(str, args)) + ')'
        else:
            caption += ' with default parameters'
        figure += nodes.caption(text=caption, align='center')

        cls_result.append(figure)
        cls_result.insert(0, nodes.title(text=augname))

        section = nodes.section('', *cls_result, ids=[augname], names=[augname])

        return [section]


def setup(app):
    app.add_directive('autoaug', AutoAugmentation)
