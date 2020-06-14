"""

Utilities for using blimp programatically

"""

from symbols import blimp

from PIL import Image
import numpy as np


def render_and_composite(layers, resources_dirname, im_bg):
    """render a layer, apply effects, and composite against a background image"""

    blimp.DEBUG = False

    canvas_height, canvas_width, _ = im_bg.shape

    canvas_layer = {"type": "empty", "width": canvas_width, "height": canvas_height}

    # this is redundant
    im_bg_chunk = im_bg[0:canvas_height, 0:canvas_width]

    im_layer = blimp.assemble_group(
        [canvas_layer] + layers,
        canvas_width, canvas_height,
        resources_dirname, True, False, False, None, None,
        [])

    # composite
    im_comp = Image.fromarray(im_bg_chunk)
    im_comp.alpha_composite(Image.fromarray(im_layer))
    im_comp = np.array(im_comp)

    # TODO: add some vertical lines to ensure that things are centered properly
    # TODO: seeing something like a one-pixel offset issue with glow

    return im_layer, im_comp
