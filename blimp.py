"""

BLIMP (Ben's Layertastic Image Manipulation Program)

"""

# Copyright (c) 2019 Ben Zimmer. All rights reserved.

import json
import os
import sys
import time

import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np

DEBUG = True
DEBUG_DIRNAME = "scratch"

if os.name == "nt":
    FONTS_DIRNAME = ""
else:
    FONTS_DIRNAME = os.path.expanduser("~/.fonts")


def main(argv):
    """main program"""

    # TODO: command line parameters
    save_layer = True
    save_total = False
    interactive = False

    # TODO: debug should be commandline parameter
    if DEBUG:
        os.makedirs(DEBUG_DIRNAME, exist_ok=True)

    input_filename = argv[1]
    project_dirname = os.path.splitext(input_filename)[0]

    os.makedirs(project_dirname, exist_ok=True)

    if interactive:

        cv2.namedWindow("final", cv2.WINDOW_NORMAL)

        while True:

            try:
                with open(input_filename, "r") as input_file:
                    config = json.load(input_file)
                canvas_width = config["width"]
                canvas_height = config["height"]
                resources_dirname = config["resources_dirname"]

                start_time = time.time()
                res = assemble_group(
                    config["layers"], canvas_width, canvas_height, resources_dirname, False,
                    save_layer, save_total, project_dirname, prefix="")

                end_time = time.time()
                total_time = end_time - start_time
                print("total time:", round(total_time, 3), "sec")

                res_cv2 = res[:, :, [2, 1, 0]]
                res_cv2 = cv2.resize(res_cv2, (int(canvas_width / 4), int(canvas_height / 4)))
                cv2.resizeWindow("final", (res_cv2.shape[1], res_cv2.shape[0]))
                cv2.imshow("final", res_cv2)
                cv2.waitKey(2000)
            except Exception as e:
                print(e)

    else:
        with open(input_filename, "r") as input_file:
            config = json.load(input_file)
        canvas_width = config["width"]
        canvas_height = config["height"]
        resources_dirname = config["resources_dirname"]

        start_time = time.time()
        res = assemble_group(
            config["layers"], canvas_width, canvas_height, resources_dirname, False,
            save_layer, save_total, project_dirname, prefix="")

        Image.fromarray(res[:, :, 0:3]).save(
            os.path.join(
                project_dirname, project_dirname + ".png"))

        if True:
            Image.fromarray(res[:, :, 0:3]).save(
                os.path.join(
                    project_dirname, project_dirname + ".jpg"))

        end_time = time.time()
        total_time = end_time - start_time
        print("total time:", round(total_time, 3), "sec")


def assemble_group(
        layers, canvas_width, canvas_height, resources_dirname, canvas_alpha,
        save_layer, save_total, project_dirname, prefix):

    start_time = time.time()

    if canvas_alpha:
        res = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
    else:
        res = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))

    for layer_idx, layer in enumerate(layers):
        print("layer", layer_idx, "...", end="", flush=True)

        # ~~~~ render

        layer_image = render_layer(
            layer, resources_dirname)

        # ~~~~ expand borders

        border_x = layer.get("border_x", 0)
        border_y = layer.get("border_y", 0)

        if border_x > 0 or border_y > 0:
            layer_image = expand_border(layer_image, border_x, border_y)

        # ~~~~ apply effects

        for effect_idx, effect in enumerate(layer.get("effects", [])):
            layer_image = apply_effect(layer_image, effect, resources_dirname)

        # ~~~~ apply mask

        mask_layer = layer.get("mask", None)
        if mask_layer is not None:
            mask_layer_image = render_layer(mask_layer, resources_dirname)
            layer_image[:, :, 3] = mask_layer_image[:, :, 3]

        # ~~~~ calculate position and trim

        # positions are calculated from inside the border
        layer_height, layer_width, _ = layer_image.shape
        layer_width = layer_width - 2 * border_x
        layer_height = layer_height - 2 * border_y

        # layer positions default to centered on the canvas
        layer_x = layer.get(
            "x", int(canvas_width * 0.5 - layer_width * 0.5))
        layer_y = layer.get(
            "y", int(canvas_height * 0.5 - layer_height * 0.5))

        # trim and update positions
        layer_image_trimmed, layer_x, layer_y = trim(
            layer_image, layer_x, layer_y, canvas_width, canvas_height)

        if save_layer:
            image_pil = Image.fromarray(layer_image_trimmed)
            image_pil.save(
                os.path.join(
                    project_dirname, "layer_" + prefix + str(layer_idx).rjust(3, "0") + ".png"))

        # ~~~~ apply opacity and accumulate

        opacity = layer.get("opacity", 1.0)
        if opacity < 1.0:
            print("blend...", end="", flush=True)
            layer_image_trimmed = blend(layer_image_trimmed, opacity)

        image_pil = Image.fromarray(layer_image_trimmed)
        # res.paste(image_pil, (layer_x - border_x, layer_y - border_y), image_pil)
        res.alpha_composite(image_pil, (layer_x - border_x, layer_y - border_y))

        if save_total:
            res.save(
                os.path.join(
                    project_dirname, "total_" + prefix + str(layer_idx).rjust(3, "0") + ".png"))

        cur_time = time.time() - start_time
        print("done", round(cur_time, 3), "sec")

    return np.array(res)


def text_custom_kerning(text, font, color, stroke_width, stroke_fill, kern_add):
    """text, controlling letter spacing"""

    letter_sizes = [font.getsize(x) for x in text]
    letter_offsets = [font.getoffset(x) for x in text]
    letter_pairs = [text[idx:(idx + 2)] for idx in range(len(text) - 1)]
    letter_pair_sizes = [font.getsize(x) for x in letter_pairs]
    letter_pair_offsets = [font.getoffset(x) for x in letter_pairs]

    # kerning "width" for a letter is width of pair
    # minus the width of the individual second letter
    widths = [
        (x[0] + z[0]) - (y[0] + w[0])
        for x, y, z, w in zip(
            letter_pair_sizes, letter_sizes[1:], letter_offsets[:-1], letter_offsets[1:])]

    # add width of final letter
    widths = widths + [letter_sizes[-1][0] + letter_offsets[-1][0]]

    # TODO: this is potentially unsafe - not sure about descenders etc
    # TODO: y offset?
    # find maximum height
    height = max([x[1] for x in letter_sizes])

    offset_x_first = letter_offsets[0][0]
    width_total = sum(widths) - offset_x_first + (len(widths) - 1) * kern_add

    if False and DEBUG:
        print(letter_pairs)
        print("ind widths:     ", [x[0] for x in letter_sizes])
        print("ind offsets:    ", [x[0] for x in letter_offsets])
        print("pair widths:    ", [x[0] for x in letter_pair_sizes])
        print("pair offsets:   ", [x[0] for x in letter_pair_offsets])
        print("true widths:    ", widths)
        print("sum ind. widths:", sum([x[0] for x in letter_sizes]))
        print("getsize width:  ", font.getsize(text)[0])
        print("width_total:    ", width_total)

    image = Image.new("RGBA", (width_total, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    offset = 0 - offset_x_first
    for letter, letter_width in zip(text, widths):
        draw.text(
            (offset, 0), letter,
            font=font, fill=color,
            stroke_width=stroke_width, stroke_fill=stroke_fill)
        offset = offset + letter_width + kern_add

    return image


def text_standard(text, font, color, stroke_width, stroke_fill):
    """standard text rendering"""
    size = font.getsize(text)
    offset = font.getoffset(text)
    image = Image.new("RGBA", (size[0], size[1]), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    # TODO: how to use offset here?
    draw.text(
        (0 - offset[0], 0),
        text, font=font, fill=color,
        stroke_width=stroke_width, stroke_fill=stroke_fill)
    return image


def render_layer(layer, resources_dirname):
    """render a single layer"""

    layer_type = layer["type"]

    if layer_type == "image":
        filename = layer["filename"]

        image = load_image(os.path.join(resources_dirname, filename))

    elif layer_type == "gaussian":
        width = layer["width"]
        height = layer["height"]
        a = layer.get("a", 255)
        mu_x = layer.get("mu_x", width * 0.5)
        mu_y = layer.get("mu_y", height * 0.5)
        sigma_x = layer.get("sigma_x", width * 0.75)
        sigma_y = layer.get("sigma_y", height * 0.75)
        transparent = layer.get("transparent", True)
        invert = layer.get("invert", False)

        d_x = 2.0 * sigma_x * sigma_x
        d_y = 2.0 * sigma_y * sigma_y
        ivals = np.tile(np.arange(height).reshape(-1, 1), (1, width))
        jvals = np.tile(np.arange(width), (height, 1))
        image = a * np.exp(
            0.0 - (((ivals - mu_y) ** 2) / d_y + ((jvals - mu_x) ** 2) / d_x))
        image = np.clip(image, 0.0, 255.0)
        image = np.array(image, dtype=np.ubyte)

        if transparent:
            black = np.zeros((height, width), dtype=np.ubyte)
            image = np.stack((black, black, black, 255 - image), axis=2)
        else:
            image = np.stack((image, image, image), axis=2)
            image = add_alpha(image)

        if invert:
            image = 255 - image

    elif layer_type == "text":
        font_filename = layer["font"]
        font_size = layer["size"]
        text = layer["text"]
        color = tuple(layer.get("color", (0, 0, 0, 255)))
        kern_add = layer.get("kern_add", 0)
        stroke_width = layer.get("stroke_width", 0)
        stroke_fill = layer.get("stroke_fill", (0, 0, 0, 255))

        font = load_font(font_filename, font_size)
        image_custom = text_custom_kerning(text, font, color, stroke_width, stroke_fill, kern_add)
        if DEBUG:
            image = text_standard(text, font, color, stroke_width, stroke_fill)
            image.save(os.path.join(DEBUG_DIRNAME, "text_" + text + "_true.png"))
            image_custom.save(os.path.join(DEBUG_DIRNAME, "text_" + text + "_custom.png"))
        image = np.array(image_custom)

    elif layer_type == "concat":
        axis = layer["axis"]
        sub_layers = layer["layers"]

        sub_layer_images = [render_layer(x, resources_dirname) for x in sub_layers]
        print("concatenating shapes", [x.shape for x in sub_layer_images], "...", end="", flush=True)
        expand_axis = 1 if axis == 0 else 0
        # add extra at the right or bottom
        # adding at bottom makes sense for text (main use case)
        max_dim = max([x.shape[expand_axis] for x in sub_layer_images])
        sub_layer_images_resized = []
        for sub_layer_image in sub_layer_images:
            if axis == 1:
                new_x = sub_layer_image.shape[1]
                new_y = max_dim
            else:
                new_x = max_dim
                new_y = sub_layer_image.shape[0]
            sub_layer_images_resized.append(expand_down_right(sub_layer_image, new_x, new_y))
        print([x.shape for x in sub_layer_images])
        image = np.concatenate(sub_layer_images_resized, axis=axis)

    elif layer_type == "group":
        sub_layers = layer["layers"]
        width = layer["width"]
        height = layer["height"]

        image = assemble_group(
            sub_layers, width, height, resources_dirname, True,
            False, False, None, None)

    else:
        image = None

    return image


def memoize(fn):
    """memoize a function"""
    cache = {}

    def call(*args):
        """wrapper"""
        res = cache.get(args)
        if res is None:
            res = fn(*args)
            cache[args] = res
        return res

    return call


@memoize
def load_font(filename, font_size):
    """load a font"""
    font_filename = os.path.join(FONTS_DIRNAME, filename)
    return ImageFont.truetype(font_filename, font_size)


@memoize
def load_image(filename):
    """load an image"""
    print("loading", filename, "from disk")
    image_pil = Image.open(filename)
    image = np.array(image_pil)
    if image.shape[2] == 3:
        image = add_alpha(image)
    return image


def add_alpha(image):
    """add an alpha channel to an image"""
    return np.concatenate(
        (image,
         np.ones((image.shape[0], image.shape[1], 1), dtype=np.ubyte) * 255),
        axis=2)


def trim(layer_image, layer_x, layer_y, canvas_width, canvas_height):
    """trim the layer to fit the canvas"""

    start_x = 0
    end_x = layer_image.shape[1]
    start_y = 0
    end_y = layer_image.shape[0]

    if layer_x < 0:
        start_x = 0 - layer_x
        layer_x = 0
    if layer_x + end_x > canvas_width:
        end_x = start_x + canvas_width

    if layer_y < 0:
        start_y = 0 - layer_y
        layer_y = 0
    if layer_y + end_y > canvas_height:
        end_y = start_y + canvas_height

    return layer_image[start_y:end_y, start_x:end_x, :], layer_x, layer_y


def apply_effect(image, effect, resources_dirname):
    """layer effects!"""
    effect_type = effect["type"]

    if effect_type == "flip_ud":
        image = np.array(Image.fromarray(image).transpose(Image.FLIP_TOP_BOTTOM))

    elif effect_type == "blend":
        opacity = effect["opacity"]
        image = blend(image, opacity)

    elif effect_type == "glow":
        dilate_size = effect.get("dilate", 16)
        blur_size = effect.get("blur", 127)
        color = tuple(effect.get("color", (0, 0, 0)))

        edge = cv2.Canny(image, 100, 200)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        edge = cv2.dilate(edge, kernel)
        edge = cv2.GaussianBlur(edge, (blur_size, blur_size), 0)
        color = np.array(color, dtype=np.ubyte)
        glow = np.tile(np.reshape(color, (1, 1, 3)), (image.shape[0], image.shape[1], 1))
        glow = np.concatenate((glow, np.expand_dims(edge, axis=2)), axis=2)
        glow = Image.fromarray(glow)
        glow.paste(Image.fromarray(image), (0, 0), Image.fromarray(image))
        image = np.array(glow)

    elif effect_type == "mask_onto":
        layer = dict(effect["layer"])

        layer["width"] = layer.get("width", image.shape[1])
        layer["height"] = layer.get("height", image.shape[0])
        effect_layer = render_layer(layer, resources_dirname)
        effect_layer[:, :, 3] = image[:, :, 3]
        image_pil = Image.fromarray(image)
        image_pil.alpha_composite(Image.fromarray(effect_layer))
        image = np.array(image_pil)

    elif effect_type == "scale":
        x_scale = effect.get("x", 1.0)
        y_scale = effect.get("y", 1.0)
        image = cv2.resize(
            image, (int(x_scale * image.shape[1]), int(y_scale * image.shape[0])))

    else:
        print("\tunrecognized effect type '" + str(effect_type) + "'")

    return image


def expand_border(image, border_x, border_y):
    """add a border to an image"""

    res = Image.new(
        "RGBA",
        (image.shape[1] + 2 * border_x, image.shape[0] + 2 * border_y),
        (255, 255, 255, 0))

    res = np.array(res)
    lim_y = res.shape[0] - border_y if border_y > 0 else res.shape[0]
    lim_x = res.shape[1] - border_x if border_x > 0 else res.shape[1]

    res[border_y:lim_y, border_x:lim_x] = image
    return res


def expand_down_right(image, new_x, new_y):
    res = Image.new("RGBA", (new_x, new_y), (255, 255, 255, 0))
    res = np.array(res)
    res[0:image.shape[0], 0:image.shape[1]] = image
    return res


def blend(image, opacity):
    """change the opacity of an image"""
    if image.shape[2] == 3:
        image = add_alpha(image)
    else:
        image = np.copy(image)
    # transparent = Image.new("RGBA", (image.shape[1], image.shape[0]), (255, 255, 255, 0))
    # return np.array(Image.blend(transparent, Image.fromarray(image), opacity))

    image[:, :, 3] = image[:, :, 3] * opacity
    return image


if __name__ == "__main__":
    main(sys.argv)
