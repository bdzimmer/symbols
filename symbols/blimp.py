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

from symbols import blimp_text

DEBUG = True
DEBUG_DIRNAME = "scratch"
DEBUG_GUIDES = True

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
                    save_layer, save_total, project_dirname, "",
                    DEBUG_GUIDES)

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
            save_layer, save_total, project_dirname, "",
            DEBUG_GUIDES)

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
        save_layer, save_total, project_dirname, prefix,
        debug_guides):

    start_time = time.time()

    if canvas_alpha:
        res = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
    else:
        res = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))

    for layer_idx, layer in enumerate(layers):
        print("layer", layer_idx, "...", end="", flush=True)

        # ~~~~ render

        layer_image = render_layer(layer, resources_dirname)

        # ~~~~ expand borders
        # (this requires some special logic depending on the layer type)

        layer_image, border_x, border_y = expand_border_layer(layer_image, layer)

        # ~~~~ apply effects

        for effect_idx, effect in enumerate(layer.get("effects", [])):
            layer_image = apply_effect(layer_image, effect, resources_dirname)

        # ~~~~ apply mask

        mask_layer = layer.get("mask", None)
        if mask_layer is not None:
            mask_layer_image = render_layer(mask_layer, resources_dirname)
            mask_layer_image, _, _ = expand_border_layer(mask_layer_image, mask_layer)
            layer_image[:, :, 3] = mask_layer_image[:, :, 3]

        # ~~~~ calculate position and trim

        # positions are calculated from inside the border
        # note that we couldn't have derived layer_height and layer_width
        # from the original layer image above due to some special
        # per-type border rules.
        layer_height, layer_width, _ = layer_image.shape
        layer_width = layer_width - 2 * border_x
        layer_height = layer_height - 2 * border_y

        # layer positions default to centered on the canvas
        layer_x = layer.get(
            "x", int(canvas_width * 0.5 - layer_width * 0.5))
        layer_y = layer.get(
            "y", int(canvas_height * 0.5 - layer_height * 0.5))

        # throughout, layer_x and layer_y remain the coordinates
        # of the logical layer (not including border)

        # however, this breaks down with the trim function below
        # and this is probably the source of some of the
        # out of bounds issues I've seen, since those only
        # seem to show up when a layer has borders.

        # trim and update positions
        # TODO: this needs to be updated to deal properly with border sizes
        # I think it's as simple as passing border_x and border_y in and adjusting
        # layer_x and layer_y accordingly
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
        res.alpha_composite(image_pil, (layer_x - border_x, layer_y - border_y))

        if debug_guides:
            # draw outer edges of borders in green and edges of
            # layer proper in black. Green lines should not be
            # visible if borders have size zero.

            draw = ImageDraw.Draw(res)

            def draw_v(x, color):
                """draw a vertical line at x"""
                draw.line([(x, 0), (x, res.size[1])], fill=color)

            def draw_h(y, color):
                """draw a horizontal line at y"""
                draw.line([(0, y), (res.size[0], y)], fill=color)

            # outer edges of borders
            draw_v(layer_x - border_x, (0, 255, 0))
            draw_v(layer_x + layer_width + border_x, (0, 255, 0))
            draw_h(layer_y - border_y, (0, 255, 0))
            draw_h(layer_y + layer_height + border_y, (0, 255, 0))
            # edges of layer proper
            draw_v(layer_x, (0, 0, 0))
            draw_v(layer_x + layer_width, (0, 0, 0))
            draw_h(layer_y, (0, 0, 0))
            draw_h(layer_y + layer_height, (0, 0, 0))

        if save_total:
            res.save(
                os.path.join(
                    project_dirname, "total_" + prefix + str(layer_idx).rjust(3, "0") + ".png"))

        cur_time = time.time() - start_time
        print("done", round(cur_time, 3), "sec")

    return np.array(res)


def text_custom_kerning(
        text, font, color, stroke_width, stroke_fill, kern_add,
        debug_guides):
    """text, controlling letter spacing"""

    letter_sizes = [blimp_text.getsize(font, x) for x in text]
    letter_offsets = [blimp_text.getoffset(font, x) for x in text]
    letter_pairs = [text[idx:(idx + 2)] for idx in range(len(text) - 1)]
    letter_pair_sizes = [blimp_text.getsize(font, x) for x in letter_pairs]
    letter_pair_offsets = [blimp_text.getoffset(font, x) for x in letter_pairs]

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
    height_total = height + stroke_width * 2

    # TODO: add stroke width logic to original text function

    # I think we can get rid of this initial offset
    # offset_x_first = letter_offsets[0][0]
    offset_x_first = 0
    width_total = (
        sum(widths)
        # - offset_x_first
        + (len(widths) - 1) * kern_add
        + stroke_width * 2  # Needs to be * 2 since it is added on all sides
    )

    if False and DEBUG:
        print(letter_pairs)
        print("ind widths:     ", [x[0] for x in letter_sizes])
        print("ind offsets:    ", [x[0] for x in letter_offsets])
        print("pair widths:    ", [x[0] for x in letter_pair_sizes])
        print("pair offsets:   ", [x[0] for x in letter_pair_offsets])
        print("true widths:    ", widths)
        print("sum ind. widths:", sum([x[0] for x in letter_sizes]))
        print("getsize width:  ", blimp_text.getsize(font, text)[0])
        print("stroke width:   ", stroke_width)
        print("width_total:    ", width_total)

    # improved font rendering
    # https://nedbatchelder.com/blog/200801/truly_transparent_text_with_pil.html

    # image = Image.new("RGBA", (width_total, height_total), (255, 255, 255, 0))
    alpha_text = Image.new("L", (width_total, height_total), 0)
    # draw_text = ImageDraw.Draw(alpha_text)

    alpha_stroke = Image.new("L", (width_total, height_total), 0)
    # draw_stroke = ImageDraw.Draw(alpha_stroke)

    offset = 0 - offset_x_first
    for letter, letter_width in zip(text, widths):
        # draw_text.text(
        #     xy=(offset, 0),
        #     text=letter,
        #     font=font,
        #     fill="white",
        #     stroke_width=stroke_width,
        #     stroke_fill="white")
        blimp_text.text(
            image=alpha_text,
            xy=(offset, 0),
            text_str=letter,
            font=font,
            fill=(255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(255, 255, 255))
        # draw_stroke.text(
        #     xy=(offset, 0),
        #     text=letter,
        #     font=font,
        #     fill="black",
        #     stroke_width=stroke_width,
        #     stroke_fill="white")
        blimp_text.text(
            image=alpha_stroke,
            xy=(offset, 0),
            text_str=letter,
            font=font,
            fill=(0, 0, 0),
            stroke_width=stroke_width,
            stroke_fill=(255, 255, 255))

        offset = offset + letter_width + kern_add

    # build text layer
    solid_text = Image.new("RGBA", (width_total, height_total), color)
    solid_text_np = np.array(solid_text)
    solid_text_np[:, :, 3] = solid_text_np[:, :, 3] * (np.array(alpha_text) / 255.0)
    # set all completely transparent pixels to (255, 255, 255, 0)
    solid_text_np[solid_text_np[:, :, 3] == 0, 0:3] = (255, 255, 255)

    image = Image.fromarray(solid_text_np)

    if stroke_width > 0:
        # build stroke layer
        solid_stroke = Image.new("RGBA", (width_total, height_total), stroke_fill)
        solid_stroke_np = np.array(solid_stroke)
        solid_stroke_np[:, :, 3] = solid_stroke_np[:, :, 3] * (np.array(alpha_stroke) / 255.0)
        # set all completely transparent pixels to (255, 255, 255, 0)
        solid_stroke_np[solid_stroke_np[:, :, 3] == 0, 0:3] = (255, 255, 255)

        # composite stroke onto text
        image = Image.alpha_composite(image, Image.fromarray(solid_stroke_np))

    if debug_guides:
        print("text offset:", offset)
        ascent, descent = blimp_text.getmetrics(font)
        draw = ImageDraw.Draw(image)
        # draw baseline
        draw.line([(0, ascent), (image.size[0] - 1, ascent)], fill=(0, 0, 0))
        # draw lines at edges of image
        # draw baseline
        draw.line([(0, 0), (image.size[0] - 1, 0)], fill=(0, 0, 0))
        draw.line([(0, image.size[1] - 1), (image.size[0] - 1, image.size[1] - 1)], fill=(0, 0, 0))
        draw.line([(0, 0), (0, image.size[1] - 1)], fill=(0, 0, 0))
        draw.line([(image.size[0] - 1, 0), (image.size[0] - 1, image.size[1] - 1)], fill=(0, 0, 0))

    # debugging
    # cv2.imwrite("text.png", solid_text_np)
    # cv2.imwrite("stroke.png", solid_stroke_np)

    return image


def text_standard(text, font, color, stroke_width, stroke_fill):
    """standard text rendering"""
    size = blimp_text.getsize(font, text)
    image = Image.new("RGBA", (size[0], size[1]), (255, 255, 255, 0))
    # draw = ImageDraw.Draw(image)
    # draw.text(
    #     xy=(0, 0),
    #     text=text,
    #     font=font,
    #     fill=color,
    #     stroke_width=stroke_width,
    #     stroke_fill=stroke_fill)
    blimp_text.text(
        image=image,
        xy=(0, 0),
        text_str=text,
        font=font,
        fill=color,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill)
    return image


def render_layer(layer, resources_dirname):
    """render a single layer"""

    layer_type = layer["type"]

    if layer_type == "image":
        filename = layer["filename"]
        print(resources_dirname)

        image = load_image(os.path.join(resources_dirname, filename))
        image = np.copy(image)

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
        image_custom = text_custom_kerning(
            text, font, color, stroke_width, stroke_fill, kern_add,
            DEBUG_GUIDES)
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
        width = layer.get("width")
        height = layer.get("height")

        image = assemble_group(
            sub_layers, width, height, resources_dirname, True,
            False, False, None, None,
            False)

    elif layer_type == "empty":
        height = layer.get("height")
        width = layer.get("width")
        image = np.ones((height, width, 4), dtype=np.uint8) * 255
        image[:, :, 3] = 0

    elif layer_type == "solid":
        height = layer.get("height")
        width = layer.get("width")
        color = tuple(layer.get("color", (0, 0, 0, 255)))
        image = np.ones((height, width, 4), dtype=np.uint8) * 255
        image[:, :, :] = color

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
        only = effect.get("only", False)

        edge = cv2.Canny(image, 100, 200)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        edge = cv2.dilate(edge, kernel)
        edge = cv2.GaussianBlur(edge, (blur_size, blur_size), 0)
        color = np.array(color, dtype=np.ubyte)
        glow = np.tile(np.reshape(color, (1, 1, 3)), (image.shape[0], image.shape[1], 1))
        # blurred edge becomes opacity
        glow = np.concatenate((glow, np.expand_dims(edge, axis=2)), axis=2)
        glow = Image.fromarray(glow)
        if not only:
            # glow.paste(Image.fromarray(image), (0, 0), Image.fromarray(image))
            glow.alpha_composite(Image.fromarray(image))
        image = np.array(glow)

    elif effect_type == "mask_onto":
        layer = dict(effect["layer"])

        layer["width"] = layer.get("width", image.shape[1])
        layer["height"] = layer.get("height", image.shape[0])
        effect_layer = render_layer(layer, resources_dirname)
        effect_layer, _, _ = expand_border_layer(effect_layer, layer)
        # trim in case the layer type doesn't respect width and height
        effect_layer = effect_layer[0:layer["height"], 0:layer["width"], :]
        effect_layer[:, :, 3] = image[:, :, 3]
        effect_layer[effect_layer[:, :, 3] == 0, 0:3] = (255, 255, 255)

        # print(effect_layer.shape, effect_layer.dtype)
        # cv2.imwrite("text_before_comp.png", image)

        # image_pil = Image.fromarray(image)
        # image_pil.alpha_composite(Image.fromarray(effect_layer))
        # image = np.array(image_pil)

        image = np.array(effect_layer)

    elif effect_type == "scale":
        x_scale = effect.get("x", 1.0)
        y_scale = effect.get("y", 1.0)
        image = cv2.resize(
            image, (int(x_scale * image.shape[1]), int(y_scale * image.shape[0])))

    else:
        print("\tunrecognized effect type '" + str(effect_type) + "'")

    return image


def expand_border_layer(layer_image, layer):
    """expand image borders with special layer logic"""

    border_x = layer.get("border_x", 0)
    border_y = layer.get("border_y", 0)

    if border_x > 0 or border_y > 0:
        # if we added stroke to a font, shrink the actual amount we
        # are expanding the border by to compensate for the increased size
        stroke_width = layer.get("stroke_width", 0)
        if layer["type"] == "text" and stroke_width > 0:
            layer_image = expand_border(
                layer_image, border_x - stroke_width, border_y - stroke_width)
        else:
            layer_image = expand_border(layer_image, border_x, border_y)

    return layer_image, border_x, border_y


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
