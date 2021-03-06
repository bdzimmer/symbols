"""

Utilities.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.


from collections import OrderedDict
import os
import time

import cv2
from trimesh import transformations
import numpy as np


X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])


def rotation_x(angle):
    """rotation around x axis"""
    return transformations.rotation_matrix(angle, X_HAT)


def rotation_y(angle):
    """rotation around y axis"""
    return transformations.rotation_matrix(angle, Y_HAT)


def rotation_z(angle):
    """rotation around z axis"""
    return transformations.rotation_matrix(angle, Z_HAT)


class Profiler:
    """simple class for totalling up times"""

    def __init__(self):
        """create a collection of timers"""
        self.times = OrderedDict()

    def tick(self, key):
        """start a timer"""
        _, total_time = self.times.setdefault(key, (None, 0.0))
        self.times[key] = (time.time(), total_time)

    def tock(self, key):
        """stop a timer"""
        end_time = time.time()
        start_time, total_time = self.times.setdefault(key, (end_time, 0.0))
        self.times[key] = (None, total_time + (end_time - start_time))

    def summary(self):
        """pretty-print results"""
        times = [(x, y) for x, (_, y) in self.times.items()]
        key_len_max = max(len(x) for x in self.times.keys())
        for key, dur in times:
            print(key.ljust(key_len_max + 4, ".") + str(round(dur, 3)))


def ffmpeg_command(images_dirname, output_filename, width, height, fps):
    """prepare a command for ffmpeg"""
    command = (
        "ffmpeg -y -r " + str(fps) +
        " -f image2 -s " + str(width) + "x" + str(height) +
        " -i " + images_dirname + "/%05d.png " +
        "-threads 2 -vcodec libx264 -crf 25 -pix_fmt yuv420p " + output_filename)
    return command


def build_frame_writer(output_dirname, prof):
    """frame writer"""

    idx = [0]

    def write(img: np.ndarray):
        """write and increment counter"""
        if prof is not None:
            prof.tick("write")

        if img.shape[2] == 3:
            swizzled = img[:, :, [2, 1, 0]]
        else:
            swizzled = img[:, :, [2, 1, 0, 3]]
        cv2.imwrite(
            os.path.join(output_dirname, str(idx[0]).rjust(5, "0") + ".png"),
            swizzled)
        idx[0] = idx[0] + 1
        if prof is not None:
            prof.tock("write")

    return write
