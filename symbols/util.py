"""

Utilities.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.


from collections import OrderedDict
import time

from trimesh import transformations
import numpy as np


X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])


# TODO: write my own rather than using trimesh

def rotation_x(angle):
    return transformations.rotation_matrix(angle, X_HAT)


def rotation_y(angle):
    return transformations.rotation_matrix(angle, Y_HAT)


def rotation_z(angle):
    return transformations.rotation_matrix(angle, Z_HAT)


class Profiler:
    """simple class for totalling up times"""
    def __init__(self):
        self.times = OrderedDict()

    def tick(self, key):
        start_time, total_time = self.times.setdefault(key, (None, 0.0))
        self.times[key] = (time.time(), total_time)

    def tock(self, key):
        end_time = time.time()
        start_time, total_time = self.times.setdefault(key, (end_time, 0.0))
        self.times[key] = (None, total_time + (end_time - start_time))


def ffmpeg_command(images_dirname, output_filename, width, height, fps):
    """prepare a command for ffmpeg"""
    command = (
        "ffmpeg -y -r " + str(fps) +
        " -f image2 -s " + str(width) + "x" + str(height) +
        " -i " + images_dirname + "/%05d.png " +
        "-threads 2 -vcodec libx264 -crf 25 -pix_fmt yuv420p " + output_filename)
    return command
