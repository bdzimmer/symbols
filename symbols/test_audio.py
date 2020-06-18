"""

Test audio functionality.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import numpy as np

from symbols import audio


def test_convert():
    """test various conversions"""

    length = 16

    # convert to segment and back - mono
    array_mono = _get_sample(length)
    seg_mono = audio.array_to_segment(array_mono, 44100)
    array_converted = audio.segment_to_array(seg_mono)
    assert seg_mono.channels == 1
    assert np.alltrue(array_mono == array_converted)

    # convert to segment and back - stereo
    array_stereo = np.column_stack((array_mono, _get_sample(length)))
    seg_stereo = audio.array_to_segment(array_stereo, 44100)
    array_converted = audio.segment_to_array(seg_stereo)
    assert seg_stereo.channels == 2
    assert np.alltrue(array_stereo == array_converted)
    assert audio.stereo_to_mono(array_stereo).shape == (length,)

    # test converting to float and back
    assert np.alltrue(array_mono == audio.float_to_int16(audio.int16_to_float(array_mono)))
    assert np.alltrue(array_stereo == audio.float_to_int16(audio.int16_to_float(array_stereo)))


def test_paulstretch():
    """test paulstretch"""
    length = 10000
    sample = audio.int16_to_float(_get_sample(length))
    sample_stretched = audio.paulstretch(44100, sample, 2.0, 0.2, lambda x: x)
    assert sample_stretched.shape == (22050,)  # ~2*10000


def test_octave_down():
    """test frequency-domain octave downshift"""
    for size in [1024, 1025]:  # test for off-by one issues
        freqs = np.array(_get_sample(size) + 0.1, dtype=np.complex)
        freqs_shifted = audio.octave_down(freqs)
        assert np.where(freqs_shifted)[0][-1] == 511


def _get_sample(length: int) -> np.ndarray:
    """get a sample for testing"""
    np.random.seed(1)
    return np.array(
        np.random.random(length) * 2.0 * audio.MAX_FLOAT - audio.MAX_FLOAT,
        np.int16)
