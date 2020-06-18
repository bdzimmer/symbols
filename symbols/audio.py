"""

Audio effects not available in pydub.

"""

import sys

from typing import Callable, Optional
import numpy as np
from pydub import AudioSegment

MAX_INT = 2**15
MAX_FLOAT = 32767.0


def segment_to_array(segment: AudioSegment) -> np.ndarray:
    """convert segment to numpy array"""
    array = np.array(segment.get_array_of_samples())
    if segment.channels == 2:
        array = array.reshape((-1, 2))
    return array


def array_to_segment(array: np.ndarray, frame_rate: int) -> AudioSegment:
    """numpy array to segment"""
    channels = 2 if (array.ndim == 2 and array.shape[1] == 2) else 1
    segment = AudioSegment(
        array.tobytes(), frame_rate=frame_rate, sample_width=2, channels=channels)
    return segment


def int16_to_float(array: np.ndarray) -> np.ndarray:
    """convert int16 to float in range (-1, 1)"""
    return array / MAX_FLOAT


def float_to_int16(array: np.ndarray) -> np.ndarray:
    """convert float in range (-1, 1) to int16"""
    return np.array(array * MAX_FLOAT, dtype=np.int16)


def stereo_to_mono(array: np.ndarray) -> np.ndarray:
    """convert an array representing a stereo sound to mono"""
    return (array[:, 0] + array[:, 1]) * 0.5


def paulstretch(
        samplerate: int,
        smp: np.ndarray,
        stretch: float,
        windowsize_seconds: float,
        freq_func: Optional[Callable]) -> np.ndarray:

    """
    Refactored paulstretch algorithm based on paulstretch_mono.py (Public Domain)
    Source: https://github.com/paulnasca/paulstretch_python/blob/master/paulstretch_mono.py

    Takes a (-1,1) float array and returns a (-1,1) float array.

    freq_func allows additional operations on frequency domain represenation.

    """

    # pylint: disable=too-many-locals

    smp = np.copy(smp)

    # make sure that windowsize is even and larger than 16
    windowsize = int(windowsize_seconds * samplerate)
    if windowsize < 16:
        windowsize = 16
    windowsize = int(windowsize / 2) * 2
    half_windowsize = int(windowsize / 2)

    # correct the end of the smp
    end_size = int(samplerate * 0.05)
    if end_size < 16:
        end_size = 16
    smp[len(smp) - end_size:len(smp)] *= np.linspace(1, 0, end_size)

    # compute the displacement inside the input file
    start_pos = 0.0
    displace_pos = (windowsize * 0.5) / stretch

    # create Hann window
    window = 0.5 - np.cos(
        np.arange(windowsize, dtype=np.float) * 2.0 * np.pi / (windowsize - 1)) * 0.5

    old_windowed_buf = np.zeros(windowsize)
    hinv_sqrt2 = (1 + np.sqrt(0.5)) * 0.5
    hinv_buf = hinv_sqrt2 - (1.0 - hinv_sqrt2) * np.cos(
        np.arange(half_windowsize, dtype=np.float) * 2.0 * np.pi / half_windowsize)

    frames = []
    while True:

        # get the windowed buffer
        istart_pos = int(np.floor(start_pos))
        buf = smp[istart_pos:istart_pos + windowsize]
        if len(buf) < windowsize:
            buf = np.append(buf, np.zeros(windowsize - len(buf)))
        buf = buf * window

        # get the amplitudes of the frequency components and discard the phases
        freqs = abs(np.fft.rfft(buf))

        # randomize the phases by multiplication with a random complex number with modulus=1
        phases = np.random.uniform(0, 2 * np.pi, len(freqs)) * 1j
        freqs = freqs * np.exp(phases)

        if freq_func is not None:
            freqs = freq_func(freqs)

        # do the inverse FFT
        buf = np.fft.irfft(freqs)

        # window again the output buffer
        buf *= window

        # overlap-add the output
        output = buf[0:half_windowsize] + old_windowed_buf[half_windowsize:windowsize]
        old_windowed_buf = buf

        # remove the resulted amplitude modulation
        output *= hinv_buf

        # clamp the values to -1..1
        output[output > 1.0] = 1.0
        output[output < -1.0] = -1.0

        # get result
        frames.append(output)

        start_pos += displace_pos
        if start_pos >= len(smp):
            print("100 %")
            break
        sys.stdout.write("%d %% \r" % int(100.0 * start_pos / len(smp)))
        sys.stdout.flush()

    res = np.concatenate(frames, axis=0)

    return res


def octave_down(freqs: np.ndarray) -> np.ndarray:
    """shift a frequency-mode representation an octave down"""
    freqs_shifted = np.zeros(freqs.shape, dtype=np.complex)
    max_bins_shifted = freqs.shape[0] // 2
    max_bins = max_bins_shifted * 2
    freqs_shifted[0:max_bins_shifted] = (
        freqs[0:max_bins:2] +
        freqs[1:max_bins:2]) * 0.5
    return freqs_shifted
