import numpy as np
import numpy.typing as npt
from typing import Tuple


def shuffle_along_axis(
    spikes: npt.ArrayLike, axis: int, random_state: int
) -> npt.ArrayLike:
    np.random.seed(random_state)
    idx = np.random.rand(*spikes.shape).argsort(axis=axis)
    return np.take_along_axis(spikes, idx, axis=axis)


def calc_randomized_shift_spikes(
    spikes: npt.ArrayLike, random_state: int, axis: int = 0
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    np.random.seed(random_state)
    nt, nc = spikes.shape
    shifted_spikes = np.zeros((nt, nc), dtype=np.int32)
    shifts = np.zeros(nc, dtype=np.int32)
    for cell in range(nc):
        shift = np.random.randint(0, nt - 1)
        shifted_spikes[:, cell] = calc_shifted_spikes(spikes[:, cell], shift, axis=axis)
        shifts[cell] = shift
    return shifted_spikes, shifts


def calc_shifted_spikes(
    spikes: npt.ArrayLike, shift: int, axis: int = 0
) -> npt.ArrayLike:
    shifted_spikes = np.roll(spikes, shift, axis=axis)
    return shifted_spikes
