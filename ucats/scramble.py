"""
Scrambling and randomization of data
"""

import numpy as np
from numba import jit


@jit
def local_jitter(v, sigma=5):
    L = len(v)
    vx = np.copy(v)
    Wvx = np.zeros(L)
    for i in range(L):
        j = i + np.int(np.round(np.random.randn() * sigma))
        j = max(0, min(j, L - 1))
        vx[i] = v[j]
        vx[j] = v[i]
    return vx


def scramble_data(frames):
    """Randomly permute (shuffle) signals in each pixel independenly
    useful for quick creation of surrogate data with zero acitvity, only noise
    - TODO: clip some too high values
    """
    L, nr, nc = frames.shape
    out = np.zeros_like(frames)
    for r in range(nr):
        for c in range(nc):
            out[:, r, c] = np.random.permutation(frames[:, r, c])
    return out


def scramble_data_local_jitter(frames, w=10):
    """Randomly permute (shuffle) signals in each pixel independenly
    useful for quick creation of surrogate data with zero acitvity, only noise
    - TODO: clip some too high values
    """
    L, nr, nc = frames.shape
    out = np.zeros_like(frames)
    for r in range(nr):
        for c in range(nc):
            out[:, r, c] = local_jitter(frames[:, r, c], w)
    return out


def shuffle_signals(m):
    "Given a collection of signals, randomly permute each"
    return np.array([np.random.permutation(v) for v in m])
