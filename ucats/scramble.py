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

@jit
def local_jitter2d(img, sigma=3):
    nr,nc = img.shape
    imgx = np.copy(img)
    for r in range(nr):
        for c in range(nc):
            rj = r + np.int(np.round(np.random.randn() * sigma))
            cj = c + np.int(np.round(np.random.randn() * sigma))

            rj = max(0, min(rj, nr-1))
            cj = max(0, min(cj, nc-1))
            #rj = np.clip(rj, 0, nr-1)
            #cj = np.clip(rj, 0, nc-1)
            imgx[r,c] = img[rj,cj]
            imgx[rj,cj] = img[r,c]
    return imgx


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


def jitter_anti_aliasing(frames, niters=1, spatial_sigma=0.33, temporal_sigma=0.5, verbose=False):
    out = np.zeros_like(frames)
    for i in range(niters):
        spj = np.array([local_jitter2d(f, spatial_sigma) for f in frames], w=temporal_sigma)
        out += scramble_data_local_jitter(spj)
    return out/niters

def shuffle_signals(m):
    "Given a collection of signals, randomly permute each"
    return np.array([np.random.permutation(v) for v in m])
