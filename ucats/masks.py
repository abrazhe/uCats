import numpy as np
from numba import jit

from scipy import ndimage as ndi

import itertools as itt

from imfun import bwmorph, cluster


@jit
def percentile_th_frames(frames, plow=5):
    sh = frames[0].shape
    medians = np.median(frames, 0)
    out = np.zeros(medians.shape)
    for r in range(sh[0]):
        for c in range(sh[1]):
            v = frames[:, r, c]
            mu = medians[r, c]
            out[r, c] = -np.percentile(v[v <= mu], plow)
    return out


def refine_mask_by_percentile_filter(m,
                                     p=50,
                                     size=3,
                                     niter=1,
                                     with_cleanup=False,
                                     min_obj_size=2):
    out = np.copy(m).astype(bool)
    for i in range(niter):
        out += ndi.percentile_filter(out, p, size).astype(bool)
        if with_cleanup:
            out = threshold_object_size(out, min_obj_size)
    return out


def select_overlapping(mask, seeds):
    labels, nl = ndi.label(mask)
    objs = ndi.find_objects(labels)
    out = np.zeros_like(mask)
    for k, o in enumerate(objs):
        cond = labels[o] == k + 1
        if np.any(seeds[o][cond]):
            out[o][cond] = True
    return out


def threshold_object_size(mask, min_size):
    labels, nlab = ndi.label(mask)
    objs = ndi.find_objects(labels)
    out_mask = np.zeros_like(mask)
    for k, o in enumerate(objs):
        cond = labels[o] == (k + 1)
        if np.sum(cond) >= min_size:
            out_mask[o][cond] = True
    return out_mask


def opening_of_closing(m):
    return ndi.binary_opening(ndi.binary_closing(m))


def closing_of_opening(m, s=None):
    return ndi.binary_closing(ndi.binary_opening(m, s), s)


def locations(shape):
    """ all locations for a shape; substitutes nested cycles"""
    return itt.product(*map(range, shape))


def points2mask(points, sh):
    out = np.zeros(sh, np.bool)
    for p in points:
        out[tuple(p)] = True
    return out


def mask2points(mask):
    "mask to a list of points, as row,col"
    return np.array([loc for loc in locations(mask.shape) if mask[loc]])


from imfun import cluster


def cleanup_mask(m, eps=3, min_pts=5):
    if not np.any(m):
        return np.zeros_like(m)
    p = mask2points(m)
    # todo: convert to either HDBSCAN or sklearn DBSCAN
    _, _, labels = cluster.dbscan(p, eps, min_pts)
    points_f = (p for p, l in zip(p, labels) if l >= 0)
    return points2mask(points_f, m.shape)


@jit
def cleanup_cluster_map(m, niter=1):
    Nr, Nc = m.shape
    cval = np.min(m) - 1
    m = np.pad(m, 1, mode='constant', constant_values=cval)
    for j in range(niter):
        for r in range(1, Nr):
            for c in range(1, Nc):
                me = m[r, c]
                neighbors = np.array(
                    [m[(r + 1), c], m[(r - 1), c], m[r, (c + 1)], m[r, (c - 1)]])
                if not np.any(neighbors == me):
                    neighbors = neighbors[neighbors > cval]
                    if len(neighbors):
                        m[r, c] = neighbors[np.random.randint(len(neighbors))]
                    else:
                        m[r, c] = cval
    return m[1:-1, 1:-1]
