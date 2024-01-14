import numpy as np
from numba import jit

from scipy import ndimage as ndi

import itertools as itt

from imfun import bwmorph, cluster


@jit(nopython=True)
def percentile_th_frames(frames, plow=5):
    sh = frames[0].shape
    #medians = np.median(frames, axis=0)
    out = np.zeros(sh)
    for r in range(sh[0]):
        for c in range(sh[1]):
            v = frames[:, r, c]
            #mu = medians[r, c]
            mu = np.median(v)
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


def select_overlapping(mask, seeds, neg=False):
    labels, nl = ndi.label(mask)
    objs = ndi.find_objects(labels)
    out = np.zeros_like(mask)
    for k, o in enumerate(objs):
        overlap = labels[o] == k + 1
        cond = np.any(seeds[o][overlap])
        if neg:
            cond = not cond
        if cond:
            out[o][overlap] = True
    return out

def get_object_sizes(mask, is_labels=False):
    if not is_labels:
        labels, nlab = ndi.label(mask)
    else:
        labels = mask
    objs = ndi.find_objects(labels)
    out = [np.sum(labels[o]==k+1) for k,o in enumerate(objs)]
    return out

def n_largest_regions(mask,  n=1):
    "get  N largest contiguous region from a binary mask"
    labels, _ = ndi.label(mask)
    sizes = get_object_sizes(labels, is_labels=True)
    ksort = np.argsort(sizes)[::-1]
    return np.sum([labels == k+1 for k in ksort[:n]],0).astype(bool)

largest_region = n_largest_regions

# def largest_region(mask):
#     labels, nlab = ndi.label(mask)
#     if nlab > 0:
#         objs = ndi.find_objects(labels)
#         sizes = [np.sum(labels[o]==k+1) for k,o in enumerate(objs)]
#         k = np.argmax(sizes)
#         return labels==k+1
#     else:
#         return mask

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

def cleanup_mask(m, eps=3, min_pts=5):
    if not np.any(m):
        return np.zeros_like(m)
    p = mask2points(m)
    # todo: convert to either HDBSCAN or sklearn DBSCAN
    _, _, labels = cluster.dbscan(p, eps, min_pts)
    points_f = (p for p, l in zip(p, labels) if l >= 0)
    return points2mask(points_f, m.shape)

def correct_small_loads(points, affs, min_loads=5, niter=1):
    for j in range(niter):
        new_affs = np.copy(affs)
        labels = np.unique(affs)
        loads = np.array([np.sum(affs == k) for k in labels])
        if not np.any(loads < min_loads):
            break
        centers = np.array([np.mean(points[affs == k], 0) for k in labels])
        point_ind = np.arange(len(points))
        for li in np.where(loads < min_loads)[0]:
            cond = affs == labels[li]
            for point, ind in zip(points[cond], point_ind[cond]):
                dists = cluster.metrics.euclidean(point, centers)
                dists[loads < min_loads] = np.amax(dists) + 1000
                k = np.argmin(dists)
                new_affs[ind] = k

        affs = new_affs.copy()
    return new_affs

def cleanup_cluster_map(m, min_neighbors=1, niter=1):
    Nr, Nc = m.shape
    cval = np.min(m) - 1
    m = np.pad(m, 1, mode='constant', constant_values=cval)
    @jit(nopython=True)
    def _process(m):
        for j in range(niter):
            for r in range(1, Nr):
                for c in range(1, Nc):
                    me = m[r, c]
                    neighbors = np.array(
                        [m[(r + 1), c], m[(r - 1), c], m[r, (c + 1)], m[r, (c - 1)]])
                    if not np.sum(neighbors == me) >= min_neighbors:
                        neighbors = neighbors[neighbors > cval]
                        if len(neighbors):
                            m[r, c] = neighbors[np.random.randint(len(neighbors))]
                        else:
                            m[r, c] = cval
        return m
    m = _process(m)
    return m[1:-1, 1:-1]
