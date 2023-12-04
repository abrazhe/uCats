"""
Handling windowed views on data, aka "patches"
"""
import sys

import numpy as np

import itertools as itt
from numpy.linalg import svd
from multiprocessing import Pool
from scipy import ndimage as ndi

from imfun.core import coords

from ucats import cluster
from .globals import _dtype_


def make_grid(shape, sizes, overlaps):
    """Make a generator over sets of slices which go through the provided shape
       and overlap for the specified amount
    """
    if not np.iterable(sizes):
        sizes = (sizes, ) * len(shape)
    if not np.iterable(overlaps):
        overlaps = (overlaps, ) * len(shape)

    origins = itt.product(*[[*range(0, dim-size, size-overlap)] + [dim-size]
                            for (dim, size, overlap) in zip(shape, sizes, overlaps)])
    squares = tuple(
        tuple(slice(a, a + size) for a, size in zip(o, sizes)) for o in origins)
    return squares


def map_patches(fn, data, patch_size=10, overlap=5, tslice=slice(None), njobs=1):
    """
    Apply some function to a square patch exscized from video
    """
    sh = data.shape[1:]
    squares = [*map(tuple, make_grid(sh, patch_size, overlap))]
    if njobs > 1:
        pool = Pool(njobs)
        expl_m = pool.map(fn, (data[(tslice, ) + s] for s in squares))
    else:
        expl_m = [fn(data[(tslice, ) + s]) for s in squares]
    out = np.zeros(sh)
    counts = np.zeros(sh)
    for _e, s in zip(expl_m, squares):
        out[s] += _e
        counts[s] += 1.
    return out / counts


def slice_center_in_square(sl, sq):
    "test if center of a smaller n-dim slice is within a bigger n-dim slice"
    c = [(s.stop + s.start) * 0.5 for s in sl]
    return np.all([dim.start <= cx < dim.stop for cx, dim in zip(c, sq)])


def slice_overlaps_square(sl, sq):
    "test if a smaller n-dim slice overlaps with a bigger n-dim slice"
    return np.all([((dim.start <= s.start < dim.stop) or (dim.start <= s.stop < dim.stop))
                   for s, dim in zip(sl, sq)])


def slice_starts_in_square(sl, sq):
    "test if start of a smaller n-dim slice is within a bigger n-dim slice"
    o = [s.start for s in sl]
    return np.all([dim.start <= ox < dim.stop for ox, dim in zip(o, sq)])


def extract_random_cubic_patch(frames, w=10):
    """Extract small cubic patch at a random location from a stack of frames
    Parameters:
     - frames: TXY 3D array-like, a stack of frames
     - w : scalar int, side of the cubic patch [10]
    """
    sl = tuple()
    starts = (np.random.randint(0, dim - w) if dim-w > 0 else 0
              for dim in np.shape(frames))
    sl = tuple(slice(j, j + w) for j in starts)
    return frames[sl]


def extract_random_column(frames, w=10):
    if not np.iterable(w):
        w = (w, ) * np.ndim(frames)
    sh = frames.shape
    loc = tuple(np.random.randint(0, s - wi) if s-wi > 0 else 0
                for s, wi in zip(sh, w))
    sl = tuple(slice(j, j + wi) for j, wi in zip(loc, w))
    #print(loc, sl)
    return frames[sl]


import itertools as itt


def loc_in_patch(loc, patch):
    sl = patch[1]
    return np.all([s.start <= l < s.stop for l, s in zip(loc, sl)])


def patch_center(p):
    "center location of an n-dimensional slice"
    return np.array([0.5 * (p_.start + p_.stop) for p_ in p])


def make_weighting_kern(size, sigma=1.5):
    """
    Make a 2d array of floats to weight signal inputs in the spatial windows/patches
    """
    #size = patch_size_
    x, y = np.mgrid[-size / 2. + 0.5:size/2. + .5, -size / 2. + .5:size/2. + .5]
    g = np.exp(-(0.5 * (x / sigma)**2 + 0.5 * (y / sigma)**2))
    return g


def weight_counts(collection, sh):
    counts = np.zeros(sh)
    for v, s, w in collection:
        wx = w.reshape(counts[tuple(s)].shape)
        counts[s] += wx
    return counts


def signals_from_array_avg(data, overlap=2, patch_size=5):
    """Convert a TXY image stack to a list of temporal signals (taken from small spatial windows/patches)"""
    d = np.array(data).astype(_dtype_)
    acc = []
    squares = [*map(tuple, make_grid(d.shape[1:], patch_size, overlap))]
    w = make_weighting_kern(patch_size, 2.5)
    w = w / w.sum()
    #print('w.shape:', w.shape)
    #print(np.argmax(w.reshape(1,-1)))

    tslice = (slice(None), )
    for sq in squares:
        patch = d[tslice + sq]
        sh = patch.shape
        wclip = w[:sh[1], :sh[2]]
        #print(np.argmax(wclip))
        #print(w.shape, sh[1:3], wclip.shape)
        #wclip /= sum(wclip)
        signal = (patch * wclip).sum(axis=(1, 2))
        acc.append((signal, sq, wclip.reshape(1, -1)))
    return acc


def signals_from_array_pca_cluster(data,
                                   stride=2,
                                   nhood=3,
                                   ncomp=2,
                                   pre_smooth=1,
                                   dbscan_eps_p=10,
                                   dbscan_minpts=3,
                                   cluster_minsize=5,
                                   walpha=1.0,
                                   mask_of_interest=None):
    """
    Convert a TXY image stack to a list of signals taken from spatial windows and aggregated according to their coherence
    """
    sh = data.shape
    if mask_of_interest is None:
        mask_of_interest = np.ones(sh[1:], dtype=np.bool)
    mask = mask_of_interest
    counts = np.zeros(sh[1:])
    acc = []
    knn_count = [0]
    cluster_count = [0]
    Ln = (2*nhood + 1)**2
    corrfn = stats.pearsonr
    patch_size = (nhood*2 + 1)**2
    if cluster_minsize > patch_size:
        cluster_minsize = patch_size
    #dbscan_eps_acc = []
    def _process_loc(r, c):
        kcenter = 2 * nhood * (nhood+1)
        sl = (slice(r - nhood, r + nhood + 1), slice(c - nhood, c + nhood + 1))
        patch = data[(slice(None), ) + sl]
        if not np.any(patch):
            return
        patch = patch.reshape(sh[0], -1).T
        patch0 = patch.copy()
        if pre_smooth > 1:
            patch = ndi.median_filter(patch, size=(pre_smooth, 1))
        Xc = patch.mean(0)
        u, s, vh = np.linalg.svd(patch - Xc, full_matrices=False)
        points = u[:, :ncomp]
        #dists = cluster.metrics.euclidean(points[kcenter],points)
        all_dists = cluster.dbscan_._pairwise_euclidean_distances(points)
        dists = all_dists[kcenter]

        max_same = np.max(np.diag(all_dists))

        #np.mean(dists)
        dbscan_eps = np.percentile(all_dists[all_dists > max_same], dbscan_eps_p)
        #dbscan_eps_acc.append(dbscan_eps)
        #print(r,c,':', dbscan_eps)
        _, _, affs = cluster.dbscan(points,
                                    dbscan_eps,
                                    dbscan_minpts,
                                    distances=all_dists)
        similar = affs == affs[kcenter]

        if sum(similar) < cluster_minsize or affs[kcenter] == -1:
            knn_count[0] += 1
            #th = min(np.argsort(dists)[cluster_minsize+1],2*dbscan_eps)
            th = dists[np.argsort(dists)[min(len(dists), cluster_minsize * 2)]]
            similar = dists <= max(th, max_same)
            #print('knn similar:', np.sum(similar), 'total signals:', len(similar))
            #dists *= 2  # shrink weights if not from cluster
        else:
            cluster_count[0] += 1

        weights = np.exp(-walpha * dists)
        #weights = np.array([corrfn(a,v)[0] for a in patch])**2

        #weights /= np.sum(weights)
        #weights = ones(len(dists))
        weights[~similar] = 0
        #weights = np.array([corrfn(a,v)[0] for a in patch])

        #weights /= np.sum(weights)
        vx = patch0[similar].mean(0)    # DONE?: weighted aggregate
        # TODO: check how weights are defined in NL-Bayes and BM3D
        # TODO: project to PCs?
        acc.append((vx, sl, weights))
        return    #  _process_loc

    for r in range(nhood, sh[1] - nhood, stride):
        for c in range(nhood, sh[2] - nhood, stride):
            sys.stderr.write('\r processing location %05d/%d ' %
                             (r * sh[1] + c + 1, np.prod(sh[1:])))
            if mask[r, c]:
                _process_loc(r, c)

    sys.stderr.write('\n')
    print('KNN:', knn_count[0])
    print('cluster:', cluster_count[0])
    m = weight_counts(acc, sh[1:])
    #print('counted %d holes'%np.sum(m==0))
    nholes = np.sum((m == 0) * mask)
    #print('N holes:', nholes)
    #print('acc len before:', len(acc))
    hole_i = 0
    for r in range(nhood, sh[1] - nhood):
        for c in range(nhood, sh[2] - nhood):
            if mask[r, c] and (m[r, c] < 1e-6):
                sys.stderr.write('\r processing additional location %05d/%05d ' %
                                 (hole_i, nholes))
                _process_loc(r, c)
                #v = data[:,r,c]
                #sl = (slice(r-1,r+1+1), slice(c-1,c+1+1))
                #weights = np.zeros((3,3))
                #weights[1,1] = 1.0
                #acc.append((v, sl, weights.ravel()))
                hole_i += 1
    #print('acc len after:', len(acc))
    #print('DBSCAN eps:', np.mean(dbscan_eps_acc), np.std(dbscan_eps_acc))
    return acc


from scipy import stats


def signals_from_array_correlation(data,
                                   stride=2,
                                   nhood=5,
                                   max_take=10,
                                   corrfn=stats.pearsonr,
                                   mask_of_interest=None):
    """
    Convert a TXY image stack to a list of signals taken from spatial windows and aggregated according to their coherence
    """
    sh = data.shape
    L = sh[0]
    if mask_of_interest is None:
        mask_of_interest = np.ones(sh[1:], dtype=np.bool)
    mask = mask_of_interest
    counts = np.zeros(sh[1:])
    acc = []
    knn_count = 0
    cluster_count = 0
    Ln = (2*nhood + 1)**2
    max_take = min(max_take, Ln)

    def _process_loc(r, c):
        v = data[:, r, c]
        kcenter = 2 * nhood * (nhood+1)
        sl = (slice(r - nhood, r + nhood + 1), slice(c - nhood, c + nhood + 1))
        patch = data[(slice(None), ) + sl]
        if not np.any(patch):
            return
        patch = patch.reshape(sh[0], -1).T
        weights = np.array([corrfn(a, v)[0] for a in patch])
        weights[weights < 2 /
                L**0.5] = 0    # set weights to 0 in statistically independent sources
        weights[np.argsort(weights)[:-max_take]] = 0
        weights = weights / np.sum(weights)    # normalize weights
        weights += 1e-6    # add small weight to avoid dividing by zero
        vx = (patch * weights.reshape(-1, 1)).sum(0)
        acc.append((vx, sl, weights))

    for r in range(nhood, sh[1] - nhood, stride):
        for c in range(nhood, sh[2] - nhood, stride):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d' %
                             (r, c, r * sh[1] + c + 1, np.prod(sh[1:])))
            if mask[r, c]:
                _process_loc(r, c)
    for _, sl, w in acc:
        counts[sl] += w.reshape(2*nhood + 1, 2*nhood + 1)
    for r in range(nhood, sh[1] - nhood):
        for c in range(nhood, sh[2] - nhood):
            if mask[r, c] and not counts[r, c]:
                sys.stderr.write('\r (2x) processing location (%03d,%03d), %05d/%d' %
                                 (r, c, r * sh[1] + c + 1, np.prod(sh[1:])))
                _process_loc(r, c)
    return acc


def combine_weighted_signals(collection, shape):
    """
    Combine a list of processed signals with weights back into TXY frame stack (nframes x nrows x ncolumns)
    """
    out_data = np.zeros(shape, dtype=_dtype_)
    counts = np.zeros(shape[1:])
    tslice = (slice(None), )
    i = 0
    for v, s, w in collection:
        pn = s[0].stop - s[0].start
        #print(s,len(w))
        wx = w.reshape(out_data[tslice + tuple(s)].shape[1:])
        out_data[tslice + tuple(s)] += v.reshape(-1, 1, 1) * wx
        counts[s] += wx
    out_data /= (1e-12 + counts)
    return out_data
