import gzip
import pickle

import numpy as np
from numpy.linalg import svd
from sklearn import cluster as skcluster
from tqdm import tqdm

from .. import baselines as bl
from .. import scramble
from ..decomposition import min_ncomp, svd_flip_signs
from ..globals import _dtype_
from ..patches import make_grid, slice_overlaps_square
from ..utils import adaptive_filter_1d, adaptive_filter_2d

_baseline_smoothness_ = 300
_nclusters_ = 32
_do_pruning_ = False
_do_scrambling_ = False


# TODO:
# - [ ] Exponential-family PCA
# - [ ] Cut svd_signals into pieces before second-stage SVD
#       - alternatively, look at neighboring patches in time
# - [X] (***) Cluster svd_signals before SVD

# TODO: decide which clustering algorithm to use.
#       candidates:
#         - KMeans (sklearn)
#         - MiniBatchKMeans (sklearn)
#         - AgglomerativeClustering with Ward or other linkage (sklearn)
#         - SOM aka Kohonen (imfun)
#         - something else?
#       - clustering algorithm should be made a parameter
# TODO: good crossfade and smaller overlap


def weight_components(data, components, rank=None, Npermutations=100, clip_percentile=95):
    """
    For a collection of signals (each row of input matrix is a signal),
    try to decide if using projection to the principal or svd components should describes
    the original signals better than time-scrambled signals. Returns a binary vector of weights

    Parameters:
     - data: (Nsignals,Nfeatures) matrix. Each row is one signal
     - compoments: temporal principal components
     - rank: number of first PCs to use
     - Npermutations: how many permutations to try (default: 100)
     - clip_percentile: P, if a signal is better represented than P% of scrambled signals,
                        the weight for this signal is 1 (default: P=95)
    Returns:
     - vector of weights (Nsignals,)
    """
    v_shuffled = (scramble.shuffle_signals(components[:rank])
                  for i in range(Npermutations))
    coefs_randomized = np.array([np.abs(data @ vt.T).T for vt in v_shuffled])
    coefs_orig = np.abs(data @ components[:rank].T).T
    w = np.zeros((len(data), len(components[:rank])), _dtype_)
    for j in np.arange(w.shape[1]):
        w[:, j] = coefs_orig[j] >= np.percentile(coefs_randomized[:, j, :], clip_percentile, axis=0)
    return w


def tsvd_rec_with_weighting(data, rank=None):
    """Do truncated SVD approximation using coefficient pruning by comparisons to shuffled data
    Input: data matrix (Nsamples, Nfeatures), each row is interpreted as a signal or feature vector
    Output: approximated data using rank-truncated SVD
    """
    dc = data.mean(1)[:, None]
    u, s, vh = svd(data - dc, False)
    if rank is None:
        rank = min_ncomp(s, data.shape) + 1
    w = weight_components(data - dc, vh, rank)
    return (u[:, :rank] * w) @ np.diag(s[:rank]) @ vh[:rank] + dc


def patch_tsvds_from_frames(frames,
                            patch_ssize=10, patch_tsize=600,
                            sstride=2, tstride=300, min_ncomps=1,
                            do_pruning=_do_pruning_,
                            tsmooth=0, ssmooth=0):
    """
    Slide a rectangle spatial window and extract local dynamics in this window (patch),
    then do truncated SVD decomposition of the local dynamics for each patch.

    Input:
     - frames: a TXY 3D stack of frames (array-like)
     - stride: spacing between windows (default: 2 px)
     - patch_size: spatial size of the window (default: 10 px)
     - min_ncomps: minimal number of components to retain
     - do_pruning: whether to do coefficient pruning based on comparison to shuffled signals
     - tsmooth: scalar, if > 0, smooth temporal components with adaptive median filter of this size
     - ssmmooth: scalar, if > 0, smooth spatial compoments with adaptive median filter of this size

    Output:
     - list of tuples of the form:
       (temporal components, spatial components, patch average, location of the patch in data, patch shape)
    """
    d = np.array(frames).astype(_dtype_)
    acc = []
    # squares =  list(map(tuple, make_grid(d.shape[1:], patch_size,stride)))
    L = len(frames)
    patch_tsize = min(L, patch_tsize)
    if tstride > patch_tsize:
        tstride = patch_tsize // 2
    tstride = min(L, tstride)
    squares = make_grid(frames.shape, (patch_tsize, patch_ssize, patch_ssize), (tstride, sstride, sstride))
    if tsmooth > 0:
        # print('Will smooth temporal components')
        # smoother = lambda v: smoothed_medianf(v, tsmooth*0.5, tsmooth)
        tsmoother = lambda v: adaptive_filter_1d(v, th=3, smooth=tsmooth, keep_clusters=False)
    if ssmooth > 0:
        ssmoother = lambda v: adaptive_filter_2d(v.reshape(patch_ssize, -1), smooth=ssmooth,
            keep_clusters=False).reshape(v.shape)

    # print('Splitting to patches and doing SVD decompositions',flush=True)
    for sq in tqdm(squares, desc='Splitting to patches and doing SVD'):

        patch_frames = d[sq]
        L = len(patch_frames)
        w_sh = patch_frames.shape

        # print(sq, w_sh, L)

        patch = patch_frames.reshape(L, -1)  # now each column is signal in one pixel
        patch_c = np.mean(patch, 0)
        patch = patch - patch_c

        u, s, vh = np.linalg.svd(patch.T, full_matrices=False)
        # rank = min_ncomp(s, patch.shape)+1
        rank = max(min_ncomps, min_ncomp(s, patch.shape) + 1)
        u, vh = svd_flip_signs(u[:, :rank], vh[:rank])

        w = weight_components(patch.T, vh, rank) if do_pruning else np.ones(u[:, :rank].shape)
        svd_signals, loadings = vh[:rank], u[:, :rank] * w
        s = s[:rank]
        svd_signals = svd_signals * s[:, None]

        if tsmooth > 0:
            svd_signals = np.array([tsmoother(v) for v in svd_signals])
        # W = np.diag(s)@vh
        W = loadings.T
        if (ssmooth > 0) and (patch.shape[1] == patch_ssize ** 2):
            W = np.array([ssmoother(v) for v in W])
        # print (svd_signals.shape, W.shape, patch.shape)
        # return
        acc.append((svd_signals, W, patch_c, sq, w_sh))
    return acc


def tanh_step(x, window):
    taper_width = window / 5
    taper_k = taper_width / 4
    return np.clip((1.01 + np.tanh((x - taper_width) / taper_k)
                    * np.tanh((window - x - taper_width) / taper_k)) / 2, 0, 1)


## TODO: crossfade for overlapping patches, at least in time
def project_from_tsvd_patches(collection, shape, with_f0=False, baseline_smoothness=_baseline_smoothness_):
    """
    Do inverse transform from local truncated SVD projections (output of `patch_tsvds_from_frames`)
    Goes through patches, calculates approximations and combines overlapping singal estimates

    Input:
     - collection of tSVD components along with patch location and shape (see `patch_tsvds_from_frames`)
     - shape of the full frame stack to reconstruct
     - with_f0: whether to calculate an estimate of baseline fluorescence level F0
     - baseline_smoothness: filter width for smoothing to calculate the baseline, has no effect of with_f0 is False

    Output:
     - if `with_f0` is False, just return the approximation of the fluorescence signal (1 frame stack)
     - if `with_f0` is True, return estimates of fluorescence and baseline fluorescence (2 frame stacks)
    """
    out_data = np.zeros(shape, dtype=_dtype_)
    if with_f0:
        out_f0 = np.zeros_like(out_data)
    # counts = np.zeros(shape[1:], np.int)
    counts = np.zeros(shape, _dtype_)  # candidate for crossfade

    # tslice = (slice(None),)
    i = 0
    # print('Doing inverse transform', flush=True)
    tqdm_desc = 'Doing inverse transform ' + ('with baseline' if with_f0 else '')
    for signals, filters, center, sq, w_sh in tqdm(collection, desc=tqdm_desc):
        L = w_sh[0]
        crossfade_coefs = tanh_step(np.arange(L), L).astype(_dtype_)[:, None, None]
        # crossfade_coefs = np.ones(L)[:,None,None]
        counts[sq] += crossfade_coefs

        rec = (signals.T @ filters).reshape(w_sh)
        out_data[tuple(sq)] += (rec + center.reshape(w_sh[1:])) * crossfade_coefs

        if with_f0:
            # bs = np.array([simple_baseline(v,plow=50,smooth=baseline_smoothness,ns=mad_std(v)) for v in signals])
            # smooth_levels = (np.array([0.25, 0.5, 1, 1.5])*baseline_smoothness).astype(int)
            # bs = np.array([multi_scale_simple_baseline(v,plow=50,smooth_levels=smooth_levels,ns=utils.mad_std(v)) for v in signals])
            # bs = np.array([symmetrized_l1_runmin(v) for v in signals])
            bs = np.array([bl.l1_baseline2(v, l1smooth=baseline_smoothness) for v in signals])
            if np.any(np.isnan(bs)):
                print('Nan in ', sq)
                # return (signals, filters, center,sq,w_sh)
            rec_b = (bs.T @ filters).reshape(w_sh)
            out_f0[tuple(sq)] += (rec_b + center.reshape(w_sh[1:])) * crossfade_coefs

    out_data /= (1e-12 + counts)
    out_data *= (counts > 1e-12)
    if with_f0:
        out_f0 /= (1e-12 + counts)
        out_f0 *= (counts > 1e-12)
        return out_data, out_f0
    return out_data


def second_stage_svd(collection, fsh, n_clusters=_nclusters_, Nhood=100, clustering_algorithm='AgglomerativeWard'):
    out_signals = [np.zeros(c[0].shape, _dtype_) for c in collection]
    out_counts = np.zeros(len(collection), np.int)  # make crossafade here
    squares = make_grid(fsh[1:], Nhood, Nhood // 2)
    tstarts = set(c[3][0].start for c in collection)
    tsquares = [(t, sq) for t in tstarts for sq in squares]
    clustering_dispatcher = {
        'AgglomerativeWard'.lower(): lambda nclust: skcluster.AgglomerativeClustering(nclust),
        'KMeans'.lower(): lambda nclust: skcluster.KMeans(nclust),
        'MiniBatchKMeans'.lower(): lambda nclust: skcluster.MiniBatchKMeans(nclust)
    }

    def _is_local_patch(p, sqx):
        t0, sq = sqx
        tstart = p[0].start
        psq = p[1:]
        return (tstart == t0) & (slice_overlaps_square(psq, sq))

    for sqx in tqdm(tsquares, desc='Going through larger squares'):
        # print(sqx, collection[0][3], slice_starts_in_square(collection[0][3], sqx))
        signals = [c[0] for c in collection if _is_local_patch(c[3], sqx)]
        if not (len(signals)):
            print(sqx)
            for c in collection:
                print(sqx, c[3], _is_local_patch(c[3], sqx))
        nclust = min(n_clusters, len(signals))
        signals = np.vstack(signals)
        # clust=skcluster.KMeans(min(n_clusters,len(signals)))
        # clust = skcluster.AgglomerativeClustering(min(n_clusters,len(signals)))
        # number of signals can be different in some patches due to boundary conditions
        clust = clustering_dispatcher[clustering_algorithm.lower()](nclust)
        if clustering_algorithm == "MiniBatchKMeans".lower():
            clust.batch_size = min(clust.batch_size, len(signals))
        labels = clust.fit_predict(signals)
        sqx_approx = np.zeros(signals.shape, _dtype_)
        for i in np.unique(labels):
            group = labels == i
            u, s, vh = svd(signals[group], False)
            r = min_ncomp(s, (u.shape[0], vh.shape[1])) + 1
            # w = weight_components(all_svd_signals[group],vh,r)
            approx = u[:, :r] @ np.diag(s[:r]) @ vh[:r]
            sqx_approx[group] = approx
        kstart = 0
        for i, c in enumerate(collection):
            if _is_local_patch(c[3], sqx):
                l = len(c[0])
                out_signals[i] += sqx_approx[kstart:kstart + l]
                out_counts[i] += 1
                kstart += l
    return [(x / (1e-7 + cnt),) + c[1:] for c, x, cnt in zip(collection, out_signals, out_counts)]


def patch_svd_denoise_frames(frames, do_second_stage=False, save_coll=None,
                             tsvd_kw=None, second_stage_kw=None, inverse_kw=None):
    """
    Split frame stack into overlapping windows (patches), do truncated SVD projection of each patch, optionally improve
    temporal components by clustering and another SVD in larger windows, and finally merge inverse transforms of each patch.

    Input:
     - frames: TXY stack of frames to denoise
     - with_f0: whether to estimate smooth baseline fluorescence
     - do_second_stage: whether to do the clustering/secondary SVD on temporal compoments
     - n_clusters: how many clusters to use  at the second stage SVD
     - **kwargs: arguments to pass to `patch_tsvds_from_frames`

    Output:
     - denoised fluorescence or fluorescence and baseline
    """
    coll = patch_tsvds_from_frames(frames, **tsvd_kw)
    if do_second_stage:
        coll = second_stage_svd(coll, frames.shape, **second_stage_kw)
    if save_coll is not None:
        with gzip.open(save_coll, 'wb') as fh:
            pickle.dump((coll, frames.shape), fh)
    return project_from_tsvd_patches(coll, frames.shape, **inverse_kw)
