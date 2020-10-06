from collections import namedtuple

import numpy as np
from numpy import linalg

from scipy import ndimage as ndi
from scipy.stats import skew

from tqdm.auto import tqdm

from . import scramble

from .anscombe import Anscombe
from .patches import make_grid
from .utils import adaptive_filter_1d, adaptive_filter_2d


from .globals import _dtype_

_do_pruning_ = False


def lambda_star(beta):
    return np.sqrt(2 * (beta+1) + (8*beta) / (beta + 1 + np.sqrt(beta**2 + 14*beta + 1)))


def omega_approx(beta):
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82*beta + 1.43


def svht(sv, sh, sigma=None):
    "Gavish and Donoho 2014"
    m, n = sh
    if m > n:
        m, n = n, m
    beta = m / n
    omg = omega_approx(beta)
    if sigma is None:
        return omg * np.median(sv)
    else:
        return lambda_star(beta) * np.sqrt(n) * sigma


def min_ncomp(sv, sh, sigma=None):
    th = svht(sv, sh, sigma)
    return np.sum(sv >= th)


def pca_flip_signs(pcf, medianw=None):
    L = len(pcf.coords)
    if medianw is None:
        medianw = L // 5
    for i, c in enumerate(pcf.coords.T):
        sk = skew(c - ndi.median_filter(c, medianw))
        sg = np.sign(sk)
        #print(i, sk)
        pcf.coords[:, i] *= sg
        pcf.tsvd.components_[i] *= sg
    return pcf


def svd_flip_signs(u, vh, mode='v'):
    "flip signs of U,V pairs of the SVD so that either V or U have positive skewness"
    for i in range(len(vh)):
        if mode == 'v':
            sg = np.sign(skew(vh[i]))
        else:
            sg = np.sign(skew(u[:, i]))
        u[:, i] *= sg
        vh[i] *= sg
    return u, vh


SVD_patch = namedtuple('SVD_patch', "signals filters pnorm center sq w_shape toverlap soverlap")


class Windowed_tSVD:
    def __init__(self,
                 patch_ssize:'spatial size of the patch'=8,
                 patch_tsize:'temporal size of the patch'=600,
                 soverlap:'spatial overlap between patches'=4,
                 toverlap:'temporal overlap between patches'=100,
                 min_ncomps:'minimal number of SVD components to use'=1,
                 max_ncomps:'maximal number of SVD components'=100,
                 do_pruning:'pruning of spatial coefficients'=_do_pruning_,
                 center_data:'subtract mean before SVD'=True,
                 tfilter:'window of adaptive median filter for temporal components'=3,
                 sfilter:'window of adaptive median filter for spatial components'=3,
                 verbose=False):

        self.patch_ssize = patch_ssize
        self.soverlap = soverlap

        self.patch_tsize = patch_tsize
        self.toverlap = toverlap

        self.min_ncomps = min_ncomps
        self.max_ncomps = max_ncomps

        self.center_data = center_data

        self.do_pruning = do_pruning
        self.t_amf = tfilter
        self.s_amf = sfilter

        self.patches_ = None
        self.verbose = verbose


        self.fit_transform_ansc = Anscombe.wrap_input(self.fit_transform)
        self.inverse_transform_ansc = Anscombe.wrap_output(self.inverse_transform)

    def fit_transform(self, frames, cuts_in=None):
        data = np.array(frames).astype(_dtype_)
        acc = []
        #squares =  list(map(tuple, make_grid(d.shape[1:], patch_size,stride)))
        L = len(frames)
        ## Sometimes chunks of the recording are missing and we want
        ## to cut them out and sometimes we want to explicitly cut
        ## the time_patches at certain points
        ## This has to be done here
        if cuts_in is None:
            cuts_in = (0,L)
        self.patch_tsize = min(L, self.patch_tsize)
        # if self.toverlap >= self.patch_tsize:
        #     self.toverlap = self.patch_tsize // 2
        if self.toverlap >= self.patch_tsize:
            self.toverlap = self.patch_tsize // 4

        squares = make_grid(np.shape(frames),
                            (self.patch_tsize, self.patch_ssize, self.patch_ssize),
                            (self.toverlap, self.soverlap, self.soverlap))
        if self.t_amf > 0:
            #print('Will smooth temporal components')
            #smoother = lambda v: smoothed_medianf(v, tsmooth*0.5, tsmooth)
            tsmoother = lambda v: adaptive_filter_1d(
                v, th=3, smooth=self.t_amf, keep_clusters=False)
        if self.s_amf > 0:
            ssmoother = lambda v: adaptive_filter_2d(v.reshape(self.patch_ssize, -1),
                                                     smooth=self.t_amf,
                                                     keep_clusters=False).reshape(v.shape)

        #print('Splitting to patches and doing SVD decompositions',flush=True)


        for sq in tqdm(squares, desc='truncSVD in patches', disable=not self.verbose):

            patch_frames = data[sq]
            L = len(patch_frames)
            w_sh = np.shape(patch_frames)

            # now each column is signal in one pixel
            patch = patch_frames.reshape(L,-1)
            #pnorm = np.linalg.norm(patch)
            patch_c = np.zeros(patch.shape[1])
            if self.center_data:
                patch_c = np.mean(patch, 0)
                patch = patch - patch_c

            u, s, vh = np.linalg.svd(patch.T, full_matrices=False)
            rank = max(self.min_ncomps, min_ncomp(s, patch.shape) + 1)
            rank = min(rank, self.max_ncomps)
            u, vh = svd_flip_signs(u[:, :rank], vh[:rank])

            if self.do_pruning:
                w = weight_components(patch.T, vh, rank)
            else:
                w = np.ones(u[:, :rank].shape)

            svd_signals, loadings = vh[:rank], u[:, :rank] * w
            s = s[:rank]
            pnorm = (s**2).sum()**0.5
            svd_signals = svd_signals * s[:, None]**0.5
            loadings = loadings * s[None,:]**0.5

            if self.t_amf > 0:
                svd_signals = np.array([tsmoother(v) for v in svd_signals])
            W = loadings.T
            if (self.s_amf > 0) and (patch.shape[1] == self.patch_ssize**2):
                W = np.array([ssmoother(v) for v in W])
            p = SVD_patch(svd_signals, W, pnorm, patch_c, sq, w_sh, self.toverlap, self.soverlap)
            acc.append(p)
        self.patches_ = acc
        self.data_shape_ = np.shape(frames)
        return self.patches_


    def inverse_transform(self, patches=None):
        if patches is None:
            patches = self.patches_

        out_data = np.zeros(self.data_shape_, dtype=_dtype_)
        counts = np.zeros(self.data_shape_, _dtype_)    # candidate for crossfade

        for p in tqdm(patches,
                      desc='truncSVD inverse transform',
                      disable=not self.verbose):

            L = p.w_shape[0]
            t_crossfade = tanh_step(np.arange(L), L, p.toverlap).astype(_dtype_)
            t_crossfade = t_crossfade[:, None, None]

            psize = np.max(p.w_shape[1:])
            scf = tanh_step(np.arange(psize), psize, p.soverlap, p.soverlap/2)
            scf = scf[:,None]
            w_crossfade = scf @ scf.T
            nr,nc = p.w_shape[1:]
            w_crossfade = w_crossfade[:nr, :nc].astype(_dtype_)
            w_crossfade = w_crossfade[None, :, :]

            counts[p.sq] += t_crossfade * w_crossfade


            rec = (p.signals.T @ p.filters).reshape(p.w_shape)

            rnorm = np.linalg.norm(rec)
            #rec = rec*p.pnorm/rnorm

            rec += p.center.reshape(p.w_shape[1:])
            out_data[tuple(p.sq)] += rec * t_crossfade * w_crossfade

        out_data /= (1e-12 + counts)
        out_data *= (counts > 1e-12)

        return out_data



# # TODO: would it be better to make it a generator?
# def patch_tsvd_transform(frames,
#                          patch_ssize=10,
#                          patch_tsize=600,
#                          soverlap=5,
#                          toverlap=100,
#                          min_ncomps=1,
#                          max_ncomps=100,
#                          do_pruning=_do_pruning_,
#                          tfilter=0,
#                          sfilter=0):
#     """
#     Slide a rectangle spatial window and extract local dynamics in this window (patch),
#     then do truncated SVD decomposition of the local dynamics for each patch.
#
#     Input:
#      - frames: a TXY 3D stack of frames (array-like)
#      - overlap: overlap between windows (default: half window)
#      - patch_size: spatial size of the window (default: 10 px)
#      - min_ncomps: minimal number of components to retain
#      - do_pruning: whether to do coefficient pruning based on comparison to shuffled signals
#      - tfilter: scalar, if > 0, adaptive median filter window for temporal components
#      - sfilter: scalar, if > 0, adaptive median filter window for spatial components
#
#     Output:
#      - list of tuples of the form:
#        (temporal components, spatial components, patch average, location of the patch in data, patch shape)
#     """
#     data = np.array(frames).astype(_dtype_)
#     acc = []
#     #squares =  list(map(tuple, make_grid(d.shape[1:], patch_size,stride)))
#     L = len(frames)
#     patch_tsize = min(L, patch_tsize)
#     if toverlap >= patch_tsize:
#         toverlap = patch_tsize // 2
#     #tstride = min(L, tstride)
#     if toverlap >= patch_tsize:
#         toverlap=patch_tsize//4
#     squares = make_grid(frames.shape, (patch_tsize, patch_ssize, patch_ssize),
#                         (toverlap, soverlap, soverlap))
#     if tfilter > 0:
#         #print('Will smooth temporal components')
#         #smoother = lambda v: smoothed_medianf(v, tsmooth*0.5, tsmooth)
#         tsmoother = lambda v: adaptive_filter_1d(v, th=3,
#                                                  smooth=tfilter,
#                                                  keep_clusters=False)
#     if sfilter > 0:
#         ssmoother = lambda v: adaptive_filter_2d(v.reshape(patch_ssize, -1),
#                                                  smooth=sfilter,
#                                                  keep_clusters=False).reshape(v.shape)
#
#     #print('Splitting to patches and doing SVD decompositions',flush=True)
#     for sq in tqdm(squares, desc='truncSVD in patches'):
#
#         patch_frames = data[sq]
#         L = len(patch_frames)
#         w_sh = patch_frames.shape
#
#         patch = patch_frames.reshape(L, -1)    # now each column is signal in one pixel
#         patch_c = np.mean(patch, 0)
#         patch = patch - patch_c
#
#         u, s, vh = np.linalg.svd(patch.T, full_matrices=False)
#         rank = max(min_ncomps, min_ncomp(s, patch.shape) + 1)
#         rank = min(rank, max_ncomps)
#         u, vh = svd_flip_signs(u[:, :rank], vh[:rank])
#
#         if do_pruning:
#             w = weight_components(patch.T, vh, rank)
#         else:
#             w = np.ones(u[:, :rank].shape)
#
#         svd_signals, loadings = vh[:rank], u[:, :rank] * w
#         s = s[:rank]
#         svd_signals = svd_signals * s[:, None]
#
#         if tfilter > 0:
#             svd_signals = np.array([tsmoother(v) for v in svd_signals])
#         W = loadings.T
#         if (sfilter > 0) and (patch.shape[1] == patch_ssize**2):
#             W = np.array([ssmoother(v) for v in W])
#         p = SVD_patch(svd_signals, W, patch_c, sq, w_sh, toverlap, soverlap)
#         acc.append(p)
#     return acc


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
        w[:, j] = coefs_orig[j] >= np.percentile(
            coefs_randomized[:, j, :], clip_percentile, axis=0)
    return w


# def tsvd_rec_with_weighting(data, rank=None):
#     """
#     Do truncated SVD approximation using coefficient pruning by comparisons to shuffled data
#     Input: data matrix (Nsamples, Nfeatures), each row is interpreted as a signal or feature vector
#     Output: approximated data using rank-truncated SVD
#     """
#     dc = data.mean(1)[:, None]
#     u, s, vh = np.linalg.svd(data - dc, False)
#     if rank is None:
#         rank = min_ncomp(s, data.shape) + 1
#     w = weight_components(data - dc, vh, rank)
#     return (u[:, :rank] * w) @ np.diag(s[:rank]) @ vh[:rank] + dc


def tanh_step(x, window, overlap, taper_k=None):
    overlap = max(1, overlap)
    taper_width = overlap / 2
    if taper_k is None:
        taper_k = overlap / 10
    A = np.tanh((x+0.5-taper_width) / taper_k)
    B = np.tanh((window-(x+0.5)-taper_width) / taper_k)
    return np.clip((1.01 + A*B)/2, 0, 1)


# ## TODO: crossfade for overlapping patches, at least in time
# def patch_tsvd_inverse_transform(collection, shape):
#     """
#     Do inverse transform from local truncated SVD projections (output of `patch_tsvds_from_frames`)
#     Goes through patches, calculates approximations and combines overlapping singal estimates
#
#     Input:
#      - collection of tSVD components along with patch location and shape (see `patch_tsvds_from_frames`)
#      - shape of the full frame stack to reconstruct
#
#     Output: approximation of the fluorescence signal
#     """
#     out_data = np.zeros(shape, dtype=_dtype_)
#     counts = np.zeros(shape, _dtype_)    # candidate for crossfade
#
#     for p in tqdm(collection, desc='truncSVD inverse transform'):
#         L = p.w_shape[0]
#         tcrossfade_coefs = tanh_step(np.arange(L), L, p.toverlap).astype(_dtype_)[:, None, None]
#         counts[p.sq] += tcrossfade_coefs
#
#         rec = (p.signals.T @ p.filters).reshape(p.w_shape)
#         out_data[tuple(p.sq)] += (rec + p.center.reshape(p.w_shape[1:])) * tcrossfade_coefs
#
#     out_data /= (1e-12 + counts)
#     out_data *= (counts > 1e-12)
#
#     return out_data
