from collections import namedtuple

import numpy as np
from numpy import linalg

from scipy import ndimage as ndi
from scipy.stats import skew
import scipy as sp

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import cluster as skclust

from tqdm.auto import tqdm

from .. import scramble
from ..patches import make_grid
from ..utils import adaptive_filter_1d, adaptive_filter_2d
from ..cluster import clustering_dispatcher_
from ..masks import cleanup_cluster_map
from ..globals import _dtype_

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


def simple_tSVD(signals, min_ncomps=1, max_ncomps=100, return_components=True):
    sh = signals.shape
    u, s, vh = np.linalg.svd(signals, False)
    r = min_ncomp(s, (u.shape[0], vh.shape[1])) + 1
    r = min(max_ncomps, max(r, min_ncomps))
    u, vh = svd_flip_signs(u[:,:r], vh[:r])
    return u,s[:r],vh

def weight_components(data, components, rank=None, Npermutations=100, clip_percentile=95):
    """
    For a collection of signals (each row of input matrix is a signal),
    try to decide if using projection to the principal or svd components should describe
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


def tanh_step(x, window, overlap, taper_k=None):
    overlap = max(1, overlap)
    taper_width = overlap / 2
    if taper_k is None:
        taper_k = overlap / 10
    A = np.tanh((x+0.5-taper_width) / taper_k)
    B = np.tanh((window-(x+0.5)-taper_width) / taper_k)
    return np.clip((1.01 + A*B)/2, 0, 1)
