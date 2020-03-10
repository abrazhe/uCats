import numpy as np
from numpy import linalg
from numpy import arange
from numpy.linalg import norm, lstsq, svd, eig

from scipy import ndimage as ndi


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


from scipy.stats import skew


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
