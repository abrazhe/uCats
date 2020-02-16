from functools import partial

import numpy as np
import pandas as pd
from imfun.core import ah
from imfun.filt.dctsplines import l2spline
from matplotlib import pyplot as plt
from numba import jit
from pathos.pools import ProcessPool as Pool
from scipy import ndimage as ndi
from skimage.restoration import denoise_tv_chambolle

from .masks import threshold_object_size, select_overlapping, percentile_th_frames, opening_of_closing


def iterated_tv_chambolle(y, weight, niters=5):
    ys = np.copy(y)
    for i in range(niters):
        ys = denoise_tv_chambolle(ys, weight)
    return ys


def downsample_image(img):
    sigma_0 = 0.6
    sigma = sigma_0 * (1 / 0.25 - 1) ** 0.5
    im = ndi.gaussian_filter(img, sigma, mode='nearest')
    return ndi.zoom(im, 0.5)


def upsample_image(img):
    return ndi.zoom(img, 2, mode='nearest')


def clip_outliers(m, plow=0.5, phigh=99.5):
    px = np.percentile(m, (plow, phigh))
    return np.clip(m, *px)


def median_std_labeler(y, th=1):
    sigma = std_median(y)
    ys = smoothed_medianf(y, 0.5, 3)
    structures, nlab = ndi.label(y > np.median(y))
    peaks = ys >= th * sigma
    return y * select_overlapping(structures, peaks)


def zproject_top_average_frames(frames, percentile=85):
    sh = frames.shape
    pmap = np.percentile(frames, percentile, axis=0)
    out = np.zeros(sh[1:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            p = pmap[r, c]
            v = frames[:, r, c]
            out[r, c] = np.mean(v[v >= p])
    return out


@jit
def avg_filter_greater(m, th=0):
    nr, nc = m.shape
    out = np.zeros_like(m)
    for r in range(nr):
        for c in range(nc):
            if m[r, c] <= th:
                continue
            count, acc = 0, 0
            for i in range(r - 1, r + 2):
                for j in range(c - 1, c + 2):
                    if (0 <= i < nr) and (0 <= j < nc):
                        if m[i, j] > th:
                            count += 1
                            acc += m[i, j]
            if count > 0:
                out[r, c] = acc / count
    return out


def find_bias(y, th=3, ns=None):
    if ns is None:
        ns = rolling_sd_pd(y)
    return np.median(y[np.abs(y - np.median(y)) <= th * ns])


@jit
def find_bias_frames(frames, th, ns):
    signals = ah.ravel_frames(frames).T
    nsr = np.ravel(ns)
    # print(nsr.shape, signals.shape)
    biases = np.zeros(nsr.shape)
    for j in range(len(biases)):
        biases[j] = find_bias(signals[j], th, nsr[j])
    # biases = np.array([find_bias(v,th,ns_) for  v,ns_ in zip(signals, nsr)])
    return biases.reshape(frames[0].shape)


def to_zscore_frames(frames):
    nsm = mad_std(frames, axis=0)
    biases = find_bias_frames(frames, 3, nsm)

    return np.where(nsm > 1e-5, (frames - biases) / (nsm + 1e-5), 0)


def activity_mask_median_filtering(frames, nw=11, th=1.0, plow=2.5, smooth=2.5,
                                   verbose=True):
    mf_frames50 = ndi.percentile_filter(frames, 50, (1, nw, nw))  # spatial median filter
    # mf_frames85 = ndi.percentile_filter(frames,85, (1,nw,nw))    # spatial top 85% filter
    mf_frames = mf_frames50  # *mf_frames85
    del mf_frames50  # ,mf_frames85

    if verbose:
        print('Done percentile filters')

    mf_frames = to_zscore_frames(mf_frames)
    mf_frames = np.clip(mf_frames, *np.percentile(mf_frames, (0.5, 99.5)))
    # return mf_frames

    th = percentile_th_frames(mf_frames, plow)
    mask = (mf_frames > th) * (ndi.gaussian_filter(mf_frames, (smooth, 0.5, 0.5)) > th)
    mask = ndi.binary_dilation(opening_of_closing(mask))
    # mask = np.array([threshold_object_size(m,)])
    # mask = threshold_object_size(mask, 4**3)
    if verbose:
        print('Done mask from spatial filters')
    return mask


def rolling_sd_pd(v, hw=None, with_plots=False, correct_factor=1., smooth_output=True, input_is_details=False):
    """
    Etimate time-varying level of noise standard deviation
    """
    if not input_is_details:
        details = v - ndi.median_filter(v, 20)
    else:
        details = v
    if hw is None: hw = int(len(details) / 10.)
    padded = np.pad(details, 2 * hw, mode='reflect')
    tv = np.arange(len(details))

    s = pd.Series(padded)
    rkw = dict(window=2 * hw, center=True)

    out = (s - s.rolling(**rkw).median()).abs().rolling(**rkw).median()
    out = 1.4826 * np.array(out)[2 * hw:-2 * hw]

    if with_plots:
        f, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(tv, details, 'gray')
        ax.plot(tv, out, 'y')
        ax.plot(tv, 2 * out, 'orange')
        ax.set_xlim(0, len(v))
        ax.set_title('Estimating running s.d.')
        ax.set_xlabel('samples')
    out = out / correct_factor
    if smooth_output:
        out = l2spline(out, s=2 * hw)
    return out


def rolling_sd(v, hw=None, with_plots=False, correct_factor=1., smooth_output=True, input_is_details=False):
    if not input_is_details:
        details = v - ndi.median_filter(v, 20)
    else:
        details = v
    if hw is None: hw = int(len(details) / 10.)
    padded = np.pad(details, hw, mode='reflect')
    tv = np.arange(len(details))
    out = np.zeros(len(details))
    for i in np.arange(len(details)):
        out[i] = mad_std(padded[i:i + 2 * hw])
    if with_plots:
        f, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(tv, details, 'gray')
        ax.plot(tv, out, 'y')
        ax.plot(tv, 2 * out, 'orange')
        ax.set_xlim(0, len(v))
        ax.set_title('Estimating running s.d.')
        ax.set_xlabel('samples')
    out = out / correct_factor
    if smooth_output:
        out = l2spline(out, s=2 * hw)
    return out


def rolling_sd_scipy(v, hw=None, with_plots=False, correct_factor=1., smooth_output=True, input_is_details=False):
    if not input_is_details:
        details = v - ndi.median_filter(v, 20)
    else:
        details = v
    if hw is None: hw = int(len(details) / 10.)
    padded = np.pad(details, hw, mode='reflect')
    tv = np.arange(len(details))
    # out = np.zeros(len(details))

    # rolling_median = lambda x: ndi.median_filter(x, 2*hw)
    rolling_median = partial(ndi.median_filter, size=2 * hw)

    out = 1.4826 * rolling_median(np.abs(padded - rolling_median(padded)))[hw:-hw]

    if with_plots:
        f, ax = plt.subplots(1, 1, sharex=True)
        ax.plot(tv, details, 'gray')
        ax.plot(tv, out, 'y')
        ax.plot(tv, 2 * out, 'orange')
        ax.set_xlim(0, len(v))
        ax.set_title('Estimating running s.d.')
        ax.set_xlabel('samples')
    out = out / correct_factor
    if smooth_output:
        out = l2spline(out, s=2 * hw)
    return out


def rolling_sd_scipy_nd(arr, hw=None, correct_factor=1., smooth_output=True):
    if hw is None: hw = int(np.ceil(np.max(arr.shape) / 10))
    padded = np.pad(arr, hw, mode='reflect')
    rolling_median = lambda x: ndi.median_filter(x, 2 * hw)
    crop = (slice(hw, -hw),) * np.ndim(arr)
    out = 1.4826 * rolling_median(np.abs(padded - rolling_median(padded)))[crop]

    out = out / correct_factor
    if smooth_output:
        out = l2spline(out, s=hw)
    return out


def smoothed_medianf(v, smooth=10, wmedian=10):
    "Robust smoothing by first applying median filter and then applying L2-spline filter"
    return l2spline(ndi.median_filter(v, wmedian), smooth)


def std_median(v, axis=None):
    if axis is None:
        N = float(len(v))
    else:
        N = float(v.shape[axis])
    md = np.median(v, axis=axis)
    return (np.sum((v - md) ** 2, axis) / N) ** 0.5


def mad_std(v, axis=None):
    mad = np.median(abs(v - np.median(v, axis=axis)), axis=axis)
    return mad * 1.4826


def iterative_noise_sd(data, cut=5, axis=None, niter=10):
    data = np.copy(data)
    for i in range(niter):
        sd = np.std(data, axis=axis)
        mu = np.mean(data, axis=axis)
        outliers = np.abs(data - mu) > cut * sd
        data = np.where(outliers, data * 0.5, data)
        # data[outliers] = cut*sd
    return sd


def adaptive_median_filter(frames, th=5, tsmooth=1, ssmooth=5, keep_clusters=False, reverse=False, min_cluster_size=7):
    smoothed_frames = ndi.median_filter(frames, (tsmooth, ssmooth, ssmooth))
    details = frames - smoothed_frames
    sdmap = mad_std(frames, axis=0)
    outliers = np.abs(details) > th * sdmap
    # s = np.array([[[0,0,0],[0,1,0],[0,0,0]]]*3)
    if keep_clusters:
        clusters = threshold_object_size(outliers, min_cluster_size)
        outliers = ~clusters if reverse else outliers ^ clusters
    else:
        if reverse:
            outliers = ~outliers
    return np.where(outliers, smoothed_frames, frames)


def adaptive_filter_1d(v, th=5, smooth=5, smoother=ndi.median_filter, keep_clusters=False, reverse=False,
                       min_cluster_size=5):
    vsmooth = smoother(v, smooth)
    details = v - vsmooth
    sd = mad_std(v)
    outliers = np.abs(details) > th * sd
    if keep_clusters:
        clusters = threshold_object_size(outliers, min_cluster_size)
        outliers = ~clusters if reverse else outliers ^ clusters
    else:
        if reverse:
            outliers = ~outliers
    return np.where(outliers, vsmooth, v)


def adaptive_filter_2d(img, th=5, smooth=5, smoother=ndi.median_filter, keep_clusters=False, reverse=False,
                       min_cluster_size=5):
    imgf = smoother(img, smooth)
    details = img - imgf
    sd = mad_std(img)
    outliers = np.abs(details) > th * sd  # in real adaptive filter the s.d. must be rolling!
    if keep_clusters:
        clusters = threshold_object_size(outliers, min_cluster_size)
        outliers = ~clusters if reverse else outliers ^ clusters
    else:
        if reverse:
            outliers = ~outliers
    return np.where(outliers, imgf, img)


def process_signals_parallel(collection, pipeline, pipeline_kw=None, njobs=4):
    """
    Process temporal signals some pipeline function and return processed signals
    (parallel version)
    """
    out = []
    pool = Pool(njobs)
    # def _pipeline_(*args):
    #    if pipeline_kw is not None:
    #        return pipeline(*args, **pipeline_kw)
    #    else:
    #        return pipeline(*args)
    _pipeline_ = pipeline if pipeline_kw is None else partial(pipeline, **pipeline_kw)
    recs = pool.map(_pipeline_, [c[0] for c in collection], chunksize=4)  # setting chunksize here is experimental
    # pool.close()
    # pool.join()
    return [(r, s, w) for r, (v, s, w) in zip(recs, collection)]


# def adaptive_median_filter_frames(frames,th=5, tsmooth=5,ssmooth=1):
#     medfilt = ndi.median_filter(frames, [tsmooth,ssmooth,ssmooth])
#     details = frames - medfilt
#     mdmap = np.median(details, axis=0)
#     sdmap = np.median(abs(details - mdmap), axis=0)*1.4826
#     return np.where(abs(details-mdmap)  >  th*sdmap, medfilt, frames)

def max_shifts(shifts, verbose=0):
    # ms = np.max(np.abs([s.fn_((0,0)) if s.fn_ else s.field[...,0,0] for s in shifts]),axis=0)
    ms = np.max([np.array([np.percentile(f, 99) for f in np.abs(w.field)]) for w in shifts], 0)
    if verbose: print('Maximal shifts were (x,y): ', ms)
    return np.ceil(ms).astype(int)


def crop_by_max_shift(data, shifts, mx_shifts=None):
    if mx_shifts is None:
        mx_shifts = max_shifts(shifts)
    lims = 2 * mx_shifts
    sh = data.shape[1:]
    return data[:, lims[1]:sh[0] - lims[1], lims[0]:sh[1] - lims[0]]
