"""
baselines -- routines to estimate baseline fluorescence in 1D or TXY data
"""

import sys

import pickle

from functools import partial

import numpy as np
from numpy import linalg
from numpy.linalg import norm, lstsq, svd, eig

from scipy import ndimage as ndi
from scipy import signal

from sklearn import decomposition as skd

from numba import jit

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from imfun import fseq
from imfun.filt.dctsplines import l2spline, l1spline
from imfun import components
from imfun.multiscale import mvm
from imfun.components.pca import PCA_frames
from imfun import core
from imfun.core import extrema


min_px_size_ = 10

from .patches import make_grid
from .utils import rolling_sd_pd, find_bias, find_bias_frames
from .utils import mad_std
from .utils import process_signals_parallel
from .utils import iterated_tv_chambolle
from .decomposition import pca_flip_signs
from . import patches
from .globals import _dtype_


def store_baseline_pickle(name, frames, ncomp=50):
    pcf = components.pca.PCA_frames(frames, npc=50)
    pickle.dump(pcf, open(name, 'wb'))


def load_baseline_pickle(name):
    pcf = pickle.load(open(name, 'rb'))
    return pcf.inverse_transform(pcf.coords)


@jit
def running_min(v):
    out = np.zeros(v.shape)
    mn = v[0]
    for i in range(len(v)):
        mn = min(mn, v[i])
        out[i] = mn
    return out


def top_running_min(v):
    return np.maximum(running_min(v), running_min(v[::-1])[::-1])


def windowed_runmin(y, wsize=50, woverlap=25):
    L = len(y)
    if woverlap >= wsize:
        woverlap = wsize//2
    sqs = make_grid((L, ), wsize, woverlap)
    out = (np.max(y) + 0.1) * np.ones(L, _dtype_)
    for sq in sqs:
        sl = sq[0]
        bx = top_running_min(y[sl])
        out[sl] = np.minimum(out[sl], bx)
    return out


def select_most_stable_branch(variants, window=50, woverlap=None):
    L = len(variants[0])
    if wsize is None:
        wsize = max(2, window // 3)
    if wovelap >= window:
        woverlap = window//2
    squares = make_grid((L, 1), window, wsize)
    out = np.zeros(L)
    gradients = [np.gradient(np.gradient(v)) for v in variants]
    for sq in squares:
        sq = sq[0]
        best = np.argmin([np.sum(abs(g[sq])) for g in gradients])
        out[sq] = variants[best][sq]
    return out


def symmetrized_l1_runmin(y, tv_weight=1, tv_niter=5, l1smooth=50, l2smooth=5):
    b1 = l1spline(y, l1smooth)
    ns = mad_std(y - b1)
    b0 = iterated_tv_chambolle(y, tv_weight * ns, tv_niter)
    b2a = windowed_runmin(b0 - b1) + b1
    b2b = -windowed_runmin(-(b0 - b1)) + b1
    b2a, b2b = (iterated_tv_chambolle(b, 1, 5)
                for b in (b2a, b2b))    # making this too slow?
    return l2spline(select_most_stable_branch((b2a, b2b)), l2smooth)


def l1_baseline2(y, smooth=25, median_window=50):
    npad = median_window // 2
    ypad = np.pad(y, npad, 'median', stat_length=smooth // 2)
    b1 = l1spline(ypad, smooth)[npad:-npad]    # get overall trend
    v = y - b1
    vm = ndi.percentile_filter(v, 50, median_window)
    vmlow = windowed_runmin(vm, 2 * median_window, 2 * median_window // 5)
    b_add = l2spline(vmlow, smooth / 2)
    return b1 + b_add

def iterated_smoothing_baseline(y, niter=10, th=3, smooth_fn=l1spline,  fnkw=None):
    """Baseline from iterated thresholded smoothing"""
    if fnkw is None:
        fnkw = {}
    ytemp = y
    ns = mad_std(np.diff(y))
    for i_ in range(niter):
        ys = smooth_fn(ytemp, **fnkw)
        ytemp = np.where(np.abs(y-ys)<ns*th, y, ys)
    return ys

def iterated_l1_baseline(y, niter=10, th=3, smooth=10):
    """Baseline from iterated thresholded filtering with L1 spline smoother"""
    return iterated_smoothing_baseline(y,
                                       niter=niter, th=th,
                                       smooth_fn=l1spline,
                                       fnkw=dict(s=smooth))

def iterated_savgol_baseline(y, niter=10, window=299, order=3, th=3, **kwargs):
    """Baseline from iterated thresholded filtering with Savitzky-Golyaev smoother"""
    if not window%2: window = window+1 # ensure odd window length
    return iterated_smoothing_baseline(y,
                                       niter=niter, th=th,
                                       smooth_fn=signal.savgol_filter,
                                       fnkw=dict(window_length=window, polyorder=order,**kwargs))

def percentile_baseline(y,
                        plow=25,
                        percentile_window=25,
                        out_smooth=25,
                        smoother=l2spline,
                        ns=None,
                        th=3,
                        npad=None):
    L = len(y)

    npad = percentile_window // 2 if (npad is None) else npad
    ypad = np.pad(y, npad, 'median', stat_length=min(L, 10)) if npad > 0 else y

    b = smoother(ndi.percentile_filter(ypad, plow, percentile_window), out_smooth)
    b = b[npad:L + npad]
    if ns is None:
        ns = rolling_sd_pd(y)
    d = y - b
    if not np.any(ns):
        ns = np.std(y)
    bg_points = d[np.abs(d) <= th * ns]
    if len(bg_points) > 10:
        b = b + np.median(bg_points)    # correct scalar shift
    return b


def baseline_with_shifts(y, l1smooth=25):
    ys_l1 = l1spline(y, l1smooth)
    ns = mad_std(y - ys_l1)
    ys = iterated_tv_chambolle(y, 1 * ns, 5)
    jump_locs, shift = find_jumps(ys, ys_l1, pre_smooth=1.5)
    trend = l1_baseline2(y - shift, l1smooth)
    baseline = trend + shift
    return baseline

def find_jumps(ys_tv, ys_l1, pre_smooth=1.5, top_gradient=95):
    v = ys_tv - ys_l1
    if pre_smooth > 0:
        v = l2spline(v, pre_smooth)
    #g = np.abs(np.gradient(v))
    maxima = extrema.locextr(v, refine=False, output='max')
    xfit, yfit, der1, maxima, minima = extrema.locextr(v, refine=False, sort_values=False)
    _, _, _, vvmx, vvmn = extrema.locextr(np.cumsum(v), refine=False, sort_values=False)
    vv_extrema = np.concatenate([vvmx, vvmn])
    g = np.abs(der1)
    ee = np.array(extrema.locextr(g, refine=False, sort_values=False, output='max'))
    ee = ee[ee[:, 1] >= np.percentile(g, top_gradient)]
    all_extrema = np.concatenate([maxima, minima])
    extrema_types = {em: (1 if em in maxima else -1) for em in all_extrema}
    jumps = []
    Lee = len(ee)
    for k, em in enumerate(ee[:, 0]):
        if np.any(all_extrema <= em):
            leftmost = np.max(all_extrema[all_extrema <= em])
            if k > 0 and leftmost <= ee[k - 1, 0]:
                continue
        else:
            continue
        if np.any(all_extrema > em):
            rightmost = np.min(all_extrema[all_extrema > em])
            if k < Lee - 1 and rightmost > ee[k + 1, 0]:
                continue
        else:
            continue
        if extrema_types[leftmost] != extrema_types[rightmost]:
            if np.sign(v[leftmost]) != np.sign(v[rightmost]):
                if np.min(np.abs(em - vv_extrema)) < 2:
                    jumps.append(int(em))

    shift = np.zeros(len(v))
    L = len(v)
    for j in jumps:
        if j < L - 1:
            shift[j + 1] = np.mean(ys_tv[j + 1:min(j + 6, L)]) - np.mean(
                ys_tv[max(0, j - 5):j])
    return jumps, np.cumsum(shift)


def first_pc_baseline(frames, niters=10, baseline_fn=l1_baseline2, fnkw=None, verbose=False):
    f0 = np.zeros(frames.shape, _dtype_)
    fnkw = {} if fnkw is None else fnkw
    iter_range = range(niters)
    if verbose:
        iter_range = tqdm(iter_range)
    for i in iter_range:
        pcf = PCA_frames(frames - f0, 1)
        pcf = pca_flip_signs(pcf)
        y = pcf.coords[:, 0]
        b = baseline_fn(y, **fnkw)
        pc_baseline = b    #+ percentile_baseline(y-b, smooth=simple_smooth)
        f0 += pcf.inverse_transform(pc_baseline.reshape(-1, 1))
    return f0

def multi_scale_simple_baseline(y,
                                plow=50,
                                th=3,
                                smooth_levels=[10, 20, 40, 80, 160],
                                ns=None):
    if ns is None:
        ns = rolling_sd_pd(y)

    b_estimates = [percentile_baseline(y, plow, th, smooth, ns)
                   for smooth in smooth_levels]

    low_env = np.amin(b_estimates, axis=0)
    low_env = np.clip(low_env, np.min(y), np.max(y))
    return l2spline(low_env, np.min(smooth_levels))


def baseline_als_spl(y,
                     k=0.5,
                     tau=11,
                     smooth=25.,
                     p=0.001,
                     niter=100,
                     eps=1e-4,
                     rsd=None,
                     rsd_smoother=None,
                     smoother=l2spline,
                     asymm_ratio=0.9,
                     correct_skew=False):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005),
    via DCT-based spline smoothing
    """
    #npad=int(smooth)
    nsmooth = np.int(np.ceil(smooth))
    npad = nsmooth

    y = np.pad(y, npad, "reflect")
    L = len(y)
    w = np.ones(L)

    if rsd is None:
        if rsd_smoother is None:
            #rsd_smoother = lambda v_: l2spline(v_, 5)
            #rsd_smoother = lambda v_: ndi.median_filter(y,7)
            rsd_smoother = partial(ndi.median_filter, size=7)
        rsd = rolling_sd_pd(y - rsd_smoother(y), input_is_details=True)
    else:
        rsd = np.pad(rsd, npad, "reflect")

    #ys = l1spline(y,tau)
    ntau = np.int(np.ceil(tau))
    ys = ndi.median_filter(y, ntau)
    s2 = l1spline(y, smooth / 4.)
    #s2 = l2spline(y,smooth/4.)
    zprev = None
    for i in range(niter):
        z = smoother(ys, s=smooth, weights=w)
        clip_symm = abs(y - z) > k * rsd
        clip_asymm = y - z > k * rsd
        clip_asymm2 = y - z <= -k * rsd
        r = asymm_ratio    #*core.rescale(1./(1e-6+rsd))

        #w = p*clip_asymm + (1-p)*(1-r)*(~clip_symm) + (1-p)*r*(clip_asymm2)
        w = p * (1-r) * clip_asymm + (1-p) * (~clip_symm) + p * r * (clip_asymm2)
        w[:npad] = (1 - p)
        w[-npad:] = (1 - p)
        if zprev is not None:
            if norm(z - zprev) / norm(zprev) < eps:
                break
        zprev = z
    z = smoother(np.min((z, s2), 0), smooth)
    if correct_skew:
        # Correction for skewness introduced by asymmetry.
        z += r * rsd
    return z[npad:-npad]


def double_scale_baseline(y, smooth1=15., smooth2=25., rsd=None, **kwargs):
    """
    Baseline estimation in 1D signals by asymmetric smoothing and using two different time scales
    """
    if rsd is None:
        #rsd_smoother = lambda v_: ndi.median_filter(y,7)
        rsd_smoother = partial(ndi.median_filter, size=7)
        rsd = rolling_sd_pd(y - rsd_smoother(y), input_is_details=True)
    b1 = baseline_als_spl(y, tau=smooth1, smooth=smooth1, rsd=rsd, **kwargs)
    b2 = baseline_als_spl(y, tau=smooth1, smooth=smooth2, rsd=rsd, **kwargs)
    return l2spline(np.amin([b1, b2], 0), smooth1)


def viz_baseline(v,
                 dt=1.,
                 baseline_fn=baseline_als_spl,
                 smoother=partial(l2spline, s=5),
                 ax=None,
                 **kwargs):
    """
    Visualize results of baseline estimation
    """
    if ax is None:
        plt.figure(figsize=(14, 6))
        ax = plt.gca()
    tv = np.arange(len(v)) * dt
    ax.plot(tv, v, 'gray')
    b = baseline_fn(v, **kwargs)
    rsd = rolling_sd_pd(v - smoother(v))
    ax.fill_between(tv, b - rsd, b + rsd, color='y', alpha=0.75)
    ax.fill_between(tv, b - 2.0*rsd, b + 2.0*rsd, color='y', alpha=0.5)
    ax.plot(tv, smoother(v), 'k')
    ax.plot(tv, b, 'teal', lw=1)
    ax.axis('tight')





def frames_pca_baseline(frames,
                        npc=None,
                        pcf=None,
                        smooth_fn=iterated_savgol_baseline,
                        fnkw=None):
    """
    Use smoothed principal components to estimate time-varying baseline fluorescence F0
    """
    fnkw = dict() if fnkw is None else fnkw
    if pcf is None:
        npc = len(frames) // 20 if npc is None else npc
        pcf = components.pca.PCA_frames(frames, npc=npc)
    pca_flip_signs(pcf)
    base_coords = np.array([smooth_fn(v, **fnkw) for v in pcf.coords.T]).T
    baseline_frames = pcf.inverse_transform(base_coords)
    return baseline_frames

def calculate_baseline_pca_asym(frames,
                                niter=50,
                                ncomp=20,
                                smooth=25,
                                th=1.5,
                                verbose=False):
    """Use asymetrically smoothed principal components to estimate time-varying baseline fluorescence F0"""
    frames_w = np.copy(frames)
    sh = frames.shape
    nbase = np.linalg.norm(frames)
    diff_prev = np.linalg.norm(frames_w) / nbase
    for i in range(niter + 1):
        pcf = components.pca.PCA_frames(frames_w, npc=ncomp)
        coefs = np.array([l2spline(v, smooth) for v in pcf.coords.T]).T
        rec = pcf.inverse_transform(coefs)
        diff_new = np.linalg.norm(frames_w - rec) / nbase
        epsx = diff_new - diff_prev
        diff_prev = diff_new

        if not i % 5:
            if verbose:
                sys.stdout.write('%0.1f %% | ' % (100*i/niter))
                print('explained variance %:',
                      100 * pcf.tsvd.explained_variance_ratio_.sum(), 'update: ', epsx)
        if i < niter:
            delta = frames_w - rec
            thv = th * np.std(delta, axis=0)
            frames_w = np.where(delta > thv, rec, frames_w)
        else:
            if verbose:
                print('\n finished iterations')
            delta = frames - rec
            #ns0 = np.median(np.abs(delta - np.median(delta,axis=0)), axis=0)*1.4826
            ns0 = mad_std(delta, axis=0)
            biases = find_bias_frames(delta, 3, ns0)
            biases[np.isnan(biases)] = 0
            frames_w = rec + biases    #np.array([find_bias(delta[k],ns=ns0[k]) for k,v in enumerate(rec)])[:,None]

    return frames_w

# def calculate_baseline(frames,
#                        pipeline=multi_scale_simple_baseline,
#                        stride=2,
#                        patch_size=5,
#                        return_type='array',
#                        pipeline_kw=None):
#     """
#     Given a TXY frame timestack estimate slowly-varying baseline level of fluorescence using patch-based processing
#     """
#     from imfun import fseq
#     collection = patches.signals_from_array_avg(frames,
#                                                 stride=stride,
#                                                 patch_size=patch_size)
#     recsb = process_signals_parallel(
#         collection,
#         pipeline=pipeline,
#         pipeline_kw=pipeline_kw,
#         njobs=4,
#     )
#     sh = frames.shape
#     out = patches.combine_weighted_signals(recsb, sh)
#     if return_type.lower() == 'array':
#         return out
#     fsx = fseq.from_array(out)
#     fsx.meta['channel'] = 'baseline'
#     return fsx

# def get_baseline_frames(frames,
#                         smooth=60,
#                         npc=None,
#                         baseline_fn=multi_scale_simple_baseline,
#                         baseline_kw=None):
#     """
#     Given a TXY frame timestack estimate slowly-varying baseline level of fluorescence, two-stage processing
#     (1) global trends via PCA
#     (2) local corrections by patch-based algorithm
#     """
#     from imfun import fseq
#     base1 = calculate_baseline_pca(frames,
#                                    smooth=smooth,
#                                    npc=npc,
#                                    smooth_fn=multi_scale_simple_baseline)
#     base2 = calculate_baseline(frames - base1,
#                                pipeline=baseline_fn,
#                                pipeline_kw=baseline_kw,
#                                patch_size=5)
#     fs_base = fseq.from_array(base1 + base2)
#     fs_base.meta['channel'] = 'baseline_comb'
#     return fs_base

# def _calculate_baseline_nmf(frames,
#                             ncomp=None,
#                             return_type='array',
#                             smooth_fn=multi_scale_simple_baseline):
#     """DOESNT WORK! Use smoothed NMF components to estimate time-varying baseline fluorescence F0"""
#     from imfun import fseq
#
#     fsh = frames[0].shape
#
#     if ncomp is None:
#         ncomp = len(frames) // 20
#     nmfx = skd.NMF(ncomp, )
#     signals = nmfx.fit_transform(core.ah.ravel_frames(frames))
#
#     #base_coords = np.array([smoothed_medianf(v, smooth=smooth1,wmedian=smooth2) for v in pcf.coords.T]).T
#     if smooth > 0:
#         base_coords = np.array([smooth_fn(v, smooth=smooth) for v in pcf.coords.T]).T
#         #base_coords = np.array([multi_scale_simple_baseline for v in pcf.coords.T]).T
#     else:
#         base_coords = pcf.coords
#     #base_coords = np.array([double_scale_baseline(v,smooth1=smooth1,smooth2=smooth2) for v in pcf.coords.T]).T
#     #base_coords = np.array([simple_get_baselines(v) for v in pcf.coords.T]).T
#     baseline_frames = pcf.tsvd.inverse_transform(base_coords).reshape(
#         len(pcf.coords), *pcf.sh) + pcf.mean_frame
#     if return_type.lower() == 'array':
#         return baseline_frames
#     #baseline_frames = base_coords.dot(pcf.vh).reshape(len(pcf.coords),*pcf.sh) + pcf.mean_frame
#     fs_base = fseq.from_array(baseline_frames)
#     fs_base.meta['channel'] = 'baseline_pca'
#     return fs_base

# def process_tmvm(v,
#                  k=3,
#                  level=7,
#                  start_scale=1,
#                  tau_smooth=1.5,
#                  rec_variant=2,
#                  nonnegative=True):
#     """
#     Process temporal signal using MVM and return reconstructions of significant fluorescence elevations
#     """
#     objs = mvm.find_objects(v,
#                             k=k,
#                             level=level,
#                             min_px_size=min_px_size_,
#                             min_nscales=3,
#                             modulus=not nonnegative,
#                             rec_variant=rec_variant,
#                             start_scale=start_scale)
#     if len(objs):
#         if nonnegative:
#             r = np.max(list(map(mvm.embedded_to_full, objs)), 0).astype(v.dtype)
#         else:
#             r = np.sum([mvm.embedded_to_full(o) for o in objs], 0).astype(v.dtype)
#         if tau_smooth > 0:
#             r = l2spline(r, tau_smooth)
#         if nonnegative:
#             r[r < 0] = 0
#     else:
#         r = np.zeros_like(v)
#     return r
#
#
# def tmvm_baseline(y, plow=25, smooth_level=100, symmetric=False):
#     """
#     Estimate time-varying baseline in 1D signal by first finding fast significant
#     changes and removing them, followed by smoothing
#     """
#     rec = process_tmvm(y, k=3, rec_variant=1)
#     if symmetric:
#         rec_minus = -process_tmvm(-y, k=3, rec_variant=1)
#         rec = rec + rec_minus
#     res = y - rec
#     b = l2spline(ndi.percentile_filter(res, plow, smooth_level), smooth_level / 2)
#     rsd = rolling_sd_pd(res - b)
#     return b, rsd, res
#
#
# def tmvm_get_baselines(y, th=3, smooth=100, symmetric=False):
#     """
#     tMVM-based baseline estimation of time-varying baseline with bias correction
#     """
#     b, ns, res = tmvm_baseline(y, smooth_level=smooth, symmetric=symmetric)
#     d = res - b
#     return b + np.median(d[d <= th * ns])    # + bias as constant shift
