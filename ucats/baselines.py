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

from .globals import _dtype_

from .decomposition import pca_flip_signs, SVD_patch
from .decomposition import Windowed_tSVD

from .patches import make_grid

from .utils import rolling_sd_pd, find_bias, find_bias_frames
from .utils import mad_std, make_odd
from .utils import process_signals_parallel
from .utils import iterated_tv_chambolle
from .utils import find_jumps
from .utils import pixelwise_smoothed_apply


def store_baseline_pickle(name, frames, ncomp=50):
    pcf = components.pca.PCA_frames(frames, npc=ncomp)
    pickle.dump(pcf, open(name, 'wb'))


def load_baseline_pickle(name):
    pcf = pickle.load(open(name, 'rb'))
    return pcf.inverse_transform(pcf.coords)


@jit(nopython=True)
def running_min(v):
    out = np.zeros(v.shape)
    mn = v[0]
    for i in range(len(v)):
        mn = min(mn, v[i])
        out[i] = mn
    return out


@jit(nopython=True)
def running_envelope(v):
    out = np.zeros(v.shape+(2,))
    mn = v[0]
    mx = v[1]
    for i in range(len(v)):
        mn = min(mn, v[i])
        mx = max(mx,v[i])
        out[i,0] = mn
        out[i,1] = mx
    return out

def tight_running_envelope(v):
    forward = running_envelope(v)
    backward = running_envelope(v[::-1])[::-1]
    out = np.zeros(v.shape + (2,))
    out[:,0] = np.maximum(forward[:,0],backward[:,0])
    out[:,1] = np.minimum(forward[:,1],backward[:,1])
    return out
    #return np.maximum(running_min(v), running_min(v[::-1])[::-1])

def local_extr1(v, kind='max'):
    "simple local extrema, todo: move to utils"
    
    dy = np.diff(v)
    dy_sign = np.sign(dy)
    operator = np.greater if kind == 'min' else np.less
    cond1 = (operator(dy_sign[1:],dy_sign[:-1]) )
    return np.where(cond1)[0] + 1

def robust_line(y, x=None, ns=None, th=3, niter=3):
    """
    A simple iterative take on robust linear regression
    """
    if ns is None:
        ns = np.std(np.diff(y))/2**0.5
    if x is None:
        x = np.arange(len(y))
    weights = np.ones(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    for i in range(niter):
        #p = np.polyfit(x,v,1,w=weights)
        #fit = np.polyval(p, x)
        k,b = np.linalg.lstsq(A*weights[:,None], y, rcond=None)[0]
        fit = k*x + b
        weights = np.where(np.abs(fit-y) > th*ns, 0, 1)
    return k,b


def correct_bias(v, baseline):
    resid = v - baseline
    bias = find_bias(resid)
    return baseline+bias


def lower_envelope(v, use_shoulders=True, x = None):
    """
    Make linear interpolation going via local minima points
    Optionally use "shoulders" along with minima.
    """
    if x is None:
        x = np.arange(len(v))
    lmmin = local_extr1(v,'min')
    if use_shoulders:
        d2 = np.diff(v, 2)
        shoulders = local_extr1(d2, 'max') + 1
        lmmin = np.concatenate([[0], lmmin, shoulders, [len(v)-1]])
        lmmin = np.sort(lmmin)
    else:
        lmmin = np.concatenate([[0], lmmin, [len(v)-1]])
    #print(len(lmmin))
    v = np.interp(x, lmmin, v[lmmin])
    return v, lmmin

def baseline_smoothed_minima(y, pre_smooth=10, post_smooth=25, niters=1, use_shoulders=True):
    """
    Estimate baseline as smoothed linear interpolation, going through local minima
    """
    x = np.arange(len(y))
    for i in range(niters):
        if pre_smooth > 0 :
            ys = ndi.median_filter(y, pre_smooth)
            yss = ndi.gaussian_filter1d(ys,pre_smooth)
        else:
            yss = y

        bdown = lower_envelope(yss, use_shoulders=use_shoulders,x=x)[0]

        if post_smooth > 0:
            y = l1spline(bdown, post_smooth)
        else:
            y = bdown
    return y

def baseline_iterated_smoothed_minima(y, pre_smooth=11, post_smooth=11, niters=2,
                                      use_shoulders=False, plow=50, noise_sigma=None,
                                      sigma_range=1.5, wsize=300):
    """
    Estimate baseline as smoothed linear interpolation going through local minima
    Iteratively substitutes values in the signal by current baseline estimate if they 
    fall outside a sigma_range x noise_sigma range
    """
    x = np.arange(len(y))

    if pre_smooth > 0 :
        ys = ndi.percentile_filter(y, percentile=plow, size=pre_smooth, )
        yss = ndi.gaussian_filter1d(ys,pre_smooth/2)
        if noise_sigma is None:
            noise_sigma = mad_std(y-yss)
    else:
        yss = y

    low2 = windowed_runmin(yss, wsize=wsize)
    if noise_sigma is None:
        noise_sigma = mad_std(y-low2)
    yss[-1] = np.mean(low2[-len(y)//2:])
    yss[0] = np.mean(low2[:len(y)//2])

    for i in range(niters):

        bdown, tpoints = lower_envelope(yss, use_shoulders=use_shoulders,x=x)

        uth = bdown + sigma_range*noise_sigma
        lth = bdown - sigma_range*noise_sigma

        yss = np.where((y < lth)|(y > uth), bdown, y)
        yss = ndi.gaussian_filter1d(yss, max(1.5, pre_smooth/2))
        baseline = bdown

    if post_smooth > 0:
        baseline = l1spline(baseline, post_smooth)

    return baseline

def baseline_smoothed_filtered_minima(y, pre_smooth=5, post_smooth=25, plow=50, wsize=300,
                                      noise_sigma=None, sigma_range=1.5,
                                      smoother=l2spline,
                                      do_padding=False,
                                      do_lower_endpoints=True,
                                      do_remove_linear_trend=True):
    """
    Estimate baseline as smoothed linear interpolation going through local minima of the signal.
    Optionally, fit and remove general linear trend before estimation. 
    Only use local minima that are within a sigma_range times noise standard deviation from some 
    draft baseline estimate (top_hat of low-hat of y)
    """
    x = np.arange(len(y))

    if noise_sigma is None:
        noise_sigma = np.std(np.diff(y))/np.sqrt(2)


    if pre_smooth > 0 :
        ys = ndi.percentile_filter(y, percentile=plow, size=pre_smooth, )
        yss = ndi.gaussian_filter1d(ys,pre_smooth/2)
    else:
        yss = y

    if do_remove_linear_trend:
        px = robust_line(y, x=x, ns=noise_sigma, niter=5,)
        trend = px[0]*x + px[1]
    else:
        trend = np.zeros_like(yss)

    yss = yss - trend

    L = len(y)
    #yss[0], yss[-1] = 0,0
    if do_lower_endpoints:
        # this is questionable though:
        yss[0] = min(yss[0], np.percentile(yss[:L//2],25))
        yss[-1] = min(yss[-1], np.percentile(yss[L//2:],25))

    npad = wsize//2
    if do_padding:
        yss_p =  pybaselines.utils.pad_edges(yss, npad,)
    else:
        yss_p = yss

    low1 = windowed_runmin(yss_p, wsize=wsize)
    low1 = -windowed_runmin(-low1, wsize=wsize//2)

    if do_padding:
        low1 = low1[npad:-npad]


    minima = local_extr1(yss, 'min')
    minima = minima[np.abs(yss[minima]-low1[minima]) <= sigma_range*noise_sigma]
    minima = np.concatenate([[0], minima, [len(yss)-1]])
    baseline = np.interp(x, minima, yss[minima])


    if post_smooth > 0:
        baseline = smoother(baseline, post_smooth)

    return baseline + trend

import os
def stack_baseline(frames, name,
                   pre_smooth = 5, post_smooth=25,
                   need_rebuild=False,
                   wsize=300,
                   smoother=l2spline,
                   ncomp_out=32,
                   fn1 = partial(baseline_smoothed_filtered_minima, post_smooth=0),
                   suff='',
                   **kwargs):

    #baseline_name = os.path.splitext(name)[0] + '-baselines-new-pcf.p'
    base, ext = os.path.splitext(name)
    # older code uses this
    if ext in ['.tif', '.lsm']:
        name = base

    baseline_name = name + suff + '-baselines-new-pcf.p'
    pcf = None

    fn1 = partial(fn1, **kwargs)

    if not os.path.exists(baseline_name) or need_rebuild:
        x = np.arange(len(frames))
        #fn1 = lambda v: lower_envelope(v, use_shoulders=False, x=x)[0]
        #fn2 = lambda v: lower_envelope(v, use_shoulders=True, x=x)[0]
        #fn1 = lambda v: baseline_smoothed_envelope(v, pre_smooth=0, post_smooth=0, niters=niters)
        #fn1 = lambda v: baseline_smoothed_minima2(v, pre_smooth=pre_smooth, post_smooth=0, niters=niters)
        #-fn1 = lambda v: baseline_smoothed_minima3(v, pre_smooth=pre_smooth, post_smooth=0, wsize=wsize, **kwargs)
        
        stage1 = pixelwise_smoothed_apply(frames, fn1, pre_smooth=0, tqdm_msg='stage1')
        #F0 = pixelwise_smoothed_apply(stage1, lambda v: ubase.l1spline(v, post_smooth), pre_smooth=0, output=stage1, tqdm_msg='post-smooth')
        F0 = pixelwise_smoothed_apply(stage1, lambda v: smoother(v, post_smooth), pre_smooth=0, output=stage1, tqdm_msg='post-smooth')
        #stage2 = stage1
        #for i in tqdm(range(stage2_iters), desc='stage2'):
        #    stage2 = pixelwise_smoothed_apply(stage2, fn2, pre_smooths[1], output = stage2, tqdm_msg='stage2/%02d'%i)


        # F0 = pixelwise_smoothed_apply(stage2, lambda v:ubase.l1spline(v, post_smooth), pre_smooth=0, output=stage2,
        #                               tqdm_msg='post-smooth')

        residuals = frames-F0
        ns = np.std(np.diff(residuals,axis=0),axis=0)/np.sqrt(2)

        bias = find_bias_frames(residuals,3, ns=ns)
        F0 = F0 + 0.5*bias

        pcf = PCA_frames(F0,npc=ncomp_out)
        pickle.dump(pcf, open(baseline_name,'wb'))
    else:
        pcf = pickle.load(open(baseline_name,'rb'))

    F0a = pcf.inverse_transform(pcf.coords)
    F0a = np.maximum(F0a,0)
    return F0a



def top_running_min(v):
    return np.maximum(running_min(v), running_min(v[::-1])[::-1])

def windowed_envelope(y, wsize=50, woverlap=25):
    L = len(y)
    if woverlap >= wsize:
        woverlap = wsize // 2
    sqs = make_grid((L, ), wsize, woverlap)
    out  = np.zeros((L,2))
    out[:,0] =  np.max(y) + 1
    out[:,1] = np.min(y) - 1
    for sq in sqs:
        sl = sq[0]
        #bx = top_running_min(y[sl])
        env = tight_running_envelope(y[sl])
        out[sl][:,0] = np.minimum(out[sl][:,0], env[:,0])
        out[sl][:,1] = np.maximum(out[sl][:,1], env[:,1])
    return out

def windowed_runmin(y, wsize=50, woverlap=None):
    L = len(y)
    if woverlap is None or woverlap >= wsize:
        woverlap = wsize // 2
    sqs = make_grid((L, ), wsize, woverlap)
    out = (np.max(y) + 0.1) * np.ones(L, _dtype_)
    for sq in sqs:
        sl = sq[0]
        bx = top_running_min(y[sl])
        out[sl] = np.minimum(out[sl], bx)
    return out

# def two_way_percentile(y, wsize=50, percentile=25):
#     forward = ndi.percentile_filter(y, percentile, wsize)
#     backward = ndi.percentile_filter(y[::-1],percentile, wsize)[::-1]
#     return np.maximum(forward, backward)


from scipy.stats import skew

def iterated_symm_runmin(v, niters=10, w=350,
                         pre_smooth=10,
                         post_smooth=25):
    out = v
    if pre_smooth > 0:
        out = l1spline(out, pre_smooth)

    for i in range(niters):
        #top = -windowed_runmin(-out, w, w//2)
        #bot = windowed_runmin(out, w, w//2)
        env = windowed_envelope(out, w, w//2)
        m = np.mean(env,1)
        #sk = np.sign(skew(v-m))
        sk = np.sign(np.sum(v-m) - np.sum(m-v))
        if sk > 0:
            m = 0.5*m + 0.5*env[:,0]
        elif sk < 0:
            m = 0.5*m + 0.5*env[:,1]
        out = m
    if post_smooth > 0:
        out = l1spline(out, post_smooth)
    return out

def iterated_flat_runmin(y, wb=200, wt=50, niter=1):
    ynext = y
    bb = np.zeros_like(y) + np.min(y)
    for i in range(niter):
        t = -windowed_runmin(-ynext,wt)
        tu = windowed_runmin(t, wt)
        yflat = ynext - tu

        b = -windowed_runmin(-windowed_runmin(yflat,wb),wb)
        t2 = windowed_runmin(-windowed_runmin(-yflat,wt),wt)

        ynext = tu + np.clip(yflat, b, t2)
    return -windowed_runmin(-windowed_runmin(ynext,wb),wb)


def select_most_stable_branch(variants, window=50, woverlap=None):
    L = len(variants[0])
    if woverlap is None:
        woverlap = max(2, window // 3)

    if woverlap >= window:
        woverlap = window // 2
    squares = make_grid((L, 1), window, woverlap)
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


def iterated_smoothing_baseline(y, niter=10, th=3, smooth_fn=l1spline, asym=True, fnkw=None):
    """Baseline from iterated thresholded smoothing"""
    if fnkw is None:
        fnkw = {}
    ytemp = y
    ns = mad_std(np.diff(y))
    for i_ in range(niter):
        ys = smooth_fn(ytemp, **fnkw)
        cond = (y > ys + ns*th) if asym else (np.abs(y - ys) > ns * th)
        ytemp = np.where(cond, ys, y)
        #ytemp = np.where(np.abs(ytemp - ys) < ns * th, y, ys)
    return ys


def iterated_smoothing_baseline2(y,
                                 niter=10,
                                 th=1.5,
                                 noise_sigma=None,
                                 asym=False,
                                 smooth_fn=l1spline,
                                 smooth_fn2=None,
                                 fnkw=None,
                                 fnkw2=None):

    fnkw = dict() if fnkw is None else fnkw
    fnkw2 = fnkw if fnkw2 is None else fnkw2
    smooth_fn2 = smooth_fn if smooth_fn2 is None else smooth_fn2


    ytemp = y

    #print (ytemp.shape, fnkw['axis'],fnkw2['axis'])

    ns = mad_std(np.diff(y)) if (noise_sigma is None) else noise_sigma

    for i_ in range(niter):
        ys = smooth_fn(ytemp, **fnkw)
        ys2 = smooth_fn2(ytemp, **fnkw2)
        cond = (y > ys) if asym else (np.abs(y - ys) > ns * th)
        ytemp = np.where(cond, ys2, y)
    return ys


def iterated_l1_baseline(y, smooth1=10, smooth2=25, **kwargs):
    fnkw1, fnkw2 = (dict(s=s) for s in (smooth1, smooth2))
    return iterated_smoothing_baseline2(y,
                                        smooth_fn=l1spline,
                                        fnkw=fnkw1,
                                        fnkw2=fnkw2,
                                        **kwargs)


def iterated_savgol_baseline2(y,
                              window=99,
                              window2=None,
                              order=3,
                              order2=3,
                              post_smooth=5,
                              axis=None,
                              **kwargs):

    window = make_odd(np.minimum(len(y) - 1, window))
    window2 = window*4 - 1 if not window2 else window2
    window2 = make_odd(np.minimum(window2, len(y) - 1))

    fnkw1, fnkw2 = (dict(window_length=w, polyorder=k)
                    for w, k in zip((window, window2), (order, order2)))

    if axis is not None:
        fnkw1['axis'] = axis
        fnkw2['axis'] = axis

    b = iterated_smoothing_baseline2(y,
                                     smooth_fn=signal.savgol_filter,
                                     fnkw=fnkw1, fnkw2=fnkw2,
                                     **kwargs)
    if post_smooth > 0:
        b = l2spline(b, post_smooth)
    return b


def percentile_baseline(y,
                        plow=25,
                        percentile_window=25,
                        out_smooth=25,
                        smoother=l2spline,
                        ns=None,
                        th=3,
                        npad=None):
    """
    Use percentile filtering to estimate slowly changing baseline level.
    Output of the percentile filter is further smoothed with `smoother` function and
    `out_smooth` parameter.
    """
    L = len(y)

    npad = percentile_window // 2 if (npad is None) else npad
    ypad = np.pad(y, npad, 'median', stat_length=min(L, 10)) if npad > 0 else y

    b = smoother(ndi.percentile_filter(ypad, plow, percentile_window), out_smooth)
    b = b[npad:L + npad]
    #if ns is None:
    #    ns = rolling_sd_pd(y)
    d = y - b
    if ns is None:
        ns = mad_std(y-b)
    if not np.any(ns):
        ns = np.std(d)
    bg_points = d[np.abs(d) <= th * ns]
    if len(bg_points) > 10:
        b = b + np.median(bg_points)    # correct scalar shift
    return b


# def baseline_with_shifts(y, l1smooth=25):
#     ys_l1 = l1spline(y, l1smooth)
#     ns = mad_std(y - ys_l1)
#     ys = iterated_tv_chambolle(y, 1 * ns, 5)
#     jump_locs, shift = find_jumps(ys, ys_l1, pre_smooth=1.5)
#     trend = l1_baseline2(y - shift, l1smooth)
#     baseline = trend + shift
#     return baseline

def baseline_with_shifts(y, l1smooth=25, trend_kw=None, with_plot=False):
    if trend_kw is None:
        trend_kw = dict()
    #ys_l1 = l1spline(y, 25)
    #ns = mad_std(y - ys_l1)
    #trend = l1_baseline2(y-shift)
    #ys = iterated_tv_chambolle(y, 1 * ns, 5)
    jump_locs, shift = find_jumps(y, pre_smooth=1.5)
    trend = iterated_savgol_baseline2(y - shift, th=1.5, window=199)
    baseline = trend + shift
    if with_plot:
        from matplotlib import pyplot as plt
        plt.figure(figsize=(16, 8))
        plt.plot(y, 'gray')
        #plt.plot(ys, 'k')
        for j in jump_locs:
            plt.axvline(j, color='pink', lw=0.5)
        plt.plot(y - shift, lw=0.5)
        plt.plot(trend, ls='--', color='gray', label='trend estimate')
        plt.plot(trend + shift, lw=3, c='m', label='baseline estimate')
        plt.plot(shift, color='steelblue', lw=0.5, label='shift estimate')
        plt.legend()
    return baseline


def first_pc_baseline(frames,
                      niters=10,
                      baseline_fn=l1_baseline2,
                      fnkw=None,
                      verbose=False):
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


def multi_scale_percentile_baseline(y,
                                plow=50,
                                th=3,
                                smooth_levels=(10, 20, 40, 80, 160),
                                ns=None):
    if ns is None:
        ns = rolling_sd_pd(y)

    b_estimates = [
        percentile_baseline(y, plow, th, smooth, ns) for smooth in smooth_levels
    ]

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
    """Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005),
    via DCT-based spline smoothing
    """
    #npad=int(smooth)
    nsmooth = int(np.ceil(smooth))
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
    ntau = int(np.ceil(tau))
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
                        smooth_fn=iterated_savgol_baseline2,
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


def rolled_baseline(v, nrolls=5, baseline_fn=iterated_savgol_baseline2,  **kwargs):
    """
    NOTE: gives wrong results at ends if there is a large trend in data
    """
    L = len(v)
    rolls = np.arange(0, L, L//nrolls)

    rolled_baselines = ((r,baseline_fn(np.roll(v, r),**kwargs)) for r in rolls)
    baseline_estimates = [np.roll(b,-r) for r,b in rolled_baselines]

    #TODO: better estimate than median?
    bmed = np.median(baseline_estimates, axis=0)
    #bmean = np.mean(baseline_estimates, axis=0)
    return bmed

def patch_tsvd_baseline(frames,
                        ssize:"spatial window size"=32,
                        soverlap:"overlap between windows"=4,
                        max_ncomps:"maximum number of SVD components"=5,
                        smooth_fn:"smoothing function for baseline"=iterated_savgol_baseline2,
                        axis_trick=False,
                        center_data:"subtract mean before SVD"=True,
                        fnkw:"arguments to pass to the smoothing function"=None,
                        verbose=False):
    """
    Use smoothed principal components in spatial windows to estimate time-varying baseline fluorescence F0
    """
    if fnkw is None and smooth_fn is iterated_savgol_baseline2:
        fnkw = dict(window=100, window2=250, order=3, order2=3, th=1.5)

    #print(f"ssize:{ssize}, soverlap:{soverlap}")
    wtsvd = Windowed_tSVD(patch_ssize=ssize,
                          patch_tsize=len(frames),
                          soverlap=soverlap,
                          center_data=center_data,
                          max_ncomps=max_ncomps,
                          verbose=verbose)
    svd_patches = wtsvd.fit_transform(frames)
    out_coll = []
    for p in tqdm(svd_patches, desc='doing baselines'):
        if axis_trick and smooth_fn==iterated_savgol_baseline2:
            baselines = smooth_fn(p.signals.T, axis=0, **fnkw).T
        else:
            baselines = np.array([smooth_fn(v, **fnkw) for v in p.signals])
        out_coll.append(p._replace(signals=baselines))
    baseline_frames = wtsvd.inverse_transform(out_coll)
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
