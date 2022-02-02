# Transient labeling algorithms
import numpy as np

import itertools as itt
from functools import partial

from numba import jit

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

from imfun.filt.dctsplines import l2spline, sp_decompose, l1spline
from imfun.core import extrema
from imfun import bwmorph

from ucats import baselines
from .scramble import local_jitter
from .utils import smoothed_medianf, rolling_sd_pd, find_bias


def percentile_label(v, percentile_low=2.5, tau=2.0, smoother=l2spline):
    mu = min(np.median(v), 0)
    low = np.percentile(v[v <= mu], percentile_low)
    vs = smoother(v, tau)
    return vs >= mu - low


def simple_label(v, threshold=1.0, tau=5., smoother=l2spline, **kwargs):
    vs = smoother(v, tau)
    return vs >= threshold


def with_local_jittering(labeler, niters=100, weight_thresh=0.85):
    def _(v, *args, **kwargs):
        if 'tau' in kwargs:
            tau = kwargs['tau']
        else:
            tau = 5.0
        labels_history = np.zeros((niters, len(v)))
        for i_ in range(niters):
            vi = local_jitter(v, 0.5 * tau)
            #labels_history.append(labeler(vi, *args, **kwargs))
            labels_history[i_] = labeler(vi, *args, **kwargs)
        return np.mean(labels_history, 0) >= weight_thresh

    return _


simple_label_lj = with_local_jittering(simple_label)
percentile_label_lj = with_local_jittering(percentile_label)

thresholds_l1 = np.array([
    2.26212451, 1.11505896, 0.52321721, 0.51701626, 0.42481402, 0.34870014, 0.29144794,
    0.24410656, 0.20409004, 0.16792375, 0.13579082, 0.10770976
])
thresholds_l1 = thresholds_l1.reshape(-1, 1)

thresholds_l2 = np.array([
    1.6452271, 0.64617428, 0.41641932, 0.32425908, 0.26115802, 0.21203462, 0.17222229,
    0.14062114, 0.11350558, 0.0896438, 0.06936852, 0.05300952
])
thresholds_l2 = thresholds_l2.reshape(-1, 1)


def multiscale_labeler_l1(signal, thresh=2, start=1, **kwargs):
    coefs = sp_decompose(signal, level=12, smoother=l1spline, base=1.5)[start:-1]
    labels = (coefs >= thresholds_l1[start:]).sum(axis=0) >= thresh
    return labels


def multiscale_labeler_l2(signal, thresh=4, start=1, **kwargs):
    #thresholds_l2 = array([1.6453141 , 0.64634246, 0.41638476, 0.3242796 , 0.2611729 ,
    #                       0.21204839, 0.17224974, 0.14053809, 0.11334435, 0.08955742,
    #                       0.06948411, 0.05307127]).reshape(-1,1)
    coefs = sp_decompose(signal, level=12, smoother=l2spline, base=1.5)[start:-1]
    labels = (coefs >= thresholds_l2[start:]).sum(axis=0) >= thresh
    return labels


def simple_pipeline_(y,
                     labeler=percentile_label,
                     labeler_kw=None,
                     smoothed_rec=True,
                     noise_sigma=None,
                     correct_bias=True,
                     rec_kw = None,
                     ):
    """
    Detect and reconstruct Ca-transients in 1D signal
    """
    if not any(y):
        return np.zeros_like(y)

    ns = rolling_sd_pd(y) if noise_sigma is None else noise_sigma

    if correct_bias:
        bias = find_bias(y, th=1.5, ns=ns)
        y = y - bias
    vn = y / ns
    #labels = simple_label_lj(vn, tau=tau_label_,with_plots=False)
    if labeler_kw is None:
        labeler_kw = {}
    labels = labeler(vn, **labeler_kw)
    if not any(labels):
        return np.zeros_like(y)
    if rec_kw is None:
        rec_kw = {}
    return sp_rec_with_labels(vn, labels, with_plots=False,
                              return_smoothed=smoothed_rec,
                              **rec_kw,
                              ) * ns


def sp_rec_with_labels(vec,
                       labels,
                       min_scale=1.0,
                       max_scale=50.,
                       with_plots=True,
                       min_size=3,
                       niters=10,
                       kgain=0.25,
                       smoother=smoothed_medianf,
                       wmedian=3,
                       return_smoothed=False):
    if min_size > 1:
        regions = bwmorph.contiguous_regions(labels)
        regions = bwmorph.filter_size_regions(regions, min_size)
        filtered_labels = np.zeros_like(labels) + np.sum([r.tomask() for r in regions],
                                                         axis=0)
    else:
        filtered_labels = labels

    if not sum(filtered_labels):
        return np.zeros_like(vec)
    vec1 = np.copy(vec)
    vs = smoother(vec1, min_scale, wmedian)
    weights = np.clip(labels, 0, 1)

    #vss = smoother(vec-vs,max_scale,weights=weights<0.9)

    vrec = smoother(vs * (vec1 > 0), min_scale, wmedian)

    for i in range(niters):
        #vec1 = vec1 - kgain*(vec1-vrec) # how to use it?
        labs, nl = ndi.label(weights)
        objs = ndi.find_objects(labs)
        #for o in objs:
        #    stop = o[0].stop
        #    while stop < len(vec) and vrec[stop]>0.25*vrec[o].max():
        #        weights[stop] = 1
        #        stop+=1
        wer_grow = ndi.binary_dilation(weights)
        wer_shrink = ndi.binary_erosion(weights)
        #weights = np.where((vec1<np.mean(vec1[vec1>0])), wer, weights)
        if np.any(vrec > 0):
            weights = np.where(vrec < 0.5 * np.mean(vrec[vrec > 0]), wer_shrink, weights)
            weights = np.where(vrec > 1.25 * np.mean(vrec[vrec > 0]), wer_grow, weights)
        vrec = smoother(vec * weights, min_scale, wmedian)
        #weights = ndi.binary_opening(weights)
        vrec[vrec < 0] = 0
        #plt.figure(); plt.plot(vec1)
        #vrec[weights<0.5] *=0.5

    if with_plots:
        f, ax = plt.subplots(1, 1)
        ax.plot(vec, '-', ms=2, color='gray', lw=0.5, alpha=0.5)
        ax.plot(vec1, '-', color='cyan', lw=0.75, alpha=0.75)
        ax.plot(weights, 'g', lw=2, alpha=0.5)
        ax.plot(vs, color='k', alpha=0.5)
        #plot(vss,color='navy',alpha=0.5)
        ax.plot(vrec, color='royalblue', lw=2)
        ll = np.where(labels)[0]
        ax.plot(ll, -1 * np.ones_like(ll), 'r|')
    if return_smoothed:
        return vrec
    else:
        return vec * (vrec > 0) * (vec > 0) * weights    #weights*(vrec>0)*vec


def simple_pipeline_nojitter_(y, tau_label=1.5):
    """
    Detect and reconstruct Ca-transients in 1D signal
    """
    ns = rolling_sd_pd(y)
    low = y < (2.5 * np.median(y))
    if not any(low):
        low = np.ones(len(y), np.bool)
    bias = np.median(y[low])
    if bias > 0:
        y = y - bias
    vn = y / ns
    labels = simple_label(vn, tau=tau_label, with_plots=False)
    return y * labels
    #return sp_rec_with_labels(vn, labels,niters=5,with_plots=False)*ns


def segment_events_1d(rec, th=0.05, th2=0.1, smoothing=6, min_lenth=3):
    levels = rec > th
    labeled, nlab = ndi.label(levels)
    smrec = l1spline(rec, smoothing)
    #smrec = l2spline(rec, 6)
    mxs = np.array(extrema.locextr(smrec, output='max', refine=False))
    mns = np.array(extrema.locextr(smrec, output='min', refine=False))
    if not len(mxs) or not len(mns) or not np.any(mxs[:, 1] > th2):
        return labeled, nlab
    mxs = mxs[mxs[:, 1] > th2]
    cuts = []

    for i in range(1, nlab + 1):
        mask = labeled == i
        lmax = [m for m in mxs if mask[int(m[0])]]
        if len(lmax) > 1:
            th = np.max([m[1] for m in lmax]) * 0.75
            lms = [mn for mn in mns if mask[int(mn[0])] and mn[1] < th]
            if len(lms):
                for lm in lms:
                    tmp_mask = mask.copy()
                    tmp_mask[int(lm[0])] = 0
                    ll_, nl_ = ndi.label(tmp_mask)
                    min_region = np.min([np.sum(ll_ == i_) for i_ in range(1, nl_ + 1)])
                    if min_region > min_lenth:
                        cuts.append(lm[0])
                        levels[int(lm[0])] = False

    labeled, nlab = ndi.label(levels)

    #plot(labeled>0)
    return labeled, nlab


def simple_pipeline_with_baseline(y, tau_label=1.5):
    """
    Detect and reconstruct Ca-transients in 1D signal after normalizing to baseline
    #b,ns,_ = tmvm_baseline(y)
    """
    #b = b + np.median(y-b)
    ns = rolling_sd_pd(y)
    b = baselines.multi_scale_simple_baseline(y, ns=ns)
    vn = (y-b) / ns
    labels = simple_label_lj(vn, tau=tau_label, with_plots=False)
    rec = sp_rec_with_labels(
        y,
        labels,
        with_plots=False,
    )
    return np.where(b > 0, rec, 0)


def make_labeler_commitee(*labelers):
    Nl = len(labelers)

    def _(v, **kwargs):
        labels = [lf(v) for lf in labelers]
        return np.sum(labels, 0) == Nl

    return _


multiscale_labeler_l1l2 = make_labeler_commitee(
    multiscale_labeler_l1, partial(multiscale_labeler_l2, start=2, thresh=3))

multiscale_labeler_joint = make_labeler_commitee(
    multiscale_labeler_l1, partial(multiscale_labeler_l2, start=2, thresh=3),
    simple_label_lj)
