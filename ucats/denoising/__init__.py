import sys

import numpy as np

from scipy import ndimage as ndi

import itertools as itt

from ..detection1d import simple_pipeline_
from ..masks import threshold_object_size
from ..utils import (avg_filter_greater, mad_std, smoothed_medianf, find_bias)

from ..decomposition import min_ncomp
from ..globals import _dtype_

from .block_svd_denoise_and_separate_ import block_svd_separate_tslices, block_svd_denoise_and_separate
# TODO: make more sane imports
from . import patch_svd_double_stage
from .patch_svd_double_stage import NL_Windowed_tSVD
from .patch_svd_double_stage import Multiscale_NL_Windowed_tSVD



def _patch_denoise_percentiles(data,
                               stride=2,
                               nhood=3,
                               mw=5,
                               px=50,
                               th=1.5,
                               mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh, _dtype_)
    counts = np.zeros(sh[1:], _dtype_)
    if mask_of_interest is None:
        mask = np.ones(counts.shape, bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood + 1)**2

    #preproc = lambda y: core.rescale(y)

    #tmp_signals = np.zeros()
    tv = np.arange(L)

    def _process_loc(r, c):
        sl = (slice(r - nhood, r + nhood + 1), slice(c - nhood, c + nhood + 1))
        tsl = (slice(None), ) + sl

        patch = data[tsl]
        w_sh = patch.shape
        signals = patch.reshape(sh[0], -1).T
        #print(signals.shape)

        #vm = np.median(signals,0)
        vm = np.percentile(signals, px, axis=0)
        vm = (vm - find_bias(vm)) / mad_std(vm)
        vma = simple_pipeline_(vm, smoothed_rec=True)
        # todo extend masks a bit in time?
        vma_mask = threshold_object_size(vma > 0.1, 5).astype(np.bool)

        nsv = np.array([mad_std(v) for v in signals]).reshape(-1, 1)
        pf = np.array([smoothed_medianf(v, 0.5, mw) for v in signals])
        pa = (pf > th * nsv)
        pa_txy = pa.T.reshape(w_sh)
        pa_txy2 = (ndi.median_filter(pa_txy.astype(np.float32),
                                     (3, 3, 3)) > 0) * vma_mask[:, None, None]

        labels, nl = ndi.label(pa_txy + pa_txy2)
        objs = ndi.find_objects(labels)
        pa_txy3 = np.zeros_like(pa_txy)
        for k, o in enumerate(objs):
            cond = labels[o] == k + 1
            if np.any(pa_txy2[o][cond]):
                pa_txy3[o][cond] = True

        pf_txy = pf.T.reshape(w_sh) * pa_txy3
        #pf_txy = (pf*vma_mask).
        rec = np.array([avg_filter_greater(m, 0) for m in pf_txy])
        #rec = pf_txy
        #rec = pf*vma*(pf>th*nsv)
        #score = score.reshape(w_sh[1:])
        score = 1.0
        out[tsl] += score * rec
        counts[sl] += score

    for r in range(nhood, sh[1] - nhood, stride):
        for c in range(nhood, sh[2] - nhood, stride):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d' %
                             (r, c, r * sh[1] + c + 1, np.prod(sh[1:])))
            if mask[r, c]:
                _process_loc(r, c)
    out = out / (1e-12 + counts[None, :, :])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r, c] == 0:
                out[:, r, c] = 0
    return out


def patch_pca_denoise2(data,
                       stride=2,
                       nhood=5,
                       npc=None,
                       temporal_filter=1,
                       spatial_filter=1,
                       mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh, _dtype_)
    counts = np.zeros(sh[1:], _dtype_)
    if mask_of_interest is None:
        mask = np.ones(counts.shape, bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood + 1)**2

    def _process_loc(r, c, rank):
        sl = (slice(r - nhood, r + nhood + 1), slice(c - nhood, c + nhood + 1))
        tsl = (slice(None), ) + sl
        patch = data[tsl]
        w_sh = patch.shape
        patch = patch.reshape(sh[0], -1)
        if not (np.any(patch)):
            out[tsl] += 0
            counts[sl] += 0
            return
        # (patch is now Nframes x Npixels, u will hold temporal components)
        u, s, vh = np.linalg.svd(patch, full_matrices=False)
        if rank is None:
            rank = min_ncomp(s, patch.shape) + 1
            sys.stderr.write(' | svd rank: %02d  ' % rank)
        ux = ndi.median_filter(u[:, :rank], size=(temporal_filter, 1))
        vh_images = vh[:rank].reshape(-1, *w_sh[1:])
        vhx = [
            ndi.median_filter(f, size=(spatial_filter, spatial_filter)) for f in vh_images
        ]
        vhx_threshs = [mad_std(f) for f in vh_images]
        vhx = np.array([
            np.where(np.abs(f - fx) > th, fx, f)
            for f, fx, th in zip(vh_images, vhx, vhx_threshs)
        ])
        vhx = vhx.reshape(rank, len(vh[0]))

        #print('\n', patch.shape, u.shape, vh.shape)
        #ux = u[:,:rank]
        proj = ux @ np.diag(s[:rank]) @ vhx[:rank]
        score = np.sum(s[:rank]**2) / np.sum(s**2)
        #score = 1
        rec = proj.reshape(w_sh)
        #if keep_baseline:
        #    # we possibly shift the baseline level due to thresholding of components
        #    rec += find_bias_frames(data[tsl]-rec,3,mad_std(data[tsl],0))
        out[tsl] += score * rec
        counts[sl] += score

    for r in itt.chain(range(nhood, sh[1] - nhood, stride), [sh[1] - nhood]):
        for c in itt.chain(range(nhood, sh[2] - nhood, stride), [sh[2] - nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d ' %
                             (r, c, r * sh[1] + c + 1, np.prod(sh[1:])))
            if mask[r, c]:
                _process_loc(r, c, npc)
    out = out / (1e-12 + counts[None, :, :])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r, c] == 0:
                out[:, r, c] = 0
    return out


# def patch_pca_denoise(data,stride=2, nhood=5, npc=6):
#     sh = data.shape
#     L = sh[0]
#     #if mask_of_interest is None:
#     #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
#     out = np.zeros(sh,_dtype_)
#     counts = np.zeros(sh[1:],int)
#     mask=np.ones(counts.shape,bool)
#     Ln = (2*nhood+1)**2
#     def _process_loc(r,c):
#         sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
#         tsl = (slice(None),)+sl
#         patch = data[tsl]
#         w_sh = patch.shape
#         patch = patch.reshape(sh[0],-1).T
#         Xc = patch.mean(0)
#         Xc = ndi.median_filter(Xc,3)
#         u,s,vh = np.linalg.svd(patch-Xc,full_matrices=False)
#         #ux = ndi.median_filter(u[:,:npc],size=(3,1))
#         ux = u[:,:npc]
#         proj = ux@np.diag(s[:npc])@vh[:npc]
#         out[tsl] += (proj+Xc).T.reshape(w_sh)
#         counts[sl] += 1

#     for r in range(nhood,sh[1]-nhood,stride):
#         for c in range(nhood,sh[2]-nhood,stride):
#             sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
#             if mask[r,c]:
#                 _process_loc(r,c)
#     out = out/counts[None,:,:]
#     for r in range(sh[1]):
#         for c in range(sh[2]):
#             if counts[r,c] ==0:
#                 out[:,r,c] = 0
#     return out

# def nonlocal_video_smooth(data, stride=2,nhood=5,corrfn = stats.pearsonr,mask_of_interest=None):
#     sh = data.shape
#     if mask_of_interest is None:
#         mask_of_interest = np.ones(sh[1:],dtype=np.bool)
#     out = np.zeros(sh,dtype=_dtype_)
#     mask = mask_of_interest
#     counts = np.zeros(sh[1:])
#     acc = []
#     knn_count = 0
#     cluster_count = 0
#     Ln = (2*nhood+1)**2
#     for r in range(nhood,sh[1]-nhood,stride):
#         for c in range(nhood,sh[2]-nhood,stride):
#             sys.stderr.write('\rprocessing location %05d/%d'%(r*sh[1] + c+1, np.prod(sh[1:])))
#             if mask[r,c]:
#                 v = data[:,r,c]
#                 kcenter = 2*nhood*(nhood+1)
#                 sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
#                 patch = data[(slice(None),)+sl]
#                 w_sh = patch.shape
#                 patch = patch.reshape(sh[0],-1).T
#                 weights = np.array([corrfn(a,v)[0] for a in patch])**2
#                 weights = weights/np.sum(weights)
#                 wx = weights.reshape(w_sh[1:])
#                 ks = np.argsort(weights)[::-1]
#                 xs = ndi.median_filter(patch, size=(5,1))
#                 out[(slice(None),)+sl] += xs[np.argsort(ks)].T.reshape(w_sh)*wx[None,:,:]
#                 counts[sl] += wx
#     out /= counts
#
#     return out

# from sklearn import decomposition as skd
# from skimage import filters as skf
# def _patch_denoise_nmf(data,stride=2, nhood=5, ncomp=None,
#                        smooth_baseline=False,
#                        max_ncomp=None,
#                        temporal_filter = None,
#                        mask_of_interest=None):
#     sh = data.shape
#     L = sh[0]
#     if max_ncomp is None:
#         max_ncomp = 0.25*(2*nhood+1)**2
#     #if mask_of_interest is None:
#     #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
#     out = np.zeros(sh,_dtype_)
#     counts = np.zeros(sh[1:],_dtype_)
#     if mask_of_interest is None:
#         mask=np.ones(counts.shape,bool)
#     else:
#         mask = mask_of_interest
#     Ln = (2*nhood+1)**2

#     #preproc = lambda y: core.rescale(y)

#     #tmp_signals = np.zeros()
#     tv = np.arange(L)

#     def _process_loc(r,c):
#         sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
#         tsl = (slice(None),)+sl

#         patch = data[tsl]
#         lift = patch.min(0)

#         patch = patch-lift # precotion against negative values in data

#         X = patch.reshape(L,-1)
#         u,s,vh = svd(X,False)
#         rank = min(np.min(X.shape), min(max_ncomp, min_ncomp(s,X.shape) + 1)) if ncomp is None else ncomp
#         if ncomp is None:
#             sys.stderr.write('  rank: %d  '%rank)

#         d = skd.NMF(rank,l1_ratio=0.95,init='nndsvdar')#,beta_loss='kullback-leibler',solver='mu')
#         nmf_signals = d.fit_transform(X).T
#         nmf_comps = np.array([m*opening_of_closing(m > 0.5*skf.threshold_otsu(m)) for m in d.components_])

#         #nmf_biases = np.array([find_bias(v) for v in nmf_signals]).reshape(-1,1)
#         #nmf_biases = np.array([multi_scale_simple_baseline(v) for v in nmf_signals])
#         if smooth_baseline:
#             nmf_biases = np.array([simple_baseline(v,50,smooth=50) for v in nmf_signals])
#         else:
#             nmf_biases = np.array([find_bias(v,ns=mad_std(v)) for v in nmf_signals]).reshape(-1,1)
#         nmf_signals_c = nmf_signals - nmf_biases

#         nmf_signals_fplus = np.array([v*percentile_label(v,percentile_low=25,tau=1.5) for v in nmf_signals_c])
#         nmf_signals_fminus = np.array([v*percentile_label(-v,percentile_low=25) for v in nmf_signals_c])
#         nmf_signals_filtered = nmf_signals_fplus + nmf_signals_fminus + nmf_biases

#         rec = nmf_signals_filtered.T@nmf_comps
#         rec_frames = rec.reshape(*patch.shape)
#         rec_frames += find_bias_frames(patch-rec_frames,3,mad_std(patch,0)) # we possibly shift the baseline level due to thresholding of components

#         #print(out[tsl].shape, patch.shape, rec.shape)
#         out[tsl] += rec_frames + lift

#         score = 1.0
#         counts[sl] += score

#     for r in itt.chain(range(nhood,sh[1]-nhood,stride), [sh[1]-nhood]):
#         for c in itt.chain(range(nhood,sh[2]-nhood,stride), [sh[2]-nhood]):
#             sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
#             if mask[r,c]:
#                 _process_loc(r,c)
#     out = out/(1e-12+counts[None,:,:])
#     for r in range(sh[1]):
#         for c in range(sh[2]):
#             if counts[r,c] == 0:
#                 out[:,r,c] = 0
#     return out
