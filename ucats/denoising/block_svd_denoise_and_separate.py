"""
Use truncated SVD in patches to simultaneously denoise and separate signal
from slowly varying baseline in framestack
"""

import sys

import numpy as np

from functools import partial
import itertools as itt

from scipy import ndimage as ndi

from sklearn import cluster as skclust

from imfun import cluster


from ..decomposition import min_ncomp
from ..patches import make_weighting_kern
from ..baselines import find_bias,percentile_baseline
from ..utils import smoothed_medianf, mad_std
from ..globals import _dtype_
from ..masks import threshold_object_size
from ..detection1d import percentile_label

def correct_small_loads(points, affs, min_loads=5, niter=1):
    for j in range(niter):
        new_affs = np.copy(affs)
        labels = np.unique(affs)
        loads = np.array([np.sum(affs==k) for k in labels])
        if not np.any(loads < min_loads):
            break
        centers = np.array([np.mean(points[affs==k],0) for k in labels])
        point_ind = np.arange(len(points))
        for li in np.where(loads < min_loads)[0]:
            cond = affs==labels[li]
            for point,ind in zip(points[cond], point_ind[cond]):
                dists = cluster.metrics.euclidean(point,centers)
                dists[loads < min_loads] = np.amax(dists)+1000
                k = np.argmin(dists)
                new_affs[ind] = k

        affs = new_affs.copy()
    return new_affs

from sklearn import cluster as skclust

def block_svd_denoise_and_separate(data, stride=2, nhood=5,
                                   ncomp=None,
                                   min_comps = 1,
                                   max_comps = None,
                                   spatial_filter=1,
                                   spatial_filter_th=5,
                                   temporal_filter = 0,
                                   spatial_min_cluster_size=7,
                                   baseline_smoothness=100,
                                   svd_detection_plow=25,
                                   cluster_detection_plow=5,
                                   correct_spatial_components = True,
                                   with_clusters=False,
                                   only_truncated_svd = False,
                                   mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out_signals = np.zeros(sh,_dtype_)
    out_baselines = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    counts_b = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2

    if max_comps is None:
        max_comps = (nhood**2)/2

    patch_size=nhood*2
    wk = make_weighting_kern(patch_size,2.5)

    # TODO: вынести во внешний namespace?
    def _process_loc(r,c,mask):
        sl = (slice(r-nhood,r+nhood), slice(c-nhood,c+nhood)) # don't need center here
        if not np.any(mask[sl]):
            return
        tsl = (slice(None),)+sl

        patch_frames = data[tsl]
        w_sh = patch_frames.shape
        psh = w_sh[1:]

        patch = patch_frames.reshape(L,-1)
        patch_c = np.median(patch,0)
        patch = patch - patch_c

        if not(np.any(patch)):
            out_signals[tsl] += 0
            out_baselines[tsl] += 0
            counts[sl] += 0
            return
        # (patch is now Nframes x Npixels, u will hold temporal components)
        u,s,vh = np.linalg.svd(patch,full_matrices=False)
        if ncomp is None:
            rank = np.int(min(max(min_comps, min_ncomp(s, patch.shape)+1), max_comps))
            rank = (min(np.min(patch.shape)-1, rank))
            #print('\n\n\n rank ', rank, patch.shape, u.shape, s.shape, vh.shape)
            #sys.stderr.write(' svd rank: %02d'% rank)
        else:
            rank = ncomp

        #score = np.sum(s[:rank]**2)/np.sum(s**2)
        score = 1

        ux = u[:,:rank]
        vh = vh[:rank]
        s = s[:rank]

        W = np.diag(s)@vh

        if spatial_filter >= 1:
            W_images = W.reshape(-1,*psh)
            Wx_b = np.array([smoothed_medianf(f,spatial_filter/5, spatial_filter) for f in W_images])
            Wx_b = Wx_b.reshape(rank,len(vh[0]))
        else:
            Wx_b = W

        if only_truncated_svd:
            sys.stderr.write('__ doing only truncated svd                 ')
            baselines = ux@Wx_b#@vhx[:rank]
            rec_baselines = baselines.reshape(w_sh) + patch_c.reshape(psh)
            rec = np.zeros(w_sh)
        else:
            svd_signals = ux.T
            if baseline_smoothness:
                biases = np.array([simple_baseline(v,plow=50,smooth=baseline_smoothness,ns=mad_std(v)) for v in svd_signals])
                #biases = np.array([smoothed_medianf(v, smooth=5, wmedian=int(baseline_smoothness)) for v in svd_signals])
            else:
                biases = np.array([find_bias(v,ns=mad_std(v)) for v in svd_signals]).reshape(-1,1)
                biases = np.zeros_like(svd_signals)+biases

            svd_signals_c = svd_signals - biases

            # NOTE: this kind of labeler may omit some events
            #       if there are both up- and down- deflections in the signal
            labeler = partial(percentile_label, percentile_low=svd_detection_plow, tau=2)
            if temporal_filter > 1:
                tsmoother = partial(smoothed_medianf,  smooth=1., wmedian=3)
            else:
                tsmoother = lambda v_: v_

            signals_fplus = np.array([tsmoother(v)*labeler(v) for v in svd_signals_c])
            signals_fminus = np.array([tsmoother(v)*labeler(-v) for v in svd_signals_c])

            signals_filtered = signals_fplus + signals_fminus
            event_tmask = np.sum(np.abs(signals_filtered)>0,0) > 0
            active_comps = np.sum(np.abs(signals_filtered)>0,1)>3 # %active component for clustering is such that was active at least for k frames
            #ux_signals = signals_filtered.T
            #ux_biases = biases.T
            nactive = np.sum(active_comps)
            #sys.stderr.write(' active components: %02d                   '%nactive)

            baselines = biases.T@Wx_b#@vhx[:rank]
            rec_baselines = baselines.reshape(w_sh) + patch_c.reshape(psh)


            if not np.any(active_comps):
                rec = np.zeros(w_sh)
            else:
                if not with_clusters:
                    if correct_spatial_components:
                        Xdiff = svd_signals_c.T@Wx_b
                        Xdiff_permuted = np.array([np.random.permutation(v) for v in Xdiff.T]).T
                        signals_permuted = np.array([np.random.permutation(v) for v in signals_filtered])

                        Wnew = signals_filtered@(Xdiff)
                        Wnew_perm = signals_permuted@(Xdiff)
                        Wnew_frames = Wnew.reshape(W_images.shape)
                        Wnew_perm_frames = Wnew_perm.reshape(W_images.shape)
                        # below some parameters need formalization and tuning for optimal results on in vivo vs in situ data
                        Wmasks = np.array([threshold_object_size(np.abs(f1) > 3*np.percentile(np.abs(f2),99),5)
                                           for f1,f2 in zip(Wnew_frames,Wnew_perm_frames)])
                        Wx_b = Wx_b*Wmasks.reshape(Wx_b.shape)

                    rec = (signals_filtered.T@Wx_b).reshape(w_sh)

                else:
                    #affs = cluster.som(Wx_b.T,(rank*2,1),min_reassign=1)
                    Wactive = Wx_b[active_comps]
                    nclusters = nactive*4 # todo: make a parameter to choose.
                    cx = skclust.AgglomerativeClustering(nclusters,affinity='l1',linkage='average')
                    affs = cx.fit_predict(Wactive.T)
                    affs = correct_small_loads(Wactive.T,affs,min_loads=5,niter=5)
                    affs = cleanup_cluster_map(affs.reshape(psh),niter=10).ravel()
                    affs = correct_small_loads(Wactive.T,affs,min_loads=5,niter=2)

                    approx_c = svd_signals_c.T@Wx_b
                    #approx_c = svd_signals_c[active_comps].T@Wactive
                    cluster_signals = np.array([approx_c.T[affs==k].mean(0) for k in np.unique(affs)])
                    #cbiases = np.array([find_bias(v) for v in cluster_signals])
                    labeler = partial(percentile_label, percentile_low=cluster_detection_plow, tau=2)
                    csignals_filtered = np.array([simple_pipeline_(v, noise_sigma=mad_std(v),labeler=labeler )
                                               for v in cluster_signals])
                    som_spatial_comps = np.array([affs==k for k in np.unique(affs)])
                    rec = (csignals_filtered.T@som_spatial_comps).reshape(-1,*psh)

        out_baselines[tsl] += score*rec_baselines
        counts_b[sl] += score

        # we possibly shift the baseline level due to thresholding of components
        ##rec += find_bias_frames(data[tsl]-rec,3,mad_std(data[tsl],0))
        out_signals[tsl] += score*rec
        counts[sl] += score#*(np.sum(rec,0)>0)
        return

    for r in itt.chain(range(nhood,sh[1]-nhood,stride), [sh[1]-nhood]):
        for c in itt.chain(range(nhood,sh[2]-nhood,stride), [sh[2]-nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            #if mask[r,c]:
            _process_loc(r,c,mask)
    out_signals = out_signals/(1e-6+counts[None,:,:])
    out_baselines = out_baselines/(1e-6+counts_b[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out_signals[:,r,c] = 0
            if counts_b[r,c] == 0:
                out_baselines[:,r,c] = 0
    deltas = data-out_signals-out_baselines
    #correction_bias = find_bias_frames(deltas, 3, mad_std(deltas,0))
    return out_signals, out_baselines#+correction_bias


def block_svd_separate_tslices(frames, twindow=200,
                               nhood=5,
                               ncomp=None,
                               mask_of_interest=None,
                               th = 0.05,
                               verbose=True,
                               baseline_post_smooth=10,
                               **denoiser_kw):

    L = len(frames)
    sh =  frames[0].shape

    if twindow < L:
        tslices = [slice(i, i+twindow) for i in range(0,L-twindow,twindow//2)] + [slice(L-twindow, L)]
    else:
        tslices = [slice(0, L)]
    counts = np.zeros(L)



    if mask_of_interest is None:
        mask_list = (np.ones(sh) for t in tslices)
    elif np.ndim(mask_of_interest) == 2:
        mask_list = (mask_of_interest for  t in tslices)
    else:
        mask_list = (mask_of_interest[t].mean(0)>th for t in tslices)

    out_s = np.zeros(frames.shape)
    out_b = np.zeros(frames.shape)
    for k,ts,m in zip(range(L), tslices, mask_list):
        s,b = block_svd_denoise_and_separate(frames[ts], mask_of_interest=m, nhood=nhood, ncomp=ncomp, **denoiser_kw)
        out_s[ts] += s
        out_b[ts] += b
        counts[ts] += 1
        if verbose:
            sys.stdout.write('\n processed time-slice %d out of %d\n'%(k+1, len(tslices)))

    out_s =  out_s/counts[:,None,None]
    out_b = out_b/counts[:,None,None]
    if baseline_post_smooth > 0:
        out_b = ndi.gaussian_filter(out_b, (baseline_post_smooth, 0, 0))
    return out_s, out_b
