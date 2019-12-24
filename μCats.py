"""
μCats -- a set of routines for detection and analysis of Ca-transients
"""

import os,sys
from numba import jit
import pickle

from functools import partial
import itertools as itt

import matplotlib.pyplot as plt


import numpy as np


from numpy import pi
from numpy import linalg

from numpy import array,zeros,zeros_like,median,diag,ravel,unique
from numpy import arange
from numpy.linalg import norm, lstsq, svd, eig
from numpy.random import randn

from scipy.fftpack import dct,idct
from scipy import sparse
from scipy import ndimage as ndi



# Requires image-funcut
# find it on github: https://github.com/abrazhe/image-funcut/tree/develop

from imfun.filt.dctsplines import l1spline, l2spline, sp_decompose
from imfun.filt.dctsplines import rolling_sd_scipy_nd
from imfun import bwmorph

from imfun import cluster
from imfun.filt.dctsplines import l2spline
from imfun.core.coords import make_grid
from imfun.core import fnutils
from imfun.multiscale import mvm

from imfun import components


_dtype_ = np.float32



def store_baseline_pickle(name, frames, ncomp=50):
    pcf = components.pca.PCA_frames(frames,npc=50)
    pickle.dump(pcf, open(name, 'wb'))

def load_baseline_pickle(name):
    pcf = pickle.load(open(name, 'rb'))
    return pcf.inverse_transform(pcf.coords)


def make_weighting_kern(size,sigma=1.5):
    """
    Make a 2d array of floats to weight signal inputs in the spatial windows/patches
    """
    #size = patch_size_
    x,y = np.mgrid[-size/2.+0.5:size/2.+.5,-size/2.+.5:size/2.+.5]
    g = np.exp(-(0.5*(x/sigma)**2 + 0.5*(y/sigma)**2))
    return g

@jit
def avg_filter_greater(m, th=0):
    nr,nc = m.shape
    out = np.zeros_like(m)
    for r in range(nr):
        for c in range(nc):
            if m[r,c] <= th:
                continue
            count,acc = 0,0
            for i in range(r-1,r+2):
                for j in range(c-1,c+2):
                    if (0 <= i < nr) and (0 <= j < nc):
                        if m[i,j] > th:
                            count +=1
                            acc += m[i,j]
            if count > 0:
                out[r,c] = acc/count
    return out

def signals_from_array_avg(data, stride=2, patch_size=5):
    """Convert a TXY image stack to a list of temporal signals (taken from small spatial windows/patches)"""
    d = np.array(data).astype(_dtype_)
    acc = []
    squares =  list(map(tuple, make_grid(d.shape[1:], patch_size,stride)))
    w = make_weighting_kern(patch_size,2.5)
    w = w/w.sum()
    #print('w.shape:', w.shape)
    #print(np.argmax(w.reshape(1,-1)))

    tslice = (slice(None),)
    for sq in squares:
        patch = d[tslice+sq]
        sh = patch.shape
        wclip = w[:sh[1],:sh[2]]
        #print(np.argmax(wclip))
        #print(w.shape, sh[1:3], wclip.shape)
        #wclip /= sum(wclip)
        signal = (patch*wclip).sum(axis=(1,2))
        acc.append((signal, sq, wclip.reshape(1,-1)))
    return acc
    #signals =  array([d[(slice(None),)+s].sum(-1).sum(-1)/prod(d[0][s].shape) for s in squares])
    #return [(v,sq,w) for v,sq in zip(signals, squares)]

def downsample_image(img):
    sigma_0 = 0.6
    sigma = sigma_0*(1/0.25 - 1 )**0.5
    im = ndi.gaussian_filter(img, sigma, mode='nearest')
    return ndi.zoom(im, 0.5)

def upsample_image(img):
    return ndi.zoom(img,2,mode='nearest')

def clip_outliers(m, plow=0.5, phigh=99.5):
    px = np.percentile(m, (plow, phigh))
    return np.clip(m, *px)

def weight_counts(collection,sh):
    counts = np.zeros(sh)
    for v,s,w in collection:
        wx = w.reshape(counts[tuple(s)].shape)
        counts[s] += wx
    return counts


def signals_from_array_pca_cluster(data,stride=2, nhood=3, ncomp=2,
                                   pre_smooth=1,
                                   dbscan_eps_p=10, dbscan_minpts=3, cluster_minsize=5,
                                   walpha=1.0,
                                   mask_of_interest=None):
    """
    Convert a TXY image stack to a list of signals taken from spatial windows and aggregated according to their coherence
    """
    sh = data.shape
    if mask_of_interest is None:
        mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    mask = mask_of_interest
    counts = np.zeros(sh[1:])
    acc = []
    knn_count = [0]
    cluster_count = [0]
    Ln = (2*nhood+1)**2
    corrfn=stats.pearsonr
    patch_size = (nhood*2+1)**2
    if cluster_minsize > patch_size:
        cluster_minsize = patch_size
    #dbscan_eps_acc = []
    def _process_loc(r,c):
        kcenter = 2*nhood*(nhood+1)
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        patch = data[(slice(None),)+sl]
        if not np.any(patch):
            return
        patch = patch.reshape(sh[0],-1).T
        patch0 = patch.copy()
        if pre_smooth > 1:
            patch = ndi.median_filter(patch, size=(pre_smooth,1))
        Xc = patch.mean(0)
        u,s,vh = np.linalg.svd(patch-Xc,full_matrices=False)
        points = u[:,:ncomp]
        #dists = cluster.metrics.euclidean(points[kcenter],points)
        all_dists = cluster.dbscan_._pairwise_euclidean_distances(points)
        dists = all_dists[kcenter]

        max_same = np.max(np.diag(all_dists))

        #np.mean(dists)
        dbscan_eps = np.percentile(all_dists[all_dists>max_same], dbscan_eps_p)
        #dbscan_eps_acc.append(dbscan_eps)
        #print(r,c,':', dbscan_eps)
        _,_,affs = cluster.dbscan(points, dbscan_eps, dbscan_minpts, distances=all_dists)
        similar = affs==affs[kcenter]

        if sum(similar) < cluster_minsize or affs[kcenter]==-1:
            knn_count[0] += 1
            #th = min(np.argsort(dists)[cluster_minsize+1],2*dbscan_eps)
            th = dists[np.argsort(dists)[min(len(dists), cluster_minsize*2)]]
            similar = dists <= max(th, max_same)
            #print('knn similar:', np.sum(similar), 'total signals:', len(similar))
            #dists *= 2  # shrink weights if not from cluster
        else:
            cluster_count[0] +=1

        weights = np.exp(-walpha*dists)
        #weights = np.array([corrfn(a,v)[0] for a in patch])**2

        #weights /= np.sum(weights)
        #weights = ones(len(dists))
        weights[~similar] = 0
        #weights = np.array([corrfn(a,v)[0] for a in patch])

        #weights /= np.sum(weights)
        vx = patch0[similar].mean(0) # DONE?: weighted aggregate
                                    # TODO: check how weights are defined in NL-Bayes and BM3D
                                    # TODO: project to PCs?
        acc.append((vx, sl, weights))
        return #  _process_loc

    for r in range(nhood,sh[1]-nhood,stride):
        for c in range(nhood,sh[2]-nhood,stride):
            sys.stderr.write('\r processing location %05d/%d '%(r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)

    sys.stderr.write('\n')
    print('KNN:', knn_count[0])
    print('cluster:',cluster_count[0])
    m = weight_counts(acc, sh[1:])
    #print('counted %d holes'%np.sum(m==0))
    nholes = np.sum((m==0)*mask)
    #print('N holes:', nholes)
    #print('acc len before:', len(acc))
    hole_i = 0
    for r in range(nhood,sh[1]-nhood):
        for c in range(nhood,sh[2]-nhood):
            if mask[r,c] and (m[r,c] < 1e-6):
                sys.stderr.write('\r processing additional location %05d/%05d '%(hole_i, nholes))
                _process_loc(r,c)
                #v = data[:,r,c]
                #sl = (slice(r-1,r+1+1), slice(c-1,c+1+1))
                #weights = np.zeros((3,3))
                #weights[1,1] = 1.0
                #acc.append((v, sl, weights.ravel()))
                hole_i += 1
    #print('acc len after:', len(acc))
    #print('DBSCAN eps:', np.mean(dbscan_eps_acc), np.std(dbscan_eps_acc))
    return acc


from scipy import stats
def signals_from_array_correlation(data,stride=2,nhood=5,
                                   max_take=10,
                                   corrfn = stats.pearsonr,
                                   mask_of_interest=None):
    """
    Convert a TXY image stack to a list of signals taken from spatial windows and aggregated according to their coherence
    """
    sh = data.shape
    L = sh[0]
    if mask_of_interest is None:
        mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    mask = mask_of_interest
    counts = np.zeros(sh[1:])
    acc = []
    knn_count = 0
    cluster_count = 0
    Ln = (2*nhood+1)**2
    max_take = min(max_take, Ln)
    def _process_loc(r,c):
        v = data[:,r,c]
        kcenter = 2*nhood*(nhood+1)
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        patch = data[(slice(None),)+sl]
        if not np.any(patch):
            return
        patch = patch.reshape(sh[0],-1).T
        weights = np.array([corrfn(a,v)[0] for a in patch])
        weights[weights < 2/L**0.5] = 0 # set weights to 0 in statistically independent sources
        weights[np.argsort(weights)[:-max_take]]=0
        weights = weights/np.sum(weights) # normalize weights
        weights += 1e-6 # add small weight to avoid dividing by zero
        vx = (patch*weights.reshape(-1,1)).sum(0)
        acc.append((vx, sl, weights))


    for r in range(nhood,sh[1]-nhood,stride):
        for c in range(nhood,sh[2]-nhood,stride):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)
    for _,sl,w in acc:
        counts[sl] += w.reshape(2*nhood+1,2*nhood+1)
    for r in range(nhood,sh[1]-nhood):
        for c in range(nhood,sh[2]-nhood):
            if mask[r,c] and not counts[r,c]:
                sys.stderr.write('\r (2x) processing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
                _process_loc(r,c)
    return acc

from imfun import components
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


def patch_pca_denoise2(data,stride=2, nhood=5, npc=None,
                       temporal_filter=1,
                       spatial_filter=1,
                       mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2
    def _process_loc(r,c,rank):
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        tsl = (slice(None),)+sl
        patch = data[tsl]
        w_sh = patch.shape
        patch = patch.reshape(sh[0],-1)
        if not(np.any(patch)):
            out[tsl] += 0
            counts[sl] += 0
            return
        # (patch is now Nframes x Npixels, u will hold temporal components)
        u,s,vh = np.linalg.svd(patch,full_matrices=False)
        if rank is None:
            rank = min_ncomp(s, patch.shape)+1
            sys.stderr.write(' | svd rank: %02d  '% rank)
        ux = ndi.median_filter(u[:,:rank],size=(temporal_filter,1))
        vh_images = vh[:rank].reshape(-1,*w_sh[1:])
        vhx = [ndi.median_filter(f, size=(spatial_filter,spatial_filter)) for f in vh_images]
        vhx_threshs = [mad_std(f) for f in vh_images]
        vhx = np.array([np.where(np.abs(f-fx) > th,fx,f) for f,fx,th in zip(vh_images,vhx,vhx_threshs)])
        vhx = vhx.reshape(rank,len(vh[0]))


        #print('\n', patch.shape, u.shape, vh.shape)
        #ux = u[:,:rank]
        proj = ux@np.diag(s[:rank])@vhx[:rank]
        score = np.sum(s[:rank]**2)/np.sum(s**2)
        #score = 1
        rec  = proj.reshape(w_sh)
        #if keep_baseline:
        #    # we possibly shift the baseline level due to thresholding of components
        #    rec += find_bias_frames(data[tsl]-rec,3,mad_std(data[tsl],0))
        out[tsl] += score*rec
        counts[sl] += score

    for r in itt.chain(range(nhood,sh[1]-nhood,stride), [sh[1]-nhood]):
        for c in itt.chain(range(nhood,sh[2]-nhood,stride), [sh[2]-nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d '%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c,npc)
    out = out/(1e-12+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out[:,r,c] = 0
    return out


def nmf_labeler(y,th=1):
    sigma = std_median(y)
    ys = smoothed_medianf(y,0.5,3)
    structures, nlab = ndi.label(y>np.median(y))
    peaks = ys>=th*sigma
    return y*select_overlapping(structures,peaks)

def convert_from_varstab(df,b):
    "convert fluorescence signals separated to fchange and f baseline from 2*√f space"
    bc = b**2/4
    dfc =  (df**2 + 2*df*b)/4
    return dfc, bc


def top_average_frames(frames,percentile=85):
    sh = frames.shape
    pmap = np.percentile(frames, percentile, axis=0)
    out = np.zeros(sh[1:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            p = pmap[r,c]
            v = frames[:,r,c]
            out[r,c] = np.mean(v[v>=p])
    return out


@jit
def cleanup_cluster_map(m,niter=1):
    Nr,Nc = m.shape
    cval = np.min(m)-1
    m = np.pad(m,1, mode='constant',constant_values=cval)
    for j in range(niter):
        for r in range(1,Nr):
            for c in range(1,Nc):
                me = m[r,c]
                neighbors = array([m[(r+1),c], m[(r-1),c],  m[r,(c+1)],  m[r,(c-1)]])
                if not np.any(neighbors==me):
                    neighbors = neighbors[neighbors>cval]
                    if len(neighbors):
                        m[r,c] = neighbors[np.random.randint(len(neighbors))]
                    else:
                        m[r,c] = cval
    return m[1:-1,1:-1]


def correct_small_loads(points, affs, min_loads=5, niter=1):
    for j in range(niter):
        new_affs = np.copy(affs)
        labels = unique(affs)
        loads = array([np.sum(affs==k) for k in labels])
        if not np.any(loads < min_loads):
            break
        centers = array([np.mean(points[affs==k],0) for k in labels])
        point_ind = arange(len(points))
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
            sys.stderr.write(' svd rank: %02d'% rank)
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
            sys.stderr.write(' active components: %02d                   '%nactive)

            baselines = biases.T@Wx_b#@vhx[:rank]
            rec_baselines = baselines.reshape(w_sh) + patch_c.reshape(psh)


            if not np.any(active_comps):
                rec = np.zeros(w_sh)
            else:
                if not with_clusters:
                    if correct_spatial_components:
                        Xdiff = svd_signals_c.T@Wx_b
                        Xdiff_permuted = array([np.random.permutation(v) for v in Xdiff.T]).T
                        signals_permuted = array([np.random.permutation(v) for v in signals_filtered])

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
                    cluster_signals = array([approx_c.T[affs==k].mean(0) for k in np.unique(affs)])
                    #cbiases = array([find_bias(v) for v in cluster_signals])
                    labeler = partial(percentile_label, percentile_low=cluster_detection_plow, tau=2)
                    csignals_filtered = array([simple_pipeline_(v, noise_sigma=mad_std(v),labeler=labeler )
                                               for v in cluster_signals])
                    som_spatial_comps = array([affs==k for k in np.unique(affs)])
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



def locations(shape):
    """ all locations for a shape; substitutes nested cycles"""
    return itt.product(*map(range, shape))


def points2mask(points,sh):
    out = np.zeros(sh, np.bool)
    for p in points:
        out[tuple(p)] = True
    return out

def mask2points(mask):
    "mask to a list of points, as row,col"
    return  np.array([loc for loc in locations(mask.shape) if mask[loc]])

from imfun import cluster
def cleanup_mask(m, eps=3, min_pts=5):
    if not np.any(m):
        return np.zeros_like(m)
    p = mask2points(m)
    _,_,labels = cluster.dbscan(p, eps, min_pts)
    points_f = (p for p,l in zip(p, labels) if l >= 0)
    return points2mask(points_f, m.shape)


from skimage.feature import register_translation
from skimage import transform as skt
from imfun import core

def shift_signal(v, shift):
    t = skt.SimilarityTransform(translation=(shift,0))
    return skt.warp(v.reshape(1,-1),t,mode='wrap').ravel()

def _register_shift_1d(target,source):
    'find translation in 1d signals (assumes input is in Fourier domain)'
    z = np.fft.ifft(target*source.conj()).real
    L = len(target)
    k1 = np.argmax(z)
    return -k1 if k1 < L/2 else (L-k1)

def _patch_pca_denoise_with_shifts(data,stride=2, nhood=5, npc=None,
                                   temporal_filter=1,
                                   max_shift = 20,
                                   mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2

    #preproc = lambda y: core.rescale(y)

    #tmp_signals = np.zeros()
    tv = np.arange(L)

    def _shift_signal_i(v, shift):
        return v[((tv+shift)%L).astype(np.int)]

    def _process_loc(r,c):
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        tsl = (slice(None),)+sl


        patch = data[tsl]
        w_sh = patch.shape
        psh = w_sh[1:]
        signals = patch.reshape(L,-1).T

        signals_ft = np.fft.fft(signals, axis=1)
        kcenter = 2*nhood*(nhood+1)
        # todo : use MAD estimate of std or other
        #kcenter = np.argmax(np.std(signals,axis=1))


        vcenter = signals[kcenter]
        vcenter_ft = signals_ft[kcenter]
        #shifts = [register_translation(v,vcenter)[0][0] for v in signals]
        shifts = np.array([_register_shift_1d(vcenter_ft,v) for v in signals_ft])
        shifts = shifts*(np.abs(shifts) < max_shift)

        vecs_shifted = np.array([_shift_signal_i(v, p)  for v,p in zip(signals, shifts)])
        #vecs_shifted = np.array([v[((tv+p)%L).astype(int)] for v,p in zip(signals, shifts)])
        corrs_shifted = np.corrcoef(vecs_shifted)[kcenter]
        coherent_mask = corrs_shifted > 0.33
        #print(r,c,': sum coherent: ', np.sum(coherent_mask),'/',len(coherent_mask),'mean coh:',np.mean(corrs_shifted), '\n',)

        u0,s0,vh0 = np.linalg.svd(vecs_shifted,full_matrices=False)
        rank = min_ncomp(s0, vecs_shifted.shape)+1 if npc is None else npc
        if temporal_filter > 1:
            vhx0 = ndi.gaussian_filter(ndi.median_filter(vh0[:rank],size=(1,temporal_filter)),sigma=(0,0.5))
        else:
            vhx0 = vh0[:rank]
        ux0 = u0[:,:rank]
        recs = ux0@np.diag(s0[:rank])@vhx0
        #score = np.sum(s0[:rank]**2)/np.sum(s0**2)*np.ones(len(signals))
        score = 1

        if np.sum(coherent_mask) > 2*rank:
            u,s,vh = np.linalg.svd(vecs_shifted[coherent_mask],False)
            vhx = ndi.median_filter(vh[:rank],size=(1,temporal_filter)) if temporal_filter > 1 else vh[:rank]
            ux = u[:,:rank]
            recs_coh = (vecs_shifted@vh[:rank].T)@vh[:rank]
            score_coh = np.sum(s[:rank]**2)/np.sum(s**2)
            recs = np.where(coherent_mask[:,None], recs_coh, recs)
            score[coherent_mask] = score_coh

        recs_unshifted = np.array([_shift_signal_i(v,-p) for v,p in zip(recs,shifts)])
        proj = recs_unshifted.T

        score = score.reshape(psh)
        #score = 1
        out[tsl] += score*proj.reshape(w_sh)
        counts[sl] += score

    for r in range(nhood,sh[1]-nhood,stride):
        for c in range(nhood,sh[2]-nhood,stride):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)
    out = out/(1e-12+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out[:,r,c] = 0
    return out



def _patch_denoise_dmd(data,stride=2, nhood=5, npc=None,
                       temporal_filter = None,
                       mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2

    #preproc = lambda y: core.rescale(y)

    #tmp_signals = np.zeros()
    tv = np.arange(L)

    def _next_x_prediction(X,lam,Phi):
        Xn = X.reshape(-1,1)
        b = lstsq(Phi, Xn, rcond=None)[0]
        Xnext =  (Phi@np.diag(lam)@b.reshape(-1,1)).real
        return Xnext
    #    return Xnext.T.reshape(f.shape)

    def _process_loc(r,c):
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        tsl = (slice(None),)+sl

        patch = data[tsl]

        X = patch.reshape(L,-1).T
        #print(patch.shape, X.shape)

        lam,Phi = dmdf_new(X,r=npc)

        rec = np.array([_next_x_prediction(f,lam,Phi) for f in X.T])

        #print(out[tsl].shape, patch.shape, rec.shape)
        out[tsl] += rec.reshape(*patch.shape)

        score = 1.0
        counts[sl] += score

    for r in itt.chain(range(nhood,sh[1]-nhood,stride), [sh[1]-nhood]):
        for c in itt.chain(range(nhood,sh[2]-nhood,stride), [sh[2]-nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)
    out = out/(1e-12+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out[:,r,c] = 0
    return out

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




def dmdf_new(X,Y=None, r=None,sort_explained=False):
    if Y is None:
        Y = X[:,1:]
        X = X[:,:-1]
    U,sv,Vh = np.linalg.svd(X,False)
    if r is None:
        r = min_ncomp(sv, X.shape) + 1
    sv = sv[:r]
    V = Vh[:r].conj().T
    Uh = U[:,:r].conj().T
    B = Y@V@(np.diag(1/sv))

    Atilde = Uh@B
    lam, W = np.linalg.eig(Atilde)
    Phi = B@W
    #print(Vh.shape)
    # approx to b
    def _bPOD(i):
        alpha1 =np.diag(sv[:r])@Vh[:r,i]
        return np.linalg.lstsq(Atilde@W,alpha1,rcond=None)[0]
    #bPOD = _bPOD(0)
    stats = (None,None)
    if sort_explained:
        #proj_dmd = Phi.T.dot(X)
        proj_dmd = np.array([_bPOD(i) for i in range(Vh.shape[1])])
        dmd_std = proj_dmd.std(0)
        dmd_mean = abs(proj_dmd).mean(0)
        stats = (dmd_mean,dmd_std)
        kind = np.argsort(dmd_std)[::-1]
    else:
        kind = np.arange(r)[::-1] # from slow to fast
    Phi = Phi[:,kind]
    lam = lam[kind]
    #bPOD=bPOD[kind]
    return lam, Phi#,bPOD,stats


def threshold_object_size(mask, min_size):
    labels, nlab = ndi.label(mask)
    objs = ndi.find_objects(labels)
    out_mask = np.zeros_like(mask)
    for k,o in enumerate(objs):
        cond = labels[o]==(k+1)
        if np.sum(cond) >= min_size:
            out_mask[o][cond] = True
    return out_mask

@jit
def percentile_th_frames(frames,plow=5):
    sh = frames[0].shape
    medians = np.median(frames,0)
    out = np.zeros(medians.shape)
    for r in range(sh[0]):
        for c in range(sh[1]):
            v = frames[:,r,c]
            mu = medians[r,c]
            out[r,c] = -np.percentile(v[v<=mu],plow)
    return out


def select_overlapping(mask, seeds):
    labels, nl = ndi.label(mask)
    objs = ndi.find_objects(labels)
    out = np.zeros_like(mask)
    for k,o in enumerate(objs):
        cond = labels[o]==k+1
        if np.any(seeds[o][cond]):
            out[o][cond] = True
    return out

#def find_events_by_median_filtering(frames, nw=5, plow=5, smooth=2.5):
#    mf_frames = ndi.median_filter(frames, (1,nw,nw))
#    ns = mad_std(mf_frames, axis=0)
#    biases = find_bias_frames(mf_frames, 3, ns)
#    mf_frames = (mf_frames-biases)/ns
#    th = percentile_th_frames(mf_frames)
#    return l2spline(mf_frames, smooth) > th

def opening_of_closing(m):
    return ndi.binary_opening(ndi.binary_closing(m))


def to_zscore_frames(frames):
    nsm = mad_std(frames, axis=0)
    biases = find_bias_frames(frames, 3, nsm)

    return np.where(nsm>1e-5,(frames-biases)/(nsm+1e-5),0)


def activity_mask_median_filtering(frames, nw=11, th=1.0, plow=2.5, smooth=2.5,
                                   verbose=True):

    mf_frames50 = ndi.percentile_filter(frames,50, (1,nw,nw))    # spatial median filter
    #mf_frames85 = ndi.percentile_filter(frames,85, (1,nw,nw))    # spatial top 85% filter
    mf_frames = mf_frames50#*mf_frames85
    del mf_frames50#,mf_frames85

    if verbose:
        print('Done percentile filters')

    mf_frames = to_zscore_frames(mf_frames)
    mf_frames = np.clip(mf_frames, *np.percentile(mf_frames, (0.5,99.5)))
    #return mf_frames

    th = percentile_th_frames(mf_frames,plow)
    mask = (mf_frames > th)*(ndi.gaussian_filter(mf_frames, (smooth,0.5,0.5))>th)
    mask = ndi.binary_dilation(opening_of_closing(mask))
    #mask = np.array([threshold_object_size(m,)])
    #mask = threshold_object_size(mask, 4**3)
    if verbose:
        print('Done mask from spatial filters')
    return mask

    #ns = mad_std(frames, axis=0)
    #frames_smooth = ndi.gaussian_filter(ndi.median_filter(frames, (5, 1,1)), (1.,0,0))
    #mask_a = frames_smooth > th*ns
    #
    #if verbose:
    #    print('Done mask from temporal filters')
    #
    #mask_seeds = (mask_a + ndi.median_filter(mask_a, (1,3,3))>0)*mask
    #mask_seeds = np.array([threshold_object_size(m, 9) for m in mask_seeds])
    #
    #mask_final = mask_seeds
    #if verbose:
    #    print('Merged masks')
    ##mask_final = select_overlapping(mask_a,mask_seeds)
    #return np.array([avg_filter_greater(f,0) for f in frames_smooth*mask_final]), mask, mask_final


def _patch_denoise_percentiles(data,stride=2, nhood=3, mw=5,
                               px = 50,
                               th = 1.5,
                               mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2

    #preproc = lambda y: core.rescale(y)

    #tmp_signals = np.zeros()
    tv = np.arange(L)

    def _process_loc(r,c):
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        tsl = (slice(None),)+sl


        patch = data[tsl]
        w_sh = patch.shape
        signals = patch.reshape(sh[0],-1).T
        #print(signals.shape)

        #vm = np.median(signals,0)
        vm = np.percentile(signals, px, axis=0)
        vm = (vm-find_bias(vm))/mad_std(vm)
        vma = simple_pipeline_(vm, smoothed_rec=True)
        # todo extend masks a bit in time?
        vma_mask = threshold_object_size(vma>0.1,5).astype(np.bool)

        nsv = np.array([mad_std(v) for v in signals]).reshape(-1,1)
        pf = np.array([smoothed_medianf(v,0.5,mw) for v in signals])
        pa = (pf > th*nsv)
        pa_txy = pa.T.reshape(w_sh)
        pa_txy2 = (ndi.median_filter(pa_txy.astype(np.float32),(3,3,3))>0)*vma_mask[:,None,None]

        labels,nl = ndi.label(pa_txy+pa_txy2)
        objs = ndi.find_objects(labels)
        pa_txy3 = np.zeros_like(pa_txy)
        for k,o in enumerate(objs):
            cond = labels[o] == k+1
            if np.any(pa_txy2[o][cond]):
                pa_txy3[o][cond] = True

        pf_txy = pf.T.reshape(w_sh)*pa_txy3
        #pf_txy = (pf*vma_mask).
        rec = np.array([avg_filter_greater(m,0) for m in pf_txy])
        #rec = pf_txy


        #rec = pf*vma*(pf>th*nsv)
        #score = score.reshape(w_sh[1:])
        score = 1.0
        out[tsl] += score*rec
        counts[sl] += score

    for r in range(nhood,sh[1]-nhood,stride):
        for c in range(nhood,sh[2]-nhood,stride):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)
    out = out/(1e-12+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out[:,r,c] = 0
    return out


from fastdtw import fastdtw
from imfun import core

def apply_warp_path(v, path):
    path = np.array(path)
    return np.interp(np.arange(len(v)), path[:,0], v[path[:,1]])

def interpolate_path(path,L):
    return np.interp(np.arange(L), path[:,0],path[:,1])

def omega_approx(beta):
    return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43

def svht(sv, sh):
    m,n = sh
    if m>n:
        m,n=n,m
    omg = omega_approx(m/n)
    return omg*np.median(sv)

def min_ncomp(sv,sh):
    th = svht(sv,sh)
    return sum(sv >=th)

def _patch_pca_denoise_with_dtw(data,stride=2, nhood=5, npc=6,
                                    temporal_filter=1,
                                    spatial_filter=1,
                                    mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2
    def _process_loc(r,c):
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        tsl = (slice(None),)+sl

        kcenter = 2*nhood*(nhood+1)

        patch = data[tsl]
        w_sh = patch.shape
        patch = patch.reshape(sh[0],-1)
        # (patch is now Nframes x Npixels, u will hold temporal components)

        signals = patch.T

        vcentral = signals[kcenter]
        dtw_warps = [np.array(fastdtw(vcentral, v)[1]) for v in signals]

        #dtw_warps_smoothed = [ for p in dtw_path]
        paths_interp = np.array([interpolate_path(p,L) for p in dtw_warps])
        paths_interp_dual = np.array([interpolate_path(np.fliplr(p),L) for p in dtw_warps])

        paths_interp_smooth = [np.clip(l2spline(ip,5).astype(int),0,L-1) for ip in paths_interp]
        paths_interp_dual_smooth = [np.clip(l2spline(ip,5).astype(int),0,L-1) for ip in paths_interp_dual]

        aligned = np.array([v[ip] for v,ip in zip(signals, paths_interp_smooth)])

        u,s,vh = np.linalg.svd(aligned.T,False)
        #u,s,vh = np.linalg.svd(patch,full_matrices=False)
        if temporal_filter>1:
            ux = ndimage.median_filter(u[:,:npc],size=(temporal_filter,1))
        else:
            ux = u[:,:npc]

        #points = vh[:npc].T
        #all_dists = cluster.dbscan_._pairwise_euclidean_distances(points)
        #dists = all_dists[kcenter]

        vh_images = vh[:npc].reshape(-1,*w_sh[1:])
        vhx = [ndimage.median_filter(f, size=(spatial_filter,spatial_filter)) for f in vh_images]
        vhx_threshs = [mad_std(f) for f in vh_images]
        vhx = np.array([np.where(f>th,fx,f) for f,fx,th in zip(vh_images,vhx,vhx_threshs)])
        vhx = vhx.reshape(npc,len(vh[0]))
        #print('\n', patch.shape, u.shape, vh.shape)
        #ux = u[:,:npc]
        proj_w = ux@np.diag(s[:npc])@vhx[:npc]
        score = np.sum(s[:npc]**2)/np.sum(s**2)

        proj = np.array([v[ip] for v,ip in zip(proj_w.T,paths_interp_dual_smooth)]).T

        #score = 1
        out[tsl] += score*proj.reshape(w_sh)
        counts[sl] += score

    for r in range(nhood,sh[1]-nhood,stride):
        for c in range(nhood,sh[2]-nhood,stride):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)
    out = out/(1e-12+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out[:,r,c] = 0
    return out



def nonlocal_video_smooth(data, stride=2,nhood=5,corrfn = stats.pearsonr,mask_of_interest=None):
    sh = data.shape
    if mask_of_interest is None:
        mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,dtype=_dtype_)
    mask = mask_of_interest
    counts = np.zeros(sh[1:])
    acc = []
    knn_count = 0
    cluster_count = 0
    Ln = (2*nhood+1)**2
    for r in range(nhood,sh[1]-nhood,stride):
        for c in range(nhood,sh[2]-nhood,stride):
            sys.stderr.write('\rprocessing location %05d/%d'%(r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                v = data[:,r,c]
                kcenter = 2*nhood*(nhood+1)
                sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
                patch = data[(slice(None),)+sl]
                w_sh = patch.shape
                patch = patch.reshape(sh[0],-1).T
                weights = np.array([corrfn(a,v)[0] for a in patch])**2
                weights = weights/np.sum(weights)
                wx = weights.reshape(w_sh[1:])
                ks = np.argsort(weights)[::-1]
                xs = ndi.median_filter(patch, size=(5,1))
                out[(slice(None),)+sl] += xs[np.argsort(ks)].T.reshape(w_sh)*wx[None,:,:]
                counts[sl] += wx
    out /= counts

    return out


def loc_in_patch(loc,patch):
    sl = patch[1]
    return np.all([s.start <= l < s.stop for l,s in zip(loc, sl)])

# def _baseline_windowed_pca(data,stride=4, nhood=7, ncomp=10,
#                           smooth = 60,
#                           walpha=1.0,
#                           mask_of_interest=None):
#     sh = data.shape
#     if mask_of_interest is None:
#         mask_of_interest = np.ones(sh[1:],dtype=np.bool)
#     mask = mask_of_interest
#     counts = np.zeros(sh[1:])
#     acc = []
#     knn_count = 0
#     cluster_count = 0
#     Ln = (2*nhood+1)**2
#     out_data = np.zeros(sh,dtype=_dtype_)
#     print(out_data.shape)
#     counts = np.zeros(sh[1:])
#     empty_slice = (slice(None),)

#     for r in range(nhood,sh[1]-nhood,stride):
#         for c in range(nhood,sh[2]-nhood,stride):
#             sys.stderr.write('\rprocessing pixel %05d/%d'%(r*sh[1] + c+1, np.prod(sh[1:])))
#             if mask[r,c]:
#                 kcenter = 2*nhood*(nhood+1)
#                 sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
#                 patch = data[(slice(None),)+sl]
#                 pshape = patch.shape
#                 patch = patch.reshape(sh[0],-1).T
#                 Xc = patch.mean(0)
#                 u,s,vh = np.linalg.svd(patch-Xc,full_matrices=False)
#                 points = u[:,:ncomp]
#                 #pc_signals = array([medismooth(s) for s in points.T])
#                 pc_signals = np.array([multi_scale_simple_baseline(s)  for s in points.T])
#                 signals = (pc_signals.T@np.diag(s[:ncomp])@vh[:ncomp] + Xc).T
#                 out_data[empty_slice+sl] += signals.reshape(pshape)
#                 counts[sl] += np.ones(pshape[1:],dtype=int)

#     out_data /= (1e-12 + counts)
#     return out_data


def combine_weighted_signals(collection,shape):
    """
    Combine a list of processed signals with weights back into TXY frame stack (nframes x nrows x ncolumns)
    """
    out_data = np.zeros(shape,dtype=_dtype_)
    counts = np.zeros(shape[1:])
    tslice = (slice(None),)
    i = 0
    for v,s,w in collection:
        pn = s[0].stop - s[0].start
        #print(s,len(w))
        wx = w.reshape(out_data[tslice+tuple(s)].shape[1:])
        out_data[tslice+tuple(s)] += v.reshape(-1,1,1)*wx
        counts[s] += wx
    out_data /= (1e-12 + counts)
    return out_data




min_px_size_ = 10

# def simple_pipeline(y,tau_label=1.5):
#     ns = rolling_sd_pd(y)
#     vn = y/ns
#     labels = simple_label_lj(vn, tau=tau_label,with_plots=False)
#     return sp_rec_with_labels(y, labels,with_plots=False,)

tau_label_=2.0


def process_tmvm(v, k=3,level=7, start_scale=1, tau_smooth=1.5,rec_variant=2,nonnegative=True):
    """
    Process temporal signal using MVM and return reconstructions of significant fluorescence elevations
    """
    objs = mvm.find_objects(v, k=k, level=level, min_px_size=min_px_size_,
                            min_nscales=3,
                            modulus=not nonnegative,
                            rec_variant=rec_variant,
                            start_scale=start_scale)
    if len(objs):
        if nonnegative:
            r = np.max(list(map(mvm.embedded_to_full, objs)),0).astype(v.dtype)
        else:
            r = np.sum([mvm.embedded_to_full(o) for o in objs],0).astype(v.dtype)
        if tau_smooth>0:
            r = l2spline(r, tau_smooth)
        if nonnegative:
            r[r<0]=0
    else:
        r = np.zeros_like(v)
    return r

import pandas as pd
def rolling_sd_pd(v,hw=None,with_plots=False,correct_factor=1.,smooth_output=True,input_is_details=False):
    """
    Etimate time-varying level of noise standard deviation
    """
    if not input_is_details:
        details = v-ndi.median_filter(v,20)
    else:
        details = v
    if hw is None: hw = int(len(details)/10.)
    padded = np.pad(details,2*hw,mode='reflect')
    tv = np.arange(len(details))

    s = pd.Series(padded)
    rkw = dict(window=2*hw,center=True)

    out = (s - s.rolling(**rkw).median()).abs().rolling(**rkw).median()
    out = 1.4826*np.array(out)[2*hw:-2*hw]

    if with_plots:
        f,ax = plt.subplots(1,1,sharex=True)
        ax.plot(tv,details,'gray')
        ax.plot(tv,out,'y')
        ax.plot(tv,2*out,'orange')
        ax.set_xlim(0,len(v))
        ax.set_title('Estimating running s.d.')
        ax.set_xlabel('samples')
    out = out/correct_factor
    if smooth_output:
        out = l2spline(out, s=2*hw)
    return out

def tmvm_baseline(y, plow=25, smooth_level=100, symmetric=False):
    """
    Estimate time-varying baseline in 1D signal by first finding fast significant
    changes and removing them, followed by smoothing
    """
    rec = process_tmvm(y,k=3,rec_variant=1)
    if symmetric:
        rec_minus = -process_signal(-y,k=3,rec_variant=1)
        rec=rec+rec_minus
    res = y-rec
    b = l2spline(ndi.percentile_filter(res,plow,smooth_level),smooth_level/2)
    rsd = rolling_sd_pd(res-b)
    return b,rsd,res

def tmvm_get_baselines(y,th=3,smooth=100,symmetric=False):
    """
    tMVM-based baseline estimation of time-varying baseline with bias correction
    """
    b,ns,res = tmvm_baseline(y,smooth_level=smooth,symmetric=symmetric)
    d = res-b
    return b + np.median(d[d<=th*ns]) # + bias as constant shift


def smoothed_medianf(v,smooth=10,wmedian=10):
    "Robust smoothing by first applying median filter and then applying L2-spline filter"
    return l2spline(ndi.median_filter(v, wmedian),smooth)

def simple_baseline(y, plow=25, th=3, smooth=25,ns=None):
    b = l2spline(ndi.percentile_filter(y,plow,smooth),smooth/5)
    if ns is None:
        ns = rolling_sd_pd(y)
    d = y-b
    if not np.any(ns):
        ns = np.std(y)
    bg_points = d[np.abs(d)<=th*ns]
    if len(bg_points) > 10:
        b = b + np.median(bg_points) # correct scalar shift
    return b


def find_bias(y, th=3, ns=None):
    if ns is None:
        ns = rolling_sd_pd(y)
    return np.median(y[np.abs(y-np.median(y)) <= th*ns])


@jit
def find_bias_frames(frames, th, ns):
    signals = core.ah.ravel_frames(frames).T
    nsr = np.ravel(ns)
    #print(nsr.shape, signals.shape)
    biases = np.zeros(nsr.shape)
    for j in range(len(biases)):
        biases[j] = find_bias(signals[j],th,nsr[j])
    #biases = np.array([find_bias(v,th,ns_) for  v,ns_ in zip(signals, nsr)])
    return biases.reshape(frames[0].shape)



def multi_scale_simple_baseline(y, plow=50, th=3, smooth_levels=[10,20,40,80,160], ns=None):
    if ns is None:
        ns = rolling_sd_pd(y)
    b_estimates = [simple_baseline(y,plow,th,smooth,ns) for smooth in smooth_levels]
    low_env = np.amin(b_estimates, axis=0)
    low_env = np.clip(low_env,np.min(y), np.max(y))
    return  l2spline(low_env, np.min(smooth_levels))






from scipy.stats import skew

@jit
def local_jitter(v, sigma=5):
    L = len(v)
    vx = np.copy(v)
    Wvx = np.zeros(L)
    for i in range(L):
        j = i + int(round(randn()*sigma))
        j = max(0,min(j,L-1))
        vx[i] = v[j]
        vx[j] = v[i]
    return vx


def std_median(v,axis=None):
    if axis is None:
        N = float(len(v))
    else:
        N = float(v.shape[axis])
    md = np.median(v,axis=axis)
    return (np.sum((v-md)**2,axis)/N)**0.5

def mad_std(v,axis=None):
    mad = np.median(abs(v-np.median(v,axis=axis)),axis=axis)
    return mad*1.4826

def iterative_noise_sd(data, cut=5, axis=None, niter=10):
    data = np.copy(data)
    for i in range(niter):
        sd = np.std(data, axis=axis)
        mu = np.mean(data, axis=axis)
        outliers = np.abs(data-mu) > cut*sd
        data = where(outliers, data*0.5, data)
        #data[outliers] = cut*sd
    return sd

def closing_of_opening(m,s=None):
    return ndi.binary_closing(ndi.binary_opening(m,s),s)

def refine_mask_by_percentile_filter(m, p=50, size=3,niter=1,with_cleanup=False,min_obj_size=2):
    out = np.copy(m).astype(bool)
    for i in range(niter):
        out += ndi.percentile_filter(out,p,size).astype(bool)
        if with_cleanup:
            out = threshold_object_size(out, min_obj_size)
    return out

def adaptive_median_filter(frames,th=5, tsmooth=1,ssmooth=5, keep_clusters=False, reverse=False, min_cluster_size=7):
    smoothed_frames = ndi.median_filter(frames, (tsmooth,ssmooth,ssmooth))
    details = frames - smoothed_frames
    sdmap = mad_std(frames,axis=0)
    outliers = np.abs(details) > th*sdmap
    #s = np.array([[[0,0,0],[0,1,0],[0,0,0]]]*3)
    if keep_clusters:
        clusters = threshold_object_size(outliers,min_cluster_size)
        outliers = ~clusters if reverse else outliers^clusters
    else:
        if reverse:
            outliers = ~outliers
    return np.where(outliers, smoothed_frames, frames)


def adaptive_filter_1d(v,th=5, smooth=5, smoother=ndi.median_filter, keep_clusters=False, reverse=False, min_cluster_size=5):
    vsmooth = smoother(v, smooth)
    details = v-vsmooth
    sd = mad_std(v)
    outliers = np.abs(details) > th*sd
    if keep_clusters:
        clusters = threshold_object_size(outliers,min_cluster_size)
        outliers = ~clusters if reverse else outliers^clusters
    else:
        if reverse:
            outliers = ~outliers
    return np.where(outliers, vsmooth, v)


def adaptive_filter_2d(img,th=5, smooth=5, smoother=ndi.median_filter, keep_clusters=False, reverse=False, min_cluster_size=5):
    imgf = smoother(img, smooth)
    details = img - imgf
    sd = mad_std(img)
    outliers = np.abs(details) > th*sd # in real adaptive filter the s.d. must be rolling!
    if keep_clusters:
        clusters = threshold_object_size(outliers,min_cluster_size)
        outliers = ~clusters if reverse else outliers^clusters
    else:
        if reverse:
            outliers = ~outliers
    return np.where(outliers, imgf, img)


# def adaptive_median_filter_frames(frames,th=5, tsmooth=5,ssmooth=1):
#     medfilt = ndi.median_filter(frames, [tsmooth,ssmooth,ssmooth])
#     details = frames - medfilt
#     mdmap = np.median(details, axis=0)
#     sdmap = np.median(abs(details - mdmap), axis=0)*1.4826
#     return np.where(abs(details-mdmap)  >  th*sdmap, medfilt, frames)

def rolling_sd(v,hw=None,with_plots=False,correct_factor=1.,smooth_output=True,input_is_details=False):
    if not input_is_details:
        details = v-ndi.median_filter(v,20)
    else:
        details = v
    if hw is None: hw = int(len(details)/10.)
    padded = np.pad(details,hw,mode='reflect')
    tv = np.arange(len(details))
    out = np.zeros(len(details))
    for i in np.arange(len(details)):
        out[i] = mad_std(padded[i:i+2*hw])
    if with_plots:
        f,ax = plt.subplots(1,1,sharex=True)
        ax.plot(tv,details,'gray')
        ax.plot(tv,out,'y')
        ax.plot(tv,2*out,'orange')
        ax.set_xlim(0,len(v))
        ax.set_title('Estimating running s.d.')
        ax.set_xlabel('samples')
    out = out/correct_factor
    if smooth_output:
        out = l2spline(out, s=2*hw)
    return out

def rolling_sd_scipy(v,hw=None,with_plots=False,correct_factor=1.,smooth_output=True,input_is_details=False):
    if not input_is_details:
        details = v-ndi.median_filter(v,20)
    else:
        details = v
    if hw is None: hw = int(len(details)/10.)
    padded = np.pad(details,hw,mode='reflect')
    tv = np.arange(len(details))
    #out = np.zeros(len(details))

    #rolling_median = lambda x: ndi.median_filter(x, 2*hw)
    rolling_median = partial(ndi.median_filter, size=2*hw)

    out = 1.4826*rolling_median(np.abs(padded-rolling_median(padded)))[hw:-hw]

    if with_plots:
        f,ax = plt.subplots(1,1,sharex=True)
        ax.plot(tv,details,'gray')
        ax.plot(tv,out,'y')
        ax.plot(tv,2*out,'orange')
        ax.set_xlim(0,len(v))
        ax.set_title('Estimating running s.d.')
        ax.set_xlabel('samples')
    out = out/correct_factor
    if smooth_output:
        out = l2spline(out, s=2*hw)
    return out

def rolling_sd_scipy_nd(arr,hw=None,correct_factor=1.,smooth_output=True):
    if hw is None: hw = int(np.ceil(np.max(arr.shape)/10))
    padded = np.pad(arr,hw,mode='reflect')
    rolling_median = lambda x: ndi.median_filter(x, 2*hw)
    crop = (slice(hw,-hw),)*np.ndim(arr)
    out = 1.4826*rolling_median(np.abs(padded-rolling_median(padded)))[crop]

    out = out/correct_factor
    if smooth_output:
        out = l2spline(out, s=hw)
    return out

def baseline_als_spl(y, k=0.5, tau=11, smooth=25., p=0.001, niter=100,eps=1e-4,
                 rsd = None,
                 rsd_smoother = None,
                 smoother = l2spline,
                 asymm_ratio = 0.9, correct_skew=False):
    """Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005),
    via DCT-based spline smoothing
    """
    #npad=int(smooth)
    nsmooth = np.int(np.ceil(smooth))
    npad =nsmooth

    y = np.pad(y,npad,"reflect")
    L = len(y)
    w = np.ones(L)

    if rsd is None:
        if rsd_smoother is None:
            #rsd_smoother = lambda v_: l2spline(v_, 5)
            #rsd_smoother = lambda v_: ndi.median_filter(y,7)
            rsd_smoother = partial(ndi.median_filter, size=7)
        rsd = rolling_sd_pd(y-rsd_smoother(y), input_is_details=True)
    else:
        rsd = np.pad(rsd, npad,"reflect")

    #ys = l1spline(y,tau)
    ntau = np.int(np.ceil(tau))
    ys = ndi.median_filter(y,ntau)
    s2 = l1spline(y, smooth/4.)
    #s2 = l2spline(y,smooth/4.)
    zprev = None
    for i in range(niter):
        z = smoother(ys,s=smooth,weights=w)
        clip_symm = abs(y-z) > k*rsd
        clip_asymm = y-z > k*rsd
        clip_asymm2 = y-z <= -k*rsd
        r = asymm_ratio#*core.rescale(1./(1e-6+rsd))

        #w = p*clip_asymm + (1-p)*(1-r)*(~clip_symm) + (1-p)*r*(clip_asymm2)
        w = p*(1-r)*clip_asymm + (1-p)*(~clip_symm) + p*r*(clip_asymm2)
        w[:npad] = (1-p)
        w[-npad:] = (1-p)
        if zprev is not None:
            if norm(z-zprev)/norm(zprev) < eps:
                break
        zprev=z
    z = smoother(np.min((z, s2),0),smooth)
    if correct_skew:
        # Correction for skewness introduced by asymmetry.
        z += r*rsd
    return z[npad:-npad]

def double_scale_baseline(y,smooth1=15.,smooth2=25.,rsd=None,**kwargs):
    """
    Baseline estimation in 1D signals by asymmetric smoothing and using two different time scales
    """
    if rsd is None:
        #rsd_smoother = lambda v_: ndi.median_filter(y,7)
        rsd_smoother = partial(ndi.median_filter, size=7)
        rsd = rolling_sd_pd(y-rsd_smoother(y), input_is_details=True)
    b1 = baseline_als_spl(y,tau=smooth1,smooth=smooth1,rsd=rsd,**kwargs)
    b2 = baseline_als_spl(y,tau=smooth1,smooth=smooth2,rsd=rsd,**kwargs)
    return l2spline(np.amin([b1,b2],0),smooth1)


def viz_baseline(v,dt=1.,baseline_fn=baseline_als_spl,
                 smoother=partial(l2spline,s=5),ax=None,**kwargs):
    """
    Visualize results of baseline estimation
    """
    if ax is None:
        plt.figure(figsize=(14,6))
        ax = plt.gca()
    tv = np.arange(len(v))*dt
    ax.plot(tv,v,'gray')
    b = baseline_fn(v,**kwargs)
    rsd = rolling_sd_pd(v-smoother(v))
    ax.fill_between(tv, b-rsd,b+rsd, color='y',alpha=0.75)
    ax.fill_between(tv, b-2.0*rsd,b+2.0*rsd, color='y',alpha=0.5)
    ax.plot(tv,smoother(v),'k')
    ax.plot(tv,b, 'teal',lw=1)
    ax.axis('tight')


# Labeling algorithms

def percentile_label(v, percentile_low=2.5,tau=2.0,smoother=l2spline):
    mu = min(np.median(v),0)
    low = np.percentile(v[v<=mu], percentile_low)
    vs = smoother(v, tau)
    return vs >= mu-low

def simple_label(v, threshold=1.0,tau=5., smoother=l2spline,**kwargs):
    vs = smoother(v, tau)
    return vs >= threshold

def with_local_jittering(labeler, niters=100, weight_thresh=0.85):
    def _(v, *args, **kwargs):
        if 'tau' in kwargs:
            tau = kwargs['tau']
        else:
            tau = 5.0
        labels_history = np.zeros((niters,len(v)))
        for i_ in range(niters):
            vi = local_jitter(v,0.5*tau)
            #labels_history.append(labeler(vi, *args, **kwargs))
            labels_history[i_] =labeler(vi, *args, **kwargs)
        return np.mean(labels_history,0) >= weight_thresh
    return _

simple_label_lj = with_local_jittering(simple_label)
percentile_label_lj = with_local_jittering(percentile_label)

thresholds_l1 = np.array([2.26212451,  1.11505896,  0.52321721,  0.51701626,  0.42481402,
                          0.34870014,  0.29144794,  0.24410656,  0.20409004,  0.16792375,
                          0.13579082,  0.10770976])
thresholds_l1 = thresholds_l1.reshape(-1,1)


thresholds_l2 = np.array([ 1.6452271 ,  0.64617428,  0.41641932,  0.32425908,  0.26115802,
                           0.21203462,  0.17222229,  0.14062114,  0.11350558,  0.0896438 ,
                           0.06936852,  0.05300952])
thresholds_l2 = thresholds_l2.reshape(-1,1)



def multiscale_labeler_l1(signal,thresh=2,start=1,**kwargs):
    coefs = sp_decompose(signal, level=12, smoother=l1spline,base=1.5)[start:-1]
    labels = (coefs>=thresholds_l1[start:]).sum(axis=0)>=thresh
    return labels

def multiscale_labeler_l2(signal,thresh=4,start=1,**kwargs):
    #thresholds_l2 = array([1.6453141 , 0.64634246, 0.41638476, 0.3242796 , 0.2611729 ,
    #                       0.21204839, 0.17224974, 0.14053809, 0.11334435, 0.08955742,
    #                       0.06948411, 0.05307127]).reshape(-1,1)
    coefs = sp_decompose(signal, level=12, smoother=l2spline,base=1.5)[start:-1]
    labels = (coefs>=thresholds_l2[start:]).sum(axis=0)>=thresh
    return labels

def make_labeler_commitee(*labelers):
    Nl = len(labelers)
    def _(v,**kwargs):
        labels = [lf(v) for lf in labelers]
        return np.sum(labels,0)==Nl
    return _

multiscale_labeler_l1l2 = make_labeler_commitee(multiscale_labeler_l1,
                                               partial(multiscale_labeler_l2, start=2,thresh=3))

multiscale_labeler_joint = make_labeler_commitee(multiscale_labeler_l1,
                                                 partial(multiscale_labeler_l2, start=2,thresh=3),
                                                 simple_label_lj)


# Reconstruction




from imfun import bwmorph


def simple_pipeline_(y, labeler=percentile_label,labeler_kw=None,smoothed_rec=True,
                     noise_sigma = None,
                     correct_bias = True):

    """
    Detect and reconstruct Ca-transients in 1D signal
    """
    if not any(y):
        return np.zeros_like(y)

    ns = rolling_sd_pd(y) if noise_sigma is None else noise_sigma

    if correct_bias:
        bias = find_bias(y,th=1.5,ns=ns)
        #low = y < np.median(y) + 1.5*ns
        #if not any(low):
        #    low = np.ones(len(y),np.bool)
        #    bias = np.median(y[low])
        #    if bias > 0:
        #        y = y-bias
        y = y-bias
    vn = y/ns
    #labels = simple_label_lj(vn, tau=tau_label_,with_plots=False)
    if labeler_kw is None:
        labeler_kw={}
    labels = labeler(vn, **labeler_kw)
    if not any(labels):
        return np.zeros_like(y)
    return sp_rec_with_labels(vn, labels,with_plots=False,return_smoothed=smoothed_rec)*ns


#from multiprocessing import Pool
from pathos.pools import ProcessPool as Pool
def process_signals_parallel(collection, pipeline=simple_pipeline_,pipeline_kw=None,njobs=4):
    """
    Process temporal signals some pipeline function and return processed signals
    (parallel version)
    """
    out =[]
    pool = Pool(njobs)
    #def _pipeline_(*args):
    #    if pipeline_kw is not None:
    #        return pipeline(*args, **pipeline_kw)
    #    else:
    #        return pipeline(*args)
    _pipeline_ = pipeline if pipeline_kw is None else partial(pipeline, **pipeline_kw)
    recs = pool.map(_pipeline_, [c[0] for c in collection], chunksize=4) # setting chunksize here is experimental
    #pool.close()
    #pool.join()
    return [(r,s,w) for r,(v,s,w) in zip(recs, collection)]




def sp_rec_with_labels(vec, labels,
                       min_scale=1.0,max_scale=50.,
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
        filtered_labels = np.zeros_like(labels)+np.sum([r.tomask() for r in regions],axis=0)
    else:
        filtered_labels=labels

    if not sum(filtered_labels):
        return np.zeros_like(vec)
    vec1 = np.copy(vec)
    vs = smoother(vec1, min_scale, wmedian)
    weights = np.clip(labels, 0,1)

    #vss = smoother(vec-vs,max_scale,weights=weights<0.9)

    vrec = smoother(vs*(vec1>0),min_scale,wmedian)

    for i in range(niters):
        #vec1 = vec1 - kgain*(vec1-vrec) # how to use it?
        labs,nl = ndi.label(weights)
        objs = ndi.find_objects(labs)
        #for o in objs:
        #    stop = o[0].stop
        #    while stop < len(vec) and vrec[stop]>0.25*vrec[o].max():
        #        weights[stop] = 1
        #        stop+=1
        wer_grow = ndi.binary_dilation(weights)
        wer_shrink = ndi.binary_erosion(weights)
        #weights = np.where((vec1<np.mean(vec1[vec1>0])), wer, weights)
        if np.any(vrec>0):
            weights = np.where(vrec<0.5*np.mean(vrec[vrec>0]), wer_shrink, weights)
            weights = np.where(vrec>1.25*np.mean(vrec[vrec>0]), wer_grow, weights)
        vrec = smoother(vec*weights,min_scale,wmedian)
        #weights = ndi.binary_opening(weights)
        vrec[vrec<0] = 0
        #plt.figure(); plt.plot(vec1)
        #vrec[weights<0.5] *=0.5


    if with_plots:
        f,ax = plt.subplots(1,1)
        ax.plot(vec, '-',ms=2, color='gray',lw=0.5,alpha=0.5)
        ax.plot(vec1, '-', color='cyan',lw=0.75,alpha=0.75)
        ax.plot(weights, 'g',lw=2,alpha=0.5)
        ax.plot(vs, color='k',alpha=0.5)
        #plot(vss,color='navy',alpha=0.5)
        ax.plot(vrec, color='royalblue',lw=2)
        ll = np.where(labels)[0]
        ax.plot(ll,-1*np.ones_like(ll),'r|')
    if return_smoothed:
        return vrec
    else:
        return vec*(vrec>0)*(vec>0)*weights #weights*(vrec>0)*vec


def simple_pipeline_nojitter_(y,tau_label=1.5):
    """
    Detect and reconstruct Ca-transients in 1D signal
    """
    ns = rolling_sd_pd(y)
    low = y < 2.5*np.median(y)
    if not any(low):
        low = np.ones(len(y),np.bool)
    bias = np.median(y[low])
    if bias > 0:
        y = y-bias
    vn = y/ns
    labels = simple_label(vn, tau=tau_label,with_plots=False)
    return y * labels
    #return sp_rec_with_labels(vn, labels,niters=5,with_plots=False)*ns


def simple_pipeline_with_baseline(y,tau_label=1.5):
    """
    Detect and reconstruct Ca-transients in 1D signal after normalizing to baseline
    #b,ns,_ = tmvm_baseline(y)
    """
    #b = b + np.median(y-b)
    ns = rolling_sd_pd(y)
    b = multi_scale_simple_baseline(y,ns=ns)
    vn = (y-b)/ns
    labels = simple_label_lj(vn, tau=tau_label,with_plots=False)
    rec = sp_rec_with_labels(y, labels,with_plots=False,)
    return np.where(b>0,rec,0)

#from multiprocessing import Pool
#def process_signals_parallel(collection, pipeline=simple_pipeline_,njobs=4):
#    """
#    Process temporal signals some pipeline function and return processed signals
#    (parallel version)
#    """
#    out =[]
#    pool = Pool(njobs)
#    recs = pool.map(pipeline, [c[0] for c in collection], chunksize=4) # setting chunksize here is experimental
#    pool.close()
#    pool.join()
#    return [(r,s,w) for r,(v,s,w) in zip(recs, collection)]


def quantify_events(rec, labeled, dt=1):
    "Collect information about transients for a 1D reconstruction"
    acc = []
    idx = np.arange(len(rec))
    for i in range(1,np.max(labeled)+1):
        mask = labeled==i
        cut = rec[mask]
        ev = dict(
            start = np.min(idx[mask]),
            stop = np.max(idx[mask]),
            peak = np.max(cut),
            time_to_peak = np.argmax(cut),
            vmean = np.mean(cut))
        acc.append(ev)
    return acc

from imfun.core import extrema

def segment_events_1d(rec, th=0.05, th2 =0.1, smoothing=6, min_lenth=3):
    levels = rec>th
    labeled, nlab = ndi.label(levels)
    smrec = l1spline(rec, smoothing)
    #smrec = l2spline(rec, 6)
    mxs = np.array(extrema.locextr(smrec, output='max',refine=False))
    mns = np.array(extrema.locextr(smrec, output='min',refine=False))
    if not len(mxs) or not len(mns) or not np.any(mxs[:,1]>th2):
        return labeled, nlab
    mxs = mxs[mxs[:,1]>th2]
    cuts = []

    for i in range(1,nlab+1):
        mask = labeled==i
        lmax = [m for m in mxs if mask[int(m[0])]]
        if len(lmax)>1:
            th = np.max([m[1] for m in lmax])*0.75
            lms = [mn for mn in mns if mask[int(mn[0])] and mn[1]<th]
            if len(lms):
                for lm in lms:
                    tmp_mask = mask.copy()
                    tmp_mask[int(lm[0])] = 0
                    ll_,nl_ = ndi.label(tmp_mask)
                    min_region = np.min([np.sum(ll_ == i_) for i_ in range(1,nl_+1)])
                    if min_region > min_lenth:
                        cuts.append(lm[0])
                        levels[int(lm[0])]=False

    labeled,nlab=ndi.label(levels)

    #plot(labeled>0)
    return labeled, nlab
    #    plot(rec*(labeled==i),alpha=0.5)


def pca_flip_signs(pcf,medianw=None):
    L = len(pcf.coords)
    if medianw is None:
        medianw = L//5
    for i,c in enumerate(pcf.coords.T):
        sk = skew(c-ndi.median_filter(c,medianw))
        sg = np.sign(sk)
        #print(i, sk)
        pcf.coords[:,i] *= sg
        pcf.tsvd.components_[i]*=sg
    return pcf

# def svd_flip_signs(u,vh,medianw=None):
#     L = len(u)
#     if medianw is None:
#         medianw = L//5
#     for i,c in enumerate(u.T):
#         sk = skew(c-ndi.median_filter(c,medianw))
#         sg = np.sign(sk)
#         u[:,i] *= sg
#         vh[i] *= sg
#     return u,vh


def calculate_baseline(frames,pipeline=multi_scale_simple_baseline, stride=2,patch_size=5,return_type='array',
                       pipeline_kw=None):
    """
    Given a TXY frame timestack estimate slowly-varying baseline level of fluorescence using patch-based processing
    """
    from imfun import fseq
    collection = signals_from_array_avg(frames,stride=stride, patch_size=patch_size)
    recsb = process_signals_parallel(collection, pipeline=pipeline, pipeline_kw=pipeline_kw,njobs=4, )
    sh = frames.shape
    out =  combine_weighted_signals(recsb, sh)
    if return_type.lower() == 'array':
        return out
    fsx = fseq.from_array(out)
    fsx.meta['channel'] = 'baseline'
    return fsx


def lambda_star(beta):
    return np.sqrt(2*(beta+1) + (8*beta)/(beta+1 + np.sqrt(beta**2 + 14*beta + 1)))

def omega_approx(beta):
    return 0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43

def svht(sv, sh, sigma=None):
    "Gavish and Donoho 2014"
    m,n = sh
    if m>n:
        m,n=n,m
    beta = m/n
    omg = omega_approx(beta)
    if sigma is None:
        return omg*np.median(sv)
    else:
        return lambda_star(beta)*np.sqrt(n)*sigma

def min_ncomp(sv,sh,sigma=None):
    th = svht(sv,sh,sigma)
    return sum(sv >=th)


def calculate_baseline_pca(frames,smooth=60,npc=None,pcf=None,return_type='array',smooth_fn=baseline_als_spl):
    """Use smoothed principal components to estimate time-varying baseline fluorescence F0
    -- deprecated
"""
    from imfun import fseq

    if pcf is None:
        if npc is None:
            npc = len(frames)//20
        pcf = components.pca.PCA_frames(frames,npc=npc)
    pca_flip_signs(pcf)
    #base_coords = np.array([smoothed_medianf(v, smooth=smooth1,wmedian=smooth2) for v in pcf.coords.T]).T
    if smooth > 0:
        base_coords = np.array([smooth_fn(v,smooth=smooth) for v in pcf.coords.T]).T
        #base_coords = np.array([multi_scale_simple_baseline(v) for v in pcf.coords.T]).T
    else:
        base_coords = pcf.coords
    #base_coords = np.array([double_scale_baseline(v,smooth1=smooth1,smooth2=smooth2) for v in pcf.coords.T]).T
    #base_coords = np.array([simple_get_baselines(v) for v in pcf.coords.T]).T
    baseline_frames = pcf.tsvd.inverse_transform(base_coords).reshape(len(pcf.coords),*pcf.sh) + pcf.mean_frame
    if return_type.lower() == 'array':
        return baseline_frames
    #baseline_frames = base_coords.dot(pcf.vh).reshape(len(pcf.coords),*pcf.sh) + pcf.mean_frame
    fs_base = fseq.from_array(baseline_frames)
    fs_base.meta['channel'] = 'baseline_pca'
    return fs_base

def calculate_baseline_pca_asym(frames,niter=50,ncomp=20,smooth=25,th=1.5,verbose=False):
    """Use asymetrically smoothed principal components to estimate time-varying baseline fluorescence F0"""
    frames_w = np.copy(frames)
    sh = frames.shape
    nbase = np.linalg.norm(frames)
    diff_prev = np.linalg.norm(frames_w)/nbase
    for i in range(niter+1):
        pcf = components.pca.PCA_frames(frames_w, npc=ncomp)
        coefs = np.array([l2spline(v,smooth) for v in pcf.coords.T]).T
        rec = pcf.inverse_transform(coefs)
        diff_new = np.linalg.norm(frames_w - rec)/nbase
        epsx = diff_new-diff_prev
        diff_prev = diff_new

        if not i%5:
            if verbose:
                sys.stdout.write('%0.1f %% | '%(100*i/niter))
                print('explained variance %:', 100*pcf.tsvd.explained_variance_ratio_.sum(), 'update: ', epsx)
        if i < niter:
            delta=frames_w-rec
            thv = th*np.std(delta,axis=0)
            frames_w = np.where(delta>thv, rec, frames_w)
        else:
            if verbose:
                print('\n finished iterations')
            delta = frames-rec
            #ns0 = np.median(np.abs(delta - np.median(delta,axis=0)), axis=0)*1.4826
            ns0 = mad_std(delta, axis=0)
            biases = find_bias_frames(delta,3,ns0)
            biases[np.isnan(biases)] = 0
            frames_w = rec + biases#np.array([find_bias(delta[k],ns=ns0[k]) for k,v in enumerate(rec)])[:,None]

    return frames_w

## TODO use NMF or NNDSVD instead of PCA?
from sklearn import decomposition as skd
from imfun import core
def _calculate_baseline_nmf(frames, ncomp=None, return_type='array',smooth_fn=multi_scale_simple_baseline):
    """DOESNT WORK! Use smoothed NMF components to estimate time-varying baseline fluorescence F0"""
    from imfun import fseq

    fsh = frames[0].shape

    if ncomp is None:
        ncomp = len(frames)//20
    nmfx = skd.NMF(ncomp,)
    signals = nmfx.fit_transform(core.ah.ravel_frames(frames))

    #base_coords = np.array([smoothed_medianf(v, smooth=smooth1,wmedian=smooth2) for v in pcf.coords.T]).T
    if smooth > 0:
        base_coords = np.array([smooth_fn(v,smooth=smooth) for v in pcf.coords.T]).T
        #base_coords = np.array([multi_scale_simple_baseline for v in pcf.coords.T]).T
    else:
        base_coords = pcf.coords
    #base_coords = np.array([double_scale_baseline(v,smooth1=smooth1,smooth2=smooth2) for v in pcf.coords.T]).T
    #base_coords = np.array([simple_get_baselines(v) for v in pcf.coords.T]).T
    baseline_frames = pcf.tsvd.inverse_transform(base_coords).reshape(len(pcf.coords),*pcf.sh) + pcf.mean_frame
    if return_type.lower() == 'array':
        return baseline_frames
    #baseline_frames = base_coords.dot(pcf.vh).reshape(len(pcf.coords),*pcf.sh) + pcf.mean_frame
    fs_base = fseq.from_array(baseline_frames)
    fs_base.meta['channel'] = 'baseline_pca'
    return fs_base


def get_baseline_frames(frames,smooth=60,npc=None,baseline_fn=multi_scale_simple_baseline,baseline_kw=None):
    """
    Given a TXY frame timestack estimate slowly-varying baseline level of fluorescence, two-stage processing
    (1) global trends via PCA
    (2) local corrections by patch-based algorithm
    """
    from imfun import fseq
    base1 = calculate_baseline_pca(frames,smooth=smooth,npc=npc,smooth_fn=multi_scale_simple_baseline)
    base2 = calculate_baseline(frames-base1, pipeline=baseline_fn, pipeline_kw=baseline_kw,patch_size=5)
    fs_base = fseq.from_array(base1+base2)
    fs_base.meta['channel']='baseline_comb'
    return fs_base

from imfun.core import coords
from numpy.linalg import svd
#from multiprocessing import Pool
def map_patches(fn, data,patch_size=10,stride=1,tslice=slice(None),njobs=1):
    """
    Apply some function to a square patch exscized from video
    """
    sh = data.shape[1:]
    squares = list(map(tuple, coords.make_grid(sh, patch_size, stride)))
    if njobs>1:
        pool = Pool(njobs)
        expl_m = pool.map(fn, (data[(tslice,) + s] for s in squares))
    else:
        expl_m = [fn(data[(tslice,) + s]) for s in squares]
    out = np.zeros(sh);
    counts = np.zeros(sh);
    for _e, s in zip(expl_m, squares):
        out[s] += _e; counts[s] +=1.
    return out/counts

from imfun.core import extrema
from numpy import fft
def roticity_fft(data,period_low = 100, period_high=5,npc=6):
    """
    Look for local areas with oscillatory dynamics in TXY framestack
    """
    L = len(data)
    if np.ndim(data)>2:
        data = data.reshape(L,-1)
    Xc = data.mean(0)
    data = data-Xc
    npc = min(npc, data.shape[-1])
    u,s,vh = svd(data,full_matrices=False)
    s2 = s**2/(s**2).sum()
    u = (u-u.mean(0))[:,:npc]
    p = (abs(fft.fft(u,axis=0))**2)[:L//2]
    nu = fft.fftfreq(len(data))[:L//2]
    nu_phys = (nu>1/period_low)*(nu<period_high)
    peak = 0
    sum_peak = 0
    for i in range(npc):
        pi = p[:,i]
        pbase = smoothed_medianf(pi, 5, 50)
        psmooth = smoothed_medianf(pi, 1,5)
        #pi = pi/psmooth-1
        lm = np.array(extrema.locextr(psmooth,x=nu,refine=True,output='max'))
        lm = lm[(lm[:,0]>1/period_low)*(lm[:,0]<1/period_high)]
        #peak_ = np.amax(lm[:,1])/psmooth[~nu_phys].mean()*s2[i]
        k = np.argmax(lm[:,1])
        nuk = lm[k,0]
        kx = np.argmin(np.abs(nu-nuk))
        peak_ = lm[k,1]/pbase[kx]*s2[i]
        #peak_ = np.amax(lm[:,1])#*s2[i]
        #print(amax(lm[:,1]),std(p[:,i]),peak_)
        sum_peak += peak_
        peak = max(peak, peak_)
    return sum_peak


def make_enh4(frames, pipeline=simple_pipeline_,
              labeler=percentile_label,
              kind='pca', nhood=5, stride=2, mask_of_interest=None,
              pipeline_kw=None,
              labeler_kw=None):
    from imfun import fseq
    #coll = signals_from_array_pca_cluster(frames,stride=2,dbscan_eps=0.05,nhood=5,walpha=0.5)
    if kind.lower()=='corr':
        coll = signals_from_array_correlation(frames,stride=stride,nhood=nhood,mask_of_interest=mask_of_interest)
    elif kind.lower()=='pca':
        coll = signals_from_array_pca_cluster(frames,stride=stride,nhood=nhood,
                                              mask_of_interest=mask_of_interest,
                                              ncomp=2,
        )
    else:
        coll = signals_from_array_avg(frames,stride=stride,patch_size=nhood*2+1,mask_of_interest=mask_of_interest)
    print('\nTime-signals, grouped,  processing (may take long time) ...')
    if pipeline_kw is None:
        pipeline_kw = {}
    pipeline_kw.update(labeler=labeler,labeler_kw=labeler_kw)
    coll_enh = process_signals_parallel(coll,pipeline=pipeline, pipeline_kw=pipeline_kw,)
    print('Time-signals processed, recombining to video...')
    out = combine_weighted_signals(coll_enh,frames.shape)
    fsx = fseq.from_array(out)
    print('Done')
    fsx.meta['channel']='-'.join(['newrec4',kind])
    return fsx

def svd_denoise_tslices(frames, twindow=50,
                        nhood=5,
                        npc=None,
                        mask_of_interest=None,
                        th = 0.05,
                        verbose=True,
                        denoiser=_patch_pca_denoise_with_shifts,
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

    out = np.zeros(frames.shape)
    for k,ts,m in zip(range(L), tslices, mask_list):
        out[ts] += denoiser(frames[ts], mask_of_interest=m, nhood=nhood, npc=npc, **denoiser_kw)
        counts[ts] +=1
        if verbose:
            sys.stdout.write('\n processed time-slice %d out of %d\n'%(k+1, len(tslices)))

    return out/counts[:,None,None]

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



def make_enh5(dfof, twindow=50, nhood=5, stride=2, temporal_filter=3, verbose=False):
    from imfun import fseq
    amask = activity_mask_median_filtering(dfof, nw=7,verbose=verbose)
    nsf = mad_std(dfof, axis=0)
    dfof_denoised = svd_denoise_tslices(dfof,twindow, mask_of_interest=amask, temporal_filter=temporal_filter, verbose=verbose)
    mask_active = dfof_denoised > nsf
    mask_active = opening_of_closing(mask_active) + ndi.median_filter(mask_active,3)>0
    dfof_denoised2 = np.array([avg_filter_greater(f, 0) for f in dfof_denoised*mask_active])
    fsx = fseq.from_array(dfof_denoised2)
    if verbose:
        print('Done')
    fsx.meta['channel']='-'.join(['newrec5'])
    return fsx


def max_shifts(shifts,verbose=0):
    #ms = np.max(np.abs([s.fn_((0,0)) if s.fn_ else s.field[...,0,0] for s in shifts]),axis=0)
    ms = np.max([np.array([np.percentile(f,99) for f in np.abs(w.field)]) for w in shifts],0)
    if verbose: print('Maximal shifts were (x,y): ', ms)
    return np.ceil(ms).astype(int)

def crop_by_max_shift(data, shifts, mx_shifts=None):
    if mx_shifts is None:
        mx_shifts = max_shifts(shifts)
    lims = 2*mx_shifts
    sh = data.shape[1:]
    return data[:,lims[1]:sh[0]-lims[1],lims[0]:sh[1]-lims[0]]


def scramble_data(frames):
    L,nr,nc = frames.shape
    out = zeros_like(frames)
    for r in range(nr):
        for c in range(nc):
            out[:,r,c] = np.random.permutation(frames[:,r,c])
    return out

def process_framestack(frames,min_area=9,verbose=False,
                       do_dfof_denoising = True,
                       baseline_fn = multi_scale_simple_baseline,
                       baseline_kw = dict(smooth_levels=(10,20,40,80)),
                       pipeline=simple_pipeline_,
                       labeler=percentile_label,
                       labeler_kw=None):
    """
    Default pipeline to process a stack of frames containing Ca fluorescence to find astrocytic Ca events
    Input: F(t): temporal stack of frames (Nframes x Nx x Ny)
    Output: Collection of three frame stacks containting ΔF/F0 signals, one thresholded and one denoised, and a baseline F0(t):
            fseq.FStackColl([fsx, dfof_filtered, F0])
    """
    from imfun import fseq
    if verbose:
        print('calculating baseline F0(t)')
    #fs_f0 = get_baseline_frames(frames[:],baseline_fn=baseline_fn, baseline_kw=baseline_kw)
    fs_f0 = calculate_baseline_pca_asym(frames[:],verbose=True,niter=20)
    fs_f0 = fseq.from_array(fs_f0)
    fs_f0.meta['channel'] = 'F0'

    dfof= frames/fs_f0.data - 1

    if do_dfof_denoising:
        if verbose:
            print('filtering ΔF/F0 data')
        dfof = patch_pca_denoise2(dfof, spatial_filter=3, temporal_filter=1, npc=5)
    fs_dfof = fseq.from_array(dfof)
    fs_dfof.meta['channel'] = 'ΔF_over_F0'

    if verbose:
        print('detecting events')
    ## todo: decide, whether we actually need the cleaning step.
    ## another idea: use dfof for detection to avoid FP, use dfof_cleaned for reconstruction because of better SNR?
    ##               but need to show that FP is lower, TP is OK and FN is low for this combo
    ## motivation:   using filters in dfof_cleaned introduces spatial correlations, which may lead to higher FP
    ##               (with low amplitude though). Alternative option would be to guess a correct amplitude threshold
    ##               afterwards
    ## note: but need to test that on real data, e.g. on slices with OGB and gcamp
    fsx = make_enh4(dfof,nhood=2,kind='pca',pipeline=pipeline,labeler=labeler,labeler_kw=labeler_kw)
    coll_ = EventCollection(fsx.data,min_area=min_area)
    meta = fsx.meta
    fsx = fseq.from_array(fsx.data*(coll_.to_filtered_array()>0),meta=meta)
    fscoll = fseq.FStackColl([fsx, fs_dfof, fs_f0])
    return fscoll


def segment_events(dataset,threshold=0.01):
    labels, nlab = ndi.label(np.asarray(dataset,dtype=_dtype_)>threshold)
    objs = ndi.find_objects(labels)
    return labels, objs


class EventCollection:
    def __init__(self, frames, threshold=0.025,
                 dfof_frames = None,
                 gf_sigma = (0.5,2,2),
                 min_duration=3,
                 min_area=9,
                 peak_threshold=0.05):
        self.min_duration = min_duration
        self.labels, self.objs = segment_events(frames,threshold)
        self.coll = [dict(duration=self.event_duration(k),
                          area = self.event_area(k),
                          volume = self.event_volume(k),
                          peak = self.data_value(k,frames),
                          avg = self.data_value(k,frames,np.mean),
                          start=self.objs[k][0].start,
                          idx=k)
                    for k in range(len(self.objs))]
        self.filtered_coll = [c for c in self.coll
                              if c['duration']>min_duration \
                              and c['peak']>peak_threshold\
                              and c['area']>min_area]
        if dfof_frames is not None:
            dfofx = ndi.gaussian_filter(dfof_frames, sigma=gf_sigma, order=(1,0,0)) # smoothed first derivatives in time
            nevents = len(self.coll)
            for (k,event), obj in zip(enumerate(self.coll), self.objs):
                vmask = self.event_volume_mask(k)
                areas = [np.sum(m) for m in vmask]
                area_diff = ndi.gaussian_filter1d(areas, 1.5, order=1)
                event['mean_area_expansion_rate'] = area_diff[area_diff>0].mean() if any(area_diff>0) else 0
                event['mean_area_shrink_rate'] = area_diff[area_diff<0].mean() if any(area_diff<0) else 0
                dx = dfofx[obj]*vmask
                flatmask = np.sum(vmask,0)>0
                event['mean_peak_rise'] = (dx.max(axis=0)[flatmask]).mean()
                event['mean_peak_decay'] = (dx.min(axis=0)[flatmask]).mean()
                event['max_peak_rise'] = (dx.max(axis=0)[flatmask]).max()
                event['max_peak_decay'] = (dx.min(axis=0)[flatmask]).min()

    def event_duration(self,k):
        o = self.objs[k]
        return o[0].stop-o[0].start
    def event_volume_mask(self,k):
        return self.labels[self.objs[k]]==k+1
    def project_event_mask(self,k):
        return np.max(self.event_volume_mask(k),axis=0)
    def event_area(self,k):
        return np.sum(self.project_event_mask(k).astype(int))
    def event_volume(self,k):
        return np.sum(self.event_volume_mask(k))
    def data_value(self,k,data,fn = np.max):
        o = self.objs[k]
        return fn(data[o][self.event_volume_mask(k)])
    def to_DataFrame(self):
        return pd.DataFrame(self.filtered_coll)
    def to_csv(self,name):
        df = self.to_DataFrame()
        df.to_csv(name)
    def to_filtered_array(self):
        sh  = self.labels.shape
        out = np.zeros(sh,dtype=np.int)
        for d in self.filtered_coll:
            k = d['idx']
            o = self.objs[k]
            cond = self.labels[o]==k+1
            out[o][cond] = k
        return out


## -- this is temporary! ---
## -- copypaste of test_denoising.py --
#!/usr/bin/env python

## TODO: add second pass (correction) as an option

import os,sys
import h5py

import argparse
import pickle
import gzip
import json

from functools import partial,reduce
import operator as op



import numpy as np
from numpy import *
from numpy.linalg import norm, svd
from numpy.random import randint

from scipy import ndimage,signal
from scipy import ndimage as ndi
from scipy import stats

from sklearn import cluster as skcluster


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from skimage.external import tifffile

import pandas as pd

from tqdm import tqdm


from imfun import fseq,core,ui
from imfun import multiscale
from imfun import ofreg
from imfun.ofreg import stackreg, imgreg
from imfun.external import czifile

from imfun.filt.dctsplines import l2spline, l1spline

from imfun.multiscale import atrous
from imfun import components

# Some global parameters
# TODO: move to argparse
_baseline_smoothness_ = 300
_nclusters_ = 32
_do_pruning_ = False
_do_scrambling_ = False
_dtype_=np.float32

class Anscombe:
    "Variance-stabilizing transformation"
    @staticmethod
    def transform(data):
        return 2*(data+3/8)**0.5
    @staticmethod
    def inverse_transform(tdata):
        tdata_squared = tdata**2
        return tdata_squared/4 + np.sqrt(3/2)*(1/(4*tdata) + 5/(8*tdata**3)) - 11/8/tdata_squared -1/8

def slice_center_in_square(sl, sq):
    "test if center of a smaller n-dim slice is within a bigger n-dim slice"
    c = [(s.stop+s.start)*0.5 for s in sl]
    return np.all([dim.start <= cx < dim.stop for cx,dim in zip(c, sq)])

def slice_overlaps_square(sl, sq):
    "test if a smaller n-dim slice overlaps with a bigger n-dim slice"
    return np.all([((dim.start <= s.start < dim.stop) or (dim.start <= s.stop < dim.stop)) for s,dim in zip(sl, sq)])

def slice_starts_in_square(sl, sq):
    "test if start of a smaller n-dim slice is within a bigger n-dim slice"
    o = [s.start for s in sl]
    return np.all([dim.start <= ox < dim.stop for ox,dim in zip(o, sq)])

from scipy.stats import skew
def svd_flip_signs(u,vh, mode='v'):
    "flip signs of U,V pairs of the SVD so that either V or U have positive skewness"
    for i in range(len(vh)):
        if mode == 'v':
            sg = sign(skew(vh[i]))
        else:
            sg = sign(skew(u[:,i]))
        u[:,i] *= sg
        vh[i] *= sg
    return u,vh


def extract_random_cubic_patch(frames, w=10):
    """Extract small cubic patch at a random location from a stack of frames
    Parameters:
     - frames: TXY 3D array-like, a stack of frames
     - w : scalar int, side of the cubic patch [10]
    """
    sl = tuple()
    starts = (randint(0, dim-w) for dim in frames.shape)
    sl =  tuple(slice(j, j+w) for j in starts)
    return frames[sl]

def extract_random_column(frames, w=10):
    if not np.iterable(w):
        w = (w,)*np.ndim(frames)
    sh = frames.shape
    loc = tuple(randint(0,s-wi,) for s,wi in zip(sh,w))
    sl = tuple(slice(j,j+wi) for j,wi in zip(loc, w))
    #print(loc, sl)
    return frames[sl]



def _simple_stats(x):
    "Just return mean and variance of a sample"
    return (x.mean(), x.var())
    #mu = x.mean()
    #sigmasq = np.var(x[np.abs(x-mu)<3*np.std(x)])
    #return mu, sigmasq

from sklearn import linear_model

from imfun.core import extrema
def estimate_offset2(frames, smooth=None,nsteps=100,with_plot=False):
    mu = np.median(frames)
    sigma = np.std(concatenate((frames[frames<=mu], mu-frames[frames<=mu])))
    print('mu', mu, 'sigma', sigma)
    biases = np.linspace(mu-sigma/4, mu+sigma/4, nsteps)
    db = biases[1]-biases[0]
    v = array([np.mean(frames<n) for n in biases])
    if smooth is not None:
        dv = ndi.gaussian_filter1d(v, smooth, order=1)
        offset = biases[argmax(dv)]
    else:
        for smooth in arange(0.1*db,100*db,0.5*db):
            dv = ndi.gaussian_filter1d(v, smooth, order=1)
            peaks = extrema.locextr(dv,x=biases,output='max')
            if len(peaks) < 1:
                continue
            offset = peaks[0][0]
            if len(peaks) <= 1:
                break
        if not len(peaks):
            offset = biases[argmax(dv)]
    if with_plot:
        plt.figure()
        plt.plot(biases, v, '.-')
        plt.plot(biases, (dv-dv.min())/(dv.max()-dv.min()+1e-7))
        plt.axvline(offset, color='r',ls='--')

    return offset

# TODO: eventually move to μCats
def estimate_gain_and_offset(frames, patch_width=10,npatches=int(1e5),
                             ntries=200,
                             with_plot=False,save_to=None,
                             return_type='mean'):
    """
    Estimage gain and offset parameters used by the imaging system
    For a given stack of frames, extract many small cubic patches and then fit a line
    as variance ~ mean. The slope of the line is the gain, the intercept of the X axis
    is the offset or system bias.

    Parameters:
     - frames : TXY 3D array-like, a stack of frames to analyze
     - patch_width: scalar int, side of a cubic patch to randomly sample from frames (default: 20)
     - npatches: scalar int, number of random patches to draw from the frames (default: 100,000)
     - ntries: scalar int, number of patch ensembles to fit at a time, next parameters will be pooled (default: 100)
     - with_plot: bool, whether to make a plot of the mean-variance line fit
     - save_to: string, name of a file to save the plot to, has no effect if with_plot is False (default: None, i.e. don't save)
     - return_type: {'mean','min','median','ransac'} -- which estimate of the line fit to return
    """

    pxr = array([_simple_stats(extract_random_column(frames,patch_width)) for i in range(npatches)])
    cut = np.percentile(pxr[:,0],95)
    pxr = pxr[pxr[:,0]<cut]
    vm,vv=pxr.T

    gains = np.zeros(ntries, _dtype_)
    offsets = np.zeros(ntries,_dtype_)

    for i in range(ntries):
        vmx,vvx = np.random.permutation(pxr,)[:npatches//10].T
        p = np.polyfit(vmx,vvx,1)
        #regressor = linear_model.RANSACRegressor()
        #regressor.fit(vmx[:,None], vvx)
        #re = regressor.estimator_
        gain, intercept = p
        offset = -intercept/gain
        gains[i] = gain
        offsets[i] = offset


    regressorg = linear_model.RANSACRegressor()
    regressorg.fit(vm[:,None],vv)
    gainx = regressorg.estimator_.coef_
    interceptx = regressorg.estimator_.intercept_
    offsetx = -interceptx/gainx
    results = {
        'min': (amin(gains), amin(offsets)),
        'mean': (np.mean(gains), np.mean(offsets)),
        'median': (np.median(gains), np.median(offsets)),
        'ransac': (gainx, offsetx)
    }

    print('RANSAC: Estimated gain %1.2f and offset %1.2f'%(results['ransac']))
    print('ML: Estimated gain %1.2f and offset %1.2f'%(results['mean']))
    print('Med: Estimated gain %1.2f and offset %1.2f'%(results['median']))

    min_gain, min_offset = amin(gains), amin(offsets)

    if with_plot:
        fmt = ' (%1.2f, %1.2f)'

        f,axs = plt.subplots(1,3, figsize=(12,4),gridspec_kw=dict(width_ratios=(2,1,1)))
        h = axs[0].hexbin(vm, vv, bins='log',cmap='viridis',mincnt=5)
        xlow,xhigh = vm.min(), percentile(vm,99)
        ylow,yhigh = vv.min(), percentile(vv,99)
        xfit = np.linspace(vm.min(),xhigh)
        linefit = lambda gain,offset: gain*(xfit-offset)
        axs[0].axis((xlow,xhigh,ylow,yhigh))
        line_fmts = [('--','skyblue'), ('-','g'), ('--','m')]
        hist_kw = dict(density=True, bins=25, color='slategray')
        axs[1].hist(gains,  **hist_kw)
        axs[2].hist(offsets,  **hist_kw)
        plt.setp(axs[1], title='Gains')
        plt.setp(axs[2], title='Offsets')
        for key, lp in zip(('min','mean','ransac'), line_fmts):
            gain,offset = results[key]
            axs[0].plot(xfit, linefit(gain,offset),ls=lp[0],color=lp[1],label=key+': '+fmt%(gain,offset))
            axs[1].axvline(gain,color=lp[1],ls=lp[0])
            axs[2].axvline(offset,color=lp[1],ls=lp[0])

        axs[0].legend(loc='upper left')
        plt.setp(axs[0], xlabel='Mean', ylabel='Variance', title='Mean-Variance for small patches')
        plt.colorbar(h,ax=axs[0])
        if save_to is not None:
            f.savefig(save_to)
    return results[return_type]



def shuffle_signals(m):
    "Given a collection of signals, randomly permute each"
    return array([np.random.permutation(v) for v in m])

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
    v_shuffled = (shuffle_signals(components[:rank]) for i in range(Npermutations))
    coefs_randomized = array([np.abs(data@vt.T).T for vt in v_shuffled])
    coefs_orig = np.abs(data@components[:rank].T).T
    w = zeros((len(data),len(components[:rank])), _dtype_)
    for j in arange(w.shape[1]):
        w[:,j] = coefs_orig[j] >= percentile(coefs_randomized[:,j,:],clip_percentile,axis=0)
    return w

def tsvd_rec_with_weighting(data, rank=None):
    """Do truncated SVD approximation using coefficient pruning by comparisons to shuffled data
    Input: data matrix (Nsamples, Nfeatures), each row is interpreted as a signal or feature vector
    Output: approximated data using rank-truncated SVD
    """
    dc = data.mean(1)[:,None]
    u,s,vh = svd(data-dc,False)
    if rank is None:
        rank = min_ncomp(s, data.shape) + 1
    w = weight_components(data-dc, vh, rank)
    return (u[:,:rank]*w)@diag(s[:rank])@vh[:rank] + dc


import itertools as itt

def make_grid2(shape,sizes,strides):
    """Make a generator over sets of slices which go through the provided shape
       by a stride
    """
    if not np.iterable(sizes):
        sizes = (sizes,)*len(shape)
    if not np.iterable(strides):
        strides = (strides,)*len(shape)

    origins =  itt.product(*[list(range(0,dim-size,stride)) + [dim-size]
                            for (dim,size,stride) in zip(shape,sizes,strides)])
    squares = tuple(tuple(slice(a,a+size) for a,size in zip(o,sizes)) for o in origins)
    return squares

# class LNL_SVD_denoise:
#     def __init__(patch_ssize=10,
#                  patch_tsize=600,
#                  sstride=2,
#                  tstride=300,
#                  min_ncomps=1):
#         return
#
#     def fit_local(self, frames):
#
#         return
#
#     def fit_nonlocal(self):
#         return
#     def inverse_transform(self):
#         return denoised

def patch_tsvds_from_frames(frames,
                            patch_ssize=10, patch_tsize=600,
                            sstride=2, tstride=300,  min_ncomps=1,
                            do_pruning=_do_pruning_,
                            tsmooth=0,ssmooth=0):
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
    #squares =  list(map(tuple, make_grid(d.shape[1:], patch_size,stride)))
    L = len(frames)
    patch_tsize = min(L, patch_tsize)
    if tstride > patch_tsize :
        tstride = patch_tsize//2
    tstride = min(L, tstride)
    squares = make_grid2(frames.shape, (patch_tsize, patch_ssize, patch_ssize), (tstride, sstride, sstride))
    if tsmooth > 0:
        #print('Will smooth temporal components')
        #smoother = lambda v: smoothed_medianf(v, tsmooth*0.5, tsmooth)
        tsmoother = lambda v: adaptive_filter_1d(v, th=3, smooth=tsmooth, keep_clusters=False)
    if ssmooth > 0:
        ssmoother = lambda v: adaptive_filter_2d(v.reshape(patch_ssize,-1),smooth=ssmooth,keep_clusters=False).reshape(v.shape)

    #print('Splitting to patches and doing SVD decompositions',flush=True)
    for sq in tqdm(squares,desc='Splitting to patches and doing SVD'):

        patch_frames = d[sq]
        L = len(patch_frames)
        w_sh = patch_frames.shape

        #print(sq, w_sh, L)

        patch = patch_frames.reshape(L,-1) # now each column is signal in one pixel
        patch_c = np.mean(patch,0)
        patch = patch - patch_c

        u,s,vh = np.linalg.svd(patch.T,full_matrices=False)
        #rank = min_ncomp(s, patch.shape)+1
        rank = max(min_ncomps, min_ncomp(s, patch.shape)+1)
        u,vh = svd_flip_signs(u[:,:rank],vh[:rank])

        w = weight_components(patch.T, vh, rank) if do_pruning else np.ones(u[:,:rank].shape)
        svd_signals,loadings = vh[:rank], u[:,:rank]*w
        s = s[:rank]
        svd_signals = svd_signals*s[:,None]

        if tsmooth > 0:
            svd_signals = array([tsmoother(v) for v in svd_signals])
        #W = np.diag(s)@vh
        W = loadings.T
        if  (ssmooth > 0) and (patch.shape[1] == patch_ssize**2):
            W = array([ssmoother(v) for v in W])
        #print (svd_signals.shape, W.shape, patch.shape)
        #return
        acc.append((svd_signals, W, patch_c, sq, w_sh))
    return acc

def tanh_step(x, window):
    taper_width=window/5
    taper_k = taper_width/4
    return np.clip((1.01 + np.tanh((x-taper_width)/taper_k) * np.tanh((window-x-taper_width)/taper_k))/2,0,1)

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
    out_data = np.zeros(shape,dtype=_dtype_)
    if with_f0:
        out_f0 = np.zeros_like(out_data)
    #counts = np.zeros(shape[1:], np.int)
    counts = np.zeros(shape,_dtype_) # candidate for crossfade

    #tslice = (slice(None),)
    i = 0
    #print('Doing inverse transform', flush=True)
    tqdm_desc = 'Doing inverse transform ' +  ('with baseline' if with_f0 else '')
    for signals,filters,center,sq, w_sh in tqdm(collection, desc=tqdm_desc):
        L = w_sh[0]
        crossfade_coefs = tanh_step(arange(L), L).astype(_dtype_)[:,None,None]
        #crossfade_coefs = np.ones(L)[:,None,None]
        counts[sq] += crossfade_coefs

        rec = (signals.T@filters).reshape(w_sh)
        out_data[tuple(sq)] += (rec + center.reshape(w_sh[1:]))*crossfade_coefs

        if with_f0:
            bs = np.array([simple_baseline(v,plow=50,smooth=baseline_smoothness,ns=mad_std(v)) for v in signals])
            if any(isnan(bs)):
                print('Nan in ', sq)
                #return (signals, filters, center,sq,w_sh)
            rec_b = (bs.T@filters).reshape(w_sh)
            out_f0[tuple(sq)] += (rec_b + center.reshape(w_sh[1:]))*crossfade_coefs

    out_data /= (1e-12 + counts)
    out_data *= (counts > 1e-12)
    if with_f0:
        out_f0 /= (1e-12 + counts)
        out_f0 *= (counts > 1e-12)
        return out_data, out_f0
    return out_data

# TODO:
# - [ ] Exponential-family PCA
# - [ ] Cut svd_signals into pieces before second-stage SVD
#       - alternatively, look at neighboring patches in time
# - [X] (***) Cluster svd_signals before SVD

def patch_center(p):
    "center location of an n-dimensional slice"
    return array([0.5*(p_.start+p_.stop) for p_ in p])

def _pairwise_euclidean_distances(points):
    """pairwise euclidean distances between points.
    Calculated as distance between vectors x and y:
    d = sqrt(dot(x,x) -2*dot(x,y) + dot(Y,Y))
    """
    X = np.asarray(points)
    XX = np.sum(X*X, axis=1)[:,np.newaxis]
    D = -2 * np.dot(X,X.T) + XX + XX.T
    np.maximum(D, 0, D)
    # todo triangular matrix, sparse matrix
    return np.sqrt(D)


from imfun.cluster import som
# TODO: decide which clustering algorithm to use.
#       candidates:
#         - KMeans (sklearn)
#         - MiniBatchKMeans (sklearn)
#         - AgglomerativeClustering with Ward or other linkage (sklearn)
#         - SOM aka Kohonen (imfun)
#         - something else?
#       - clustering algorithm should be made a parameter
# TODO: good crossfade and smaller overlap

def second_stage_svd(collection, fsh,  n_clusters=_nclusters_, Nhood=100, clustering_algorithm='AgglomerativeWard'):
    out_signals = [zeros(c[0].shape,_dtype_) for c in collection]
    out_counts = zeros(len(collection), np.int) # make crossafade here
    squares = make_grid2(fsh[1:], Nhood, Nhood//2)
    tstarts = set(c[3][0].start for c in collection)
    tsquares = [(t, sq) for t in tstarts for sq in squares]
    clustering_dispatcher = {
        'AgglomerativeWard'.lower(): lambda nclust : skcluster.AgglomerativeClustering(nclust),
        'KMeans'.lower() : lambda nclust: skcluster.KMeans(nclust),
        'MiniBatchKMeans'.lower(): lambda nclust: skcluster.MiniBatchKMeans(nclust)
    }
    def _is_local_patch(p, sqx):
        t0, sq = sqx
        tstart = p[0].start
        psq = p[1:]
        return (tstart==t0) & (slice_overlaps_square(psq, sq))
    for sqx in tqdm(tsquares, desc='Going through larger squares'):
        #print(sqx, collection[0][3], slice_starts_in_square(collection[0][3], sqx))
        signals = [c[0] for c in collection if _is_local_patch(c[3], sqx)]
        if not(len(signals)):
            print(sqx)
            for c in collection:
                print(sqx, c[3], _is_local_patch(c[3], sqx))
        nclust = min(n_clusters, len(signals))
        signals = vstack(signals)
        #clust=skcluster.KMeans(min(n_clusters,len(signals)))
        #clust = skcluster.AgglomerativeClustering(min(n_clusters,len(signals)))
        # number of signals can be different in some patches due to boundary conditions
        clust = clustering_dispatcher[clustering_algorithm.lower()](nclust)
        if clustering_algorithm == "MiniBatchKMeans".lower():
            clust.batch_size  = min(clust.batch_size, len(signals))
        labels = clust.fit_predict(signals)
        sqx_approx = np.zeros(signals.shape, _dtype_)
        for i in unique(labels):
            group = labels==i
            u,s,vh = svd(signals[group],False)
            r = min_ncomp(s, (u.shape[0],vh.shape[1]))+1
            #w = weight_components(all_svd_signals[group],vh,r)
            approx = u[:,:r]@diag(s[:r])@vh[:r]
            sqx_approx[group] = approx
        kstart=0
        for i,c in enumerate(collection):
            if _is_local_patch(c[3], sqx):
                l =len(c[0])
                out_signals[i] += sqx_approx[kstart:kstart+l]
                out_counts[i] += 1
                kstart+=l
    return [(x/(1e-7+cnt),) +  c[1:] for c,x,cnt in zip(collection, out_signals, out_counts)]


import pickle
def patch_svd_denoise_frames(frames, do_second_stage=False, save_coll=None,
                             tsvd_kw=None, second_stage_kw=None,inverse_kw=None):
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
        coll = second_stage_svd(coll, frames.shape,**second_stage_kw)
    if save_coll is not None:
        with gzip.open(save_coll, 'wb') as fh:
            pickle.dump((coll,frames.shape), fh)
    return project_from_tsvd_patches(coll, frames.shape, **inverse_kw)



# TODO: may be scrambled data are the best for robust gain,offset estimates???

def scramble_data(frames):
    """Randomly permute (shuffle) signals in each pixel independenly
    useful for quick creation of surrogate data with zero acitvity, only noise
    - TODO: clip some too high values
    """
    L,nr,nc = frames.shape
    out = np.zeros_like(frames)
    for r in range(nr):
        for c in range(nc):
            out[:,r,c] = np.random.permutation(frames[:,r,c])
    return out

def scramble_data_local_jitter(frames,w=10):
    """Randomly permute (shuffle) signals in each pixel independenly
    useful for quick creation of surrogate data with zero acitvity, only noise
    - TODO: clip some too high values
    """
    L,nr,nc = frames.shape
    out = np.zeros_like(frames)
    for r in range(nr):
        for c in range(nc):
            out[:,r,c] = local_jitter(frames[:,r,c], w)
    return out

from matplotlib import animation
from skimage.feature import peak_local_max
from scipy import ndimage

def make_denoising_animation(frames, yhat,f0,  movie_name, start_loc=None,path=None):
    figh = plt.figure(figsize=(10,10))
    axleft = plt.subplot2grid((2,2), (0,0))
    axright = plt.subplot2grid((2,2), (0,1))

    L = len(frames)

    for ax in (axleft,axright):
        plt.setp(ax,xticks=[],yticks=[])
    axbottom = plt.subplot2grid((2,2), (1,0),colspan=2)
    fsh = frames[0].shape


    if (start_loc is None) :
        f = ndimage.gaussian_filter(np.max(yhat-f0,0),3)
        k = argmax(ravel(f))
        nrows,ncols = frames[0].shape
        loc = (k//ncols, k%ncols )
    else:
        loc = start_loc

    axleft.set_title('Raw (x10 speed)',size=14)
    axright.set_title('Denoised (x10 speed)',size=14)
    axbottom.set_title('Signals at point',size=14)
    low,high = np.percentile(yhat, (0.5, 99.5))
    low,high = 0.9*low,1.1*high
    h1 = axleft.imshow(frames[0],clim=(low,high),cmap='gray',animated=True)
    h2 = axright.imshow(yhat[0],clim=(low,high),cmap='gray',animated=True)
    axbottom.set_ylim(yhat.min(), yhat.max())
    plt.tight_layout()

    lhc = axright.axvline(loc[1], color='y',lw=1)
    lhr = axright.axhline(loc[0], color='y',lw=1)
    lhb = axbottom.axvline(0,color='y',lw=1)
    if path is None:
        locs = (loc + np.cumsum([(0,0)] + [np.random.randint(-1,2,size=2)*0.5+(1.25,1.25)
                                           for i in range(L)],axis=0)).astype(int)
    else:
        locs = []
        current_loc = array(start_loc)
        apath = asarray(path)
        keypoints = apath[:,2]
        for kf in range(L):
            ktarget = argmax(keypoints>=kf)
            target = apath[ktarget,:2][::-1] # switch from xy to rc
            kft = keypoints[ktarget]
            #print(target, current_loc)
            if kft == kf:
                v = 0
            else:
                v = (target-current_loc)#/(ktarget-kf)
                #vl = norm(v)
                v = v/(kft-kf)

            current_loc = current_loc + v
            locs.append(current_loc.copy())

    loc = locs[0].astype(int)
    xsl = (slice(None), loc[0], loc[1])
    lraw = axbottom.plot(frames[xsl], color='gray',label='Fraw')[0]#,animated=True)
    ly = axbottom.plot(yhat[xsl], color='royalblue',label=r'$\hat F$')[0]#,animated=True)
    lb = axbottom.plot(f0[xsl], color=(0.2,0.8,0.5),lw=2,label='F0')[0]#,animated=True)

    axbottom.legend(loc='upper right')
    nrows,ncols = fsh

    nn = np.concatenate([np.diag(ones(2,np.int)),-np.diag(ones(2,np.int))])

    loc = [loc]
    def _animate(frame_ind):
        #loc += randint(-1,2,size=2)

        #loc = locs[frame_ind]

        #f = ndimage.gaussian_filter(yhat[frame_ind]/f0[frame_ind],3)
        #lmx = peak_local_max(f)
        #labels, nlab = ndi.label(lmx)
        #objs = ndi.find_objects(labels)
        #peak = argmax([f[o].mean() for o in objs])
        #loc =  [(oi.start+oi.stop)/2 for oi in objs[peak]]

        #f = ndimage.gaussian_filter(yhat[frame_ind],3)
        #k = argmax([f[n[0]] for n in loc[0]+nn])
        #k = argmax(ravel(f))
        #loc[0] = (k//ncols, k%ncols )
        #loc[0] = loc[0] + nn[k]
        loc[0] = locs[frame_ind]
        loc[0] = asarray((loc[0][0]%nrows,loc[0][1]%ncols),np.int)
        xsl = (slice(None), loc[0][0], loc[0][1])
        h1.set_data(frames[frame_ind])
        h2.set_data(yhat[frame_ind])
        lraw.set_ydata(frames[xsl])
        ly.set_ydata(yhat[xsl])
        lb.set_ydata(f0[xsl])
        lhc.set_xdata(loc[0][1])
        lhr.set_ydata(loc[0][0])
        lhb.set_xdata(frame_ind)
        return [h1,h2,lraw,ly,lb,lhc,lhr]
    anim = animation.FuncAnimation(figh, _animate, frames=int(L), blit=True)
    Writer = animation.writers.avail['ffmpeg']
    w = Writer(fps=10,codec='libx264',bitrate=16000)
    anim.save(movie_name)
    return locs
