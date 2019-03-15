"""
μCats -- a set of routines for detection and analysis of Ca-transients
"""

import os,sys
from numba import jit

from functools import partial
import itertools as itt

import matplotlib.pyplot as plt


import numpy as np


from numpy import pi
from numpy import linalg
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
from imfun.multiscale import mvm

from imfun import components


_dtype_ = np.float32


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
    
    tslice = (slice(None),)
    for sq in squares:
        patch = d[tslice+sq]
        sh = patch.shape
        wclip = w[:sh[1],:sh[2]]
        #print(w.shape, sh[1:3], wclip.shape)
        #wclip /= sum(wclip)
        signal = (patch*wclip).sum(axis=(1,2))
        acc.append((signal, sq, wclip.reshape(1,-1)))
    return acc
    #signals =  array([d[(slice(None),)+s].sum(-1).sum(-1)/prod(d[0][s].shape) for s in squares])
    #return [(v,sq,w) for v,sq in zip(signals, squares)]

def weight_counts(collection,sh):
    counts = np.zeros(sh)
    for v,s,w in collection:
        wx = w.reshape(counts[tuple(s)].shape)
        counts[s] += wx
    return counts


def signals_from_array_pca_cluster(data,stride=2, nhood=3, ncomp=2,
                                   pre_smooth=3,
                                   dbscan_eps_p=5, dbscan_minpts=3, cluster_minsize=5,
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
        if pre_smooth > 1:
            patch = ndi.median_filter(patch, size=(pre_smooth,1))
        Xc = patch.mean(0)
        u,s,vh = np.linalg.svd(patch-Xc,full_matrices=False)
        points = u[:,:ncomp]
        #dists = cluster.metrics.euclidean(points[kcenter],points)
        all_dists = cluster.dbscan_._pairwise_euclidean_distances(points)
        dists = all_dists[kcenter]

        #np.mean(dists)
        dbscan_eps = np.percentile(all_dists[all_dists>1e-6], dbscan_eps_p)
        #dbscan_eps_acc.append(dbscan_eps)
        #print(r,c,':', dbscan_eps)
        _,_,affs = cluster.dbscan(points, dbscan_eps, dbscan_minpts, distances=all_dists)
        similar = affs==affs[kcenter]

        if sum(similar) < cluster_minsize or affs[kcenter]==-1:
            knn_count[0] += 1
            th = min(np.argsort(dists)[cluster_minsize],2*dbscan_eps)
            similar = dists <= max(th, dists[kcenter])
            #print('knn similar:', np.sum(similar))
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
        vx = patch[similar].mean(0) # DONE?: weighted aggregate
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
                       threshold_time_signals=False,
                       smooth_baseline=False,
                       keep_baseline=False,
                       mask_of_interest=None):
    sh = data.shape
    L = sh[0]
    
    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh,_dtype_)
    out_b = np.zeros(sh,_dtype_)
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
            sys.stderr.write(' svd rank: %02d'% rank)
        ux = ndi.median_filter(u[:,:rank],size=(temporal_filter,1))
        vh_images = vh[:rank].reshape(-1,*w_sh[1:])
        vhx = [ndi.median_filter(f, size=(spatial_filter,spatial_filter)) for f in vh_images]
        vhx_threshs = [mad_std(f) for f in vh_images]
        vhx = np.array([np.where(f>th,fx,f) for f,fx,th in zip(vh_images,vhx,vhx_threshs)])
        vhx = vhx.reshape(rank,len(vh[0]))

        # 13.03.19 -- доделать сохранение "базовой линии" на более свежую голову
        if threshold_time_signals or keep_baseline or smooth_baseline:
            svd_signals = ux.T
            if smooth_baseline:
                biases = np.array([simple_baseline(v,50,smooth=50) for v in svd_signals])
            else:
                biases = np.array([find_bias(v,ns=mad_std(v)) for v in svd_signals]).reshape(-1,1)

            svd_signals_c = svd_signals - biases

            signals_fplus = np.array([v*percentile_label(v,percentile_low=25,tau=1.5) for v in svd_signals_c])
            signals_fminus = np.array([v*percentile_label(-v,percentile_low=25) for v in svd_signals_c])
            signals_filtered = signals_fplus + signals_fminus
            if keep_baseline:
                signals_filtered +=  biases
                
            ux = signals_filtered.T
            
        
        #print('\n', patch.shape, u.shape, vh.shape)
        #ux = u[:,:rank]
        proj = ux@np.diag(s[:rank])@vhx[:rank]
        score = np.sum(s[:rank]**2)/np.sum(s**2)
        #score = 1
        rec  = proj.reshape(w_sh)
        if keep_baseline:
            # we possibly shift the baseline level due to thresholding of components
            rec += find_bias_frames(data[tsl]-rec,3,mad_std(data[tsl],0)) 
        out[tsl] += score*rec
        counts[sl] += score
        
    for r in itt.chain(range(nhood,sh[1]-nhood,stride), [sh[1]-nhood]):
        for c in itt.chain(range(nhood,sh[2]-nhood,stride), [sh[2]-nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c,npc)
    out = out/(1e-12+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out[:,r,c] = 0
    return out

def block_svd_denoise_and_separate(data, stride=2, nhood=5,
                                   ncomp=None,
                                   spatial_filter=1,
                                   baseline_smoothness=100,
                                   baseline_post_smooth=10,
                                   mask_of_interest=None):
    sh = data.shape
    L = sh[0]
    
    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out_signals = np.zeros(sh,_dtype_)
    out_baselines = np.zeros(sh,_dtype_)
    counts = np.zeros(sh[1:],_dtype_)
    if mask_of_interest is None:
        mask=np.ones(counts.shape,bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood+1)**2
    
    def _process_loc(r,c):
        sl = (slice(r-nhood,r+nhood+1), slice(c-nhood,c+nhood+1))
        tsl = (slice(None),)+sl

        patch_frames = data[tsl]
        w_sh = patch_frames.shape

        patch = patch_frames.reshape(sh[0],-1)

        if not(np.any(patch)):
            out_signals[tsl] += 0
            out_baselines[tsl] += 0
            counts[sl] += 0
            return
        # (patch is now Nframes x Npixels, u will hold temporal components)
        u,s,vh = np.linalg.svd(patch,full_matrices=False)
        if ncomp is None:
            rank = min_ncomp(s, patch.shape)+1
            sys.stderr.write(' svd rank: %02d'% rank)
        else:
            rank = ncomp

        ux = u[:,:rank]
        
        vh_images = vh[:rank].reshape(-1,*w_sh[1:])
        vhx = [ndi.median_filter(f, size=(spatial_filter,spatial_filter)) for f in vh_images]
        vhx_threshs = [mad_std(f) for f in vh_images]
        vhx = np.array([np.where(f>th,fx,f) for f,fx,th in zip(vh_images,vhx,vhx_threshs)])
        vhx = vhx.reshape(rank,len(vh[0]))

        svd_signals = ux.T
        if baseline_smoothness:
            biases = np.array([simple_baseline(v,50,smooth=baseline_smoothness,ns=mad_std(v)) for v in svd_signals])
            #biases = np.array([smoothed_medianf(v, smooth=5, wmedian=int(baseline_smoothness)) for v in svd_signals])
        else:
            biases = np.array([find_bias(v,ns=mad_std(v)) for v in svd_signals]).reshape(-1,1)
            biases = np.zeros_like(svd_signals)+biases

        svd_signals_c = svd_signals - biases

        signals_fplus = np.array([v*percentile_label(v,percentile_low=25,tau=1.5) for v in svd_signals_c])
        signals_fminus = np.array([v*percentile_label(-v,percentile_low=25) for v in svd_signals_c])
        signals_filtered = signals_fplus + signals_fminus
            
        ux = signals_filtered.T
        ux_biases = biases.T
        
        #print('\n', patch.shape, u.shape, vh.shape)
        #ux = u[:,:rank]
        signals = ux@np.diag(s[:rank])@vhx[:rank]
        baselines = ux_biases@np.diag(s[:rank])@vhx[:rank]
        
        score = np.sum(s[:rank]**2)/np.sum(s**2)
        #score = 1
        rec  = signals.reshape(w_sh)
        rec_baselines = baselines.reshape(w_sh)
        # we possibly shift the baseline level due to thresholding of components
        ##rec += find_bias_frames(data[tsl]-rec,3,mad_std(data[tsl],0))
        rec_baselines += find_bias_frames(data[tsl]-rec-rec_baselines,3,mad_std(data[tsl],0))
        out_signals[tsl] += score*rec
        out_baselines[tsl] += score*rec_baselines
        counts[sl] += score
        
    for r in itt.chain(range(nhood,sh[1]-nhood,stride), [sh[1]-nhood]):
        for c in itt.chain(range(nhood,sh[2]-nhood,stride), [sh[2]-nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d'%(r,c, r*sh[1] + c+1, np.prod(sh[1:])))
            if mask[r,c]:
                _process_loc(r,c)
    out_signals = out_signals/(1e-6+counts[None,:,:])
    out_baselines = out_baselines/(1e-6+counts[None,:,:])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r,c] == 0:
                out_signals[:,r,c] = 0
                out_baselines[:,r,c] = 0
    if baseline_post_smooth > 0:
        out_baselines = ndi.gaussian_filter(out_baselines, (baseline_post_smooth, 0, 0))
    return out_signals,out_baselines



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
    points = []
    for loc in locations(mask.shape):
        if mask[loc]:
            points.append(loc) 
    return points

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
        score = np.sum(s0[:rank]**2)/np.sum(s0**2)*np.ones(len(signals))
        
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

        score = score.reshape(w_sh[1:])
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

from sklearn import decomposition as skd
from skimage import filters as skf
def _patch_denoise_nmf(data,stride=2, nhood=5, ncomp=None,
                       smooth_baseline=False,
                       max_ncomp=None,
                       temporal_filter = None,
                       mask_of_interest=None):
    sh = data.shape
    L = sh[0]
    if max_ncomp is None:
        max_ncomp = 0.25*(2*nhood+1)**2
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
        lift = patch.min(0)

        patch = patch-lift # precotion against negative values in data
        
        X = patch.reshape(L,-1)
        u,s,vh = svd(X,False)
        rank = min(max_ncomp, min_ncomp(s,X.shape) + 1) if ncomp is None else ncomp
        if ncomp is None:
            sys.stderr.write('  rank: %d  '%rank)

        d = skd.NMF(rank,l1_ratio=0.95,init='nndsvdar')#,beta_loss='kullback-leibler',solver='mu')
        nmf_signals = d.fit_transform(X).T
        nmf_comps = np.array([m*opening_of_closing(m > 0.5*skf.threshold_otsu(m)) for m in d.components_])


        #nmf_biases = np.array([find_bias(v) for v in nmf_signals]).reshape(-1,1)
        #nmf_biases = np.array([multi_scale_simple_baseline(v) for v in nmf_signals])
        if smooth_baseline:
            nmf_biases = np.array([simple_baseline(v,50,smooth=50) for v in nmf_signals])
        else:
            nmf_biases = np.array([find_bias(v,ns=mad_std(v)) for v in nmf_signals]).reshape(-1,1)
        nmf_signals_c = nmf_signals - nmf_biases

        nmf_signals_fplus = np.array([v*percentile_label(v,percentile_low=25,tau=1.5) for v in nmf_signals_c])
        nmf_signals_fminus = np.array([v*percentile_label(-v,percentile_low=25) for v in nmf_signals_c])
        nmf_signals_filtered = nmf_signals_fplus + nmf_signals_fminus + nmf_biases

        rec = nmf_signals_filtered.T@nmf_comps
        rec_frames = rec.reshape(*patch.shape)
        rec_frames += find_bias_frames(patch-rec_frames,3,mad_std(patch,0)) # we possibly shift the baseline level due to thresholding of components

        #print(out[tsl].shape, patch.shape, rec.shape)
        out[tsl] += rec_frames + lift
        
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
    return np.where(nsm>1e-5,(frames-biases)/nsm,0)


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

def percentile_label(v, percentile_low=2.5,tau=2.0,smoother=l2spline):
    mu = min(np.median(v),0)
    low = np.percentile(v[v<=mu], percentile_low)
    vs = smoother(v, tau)
    return vs >= -low
    


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
    return b + np.median(d[d<th*ns]) # + bias as constant shift


def smoothed_medianf(v,smooth=10,wmedian=10):
    "Robust smoothing by first applying median filter and then applying L2-spline filter" 
    return l2spline(ndi.median_filter(v, wmedian),smooth)

def simple_baseline(y, plow=25, th=3, smooth=25,ns=None):
    b = l2spline(ndi.percentile_filter(y,plow,smooth),smooth/5)
    if ns is None:
        ns = rolling_sd_pd(y)
    d = y-b
    b2 = b + np.median(d[d<th*ns]) # correct scalar shift 
    return b2


def find_bias(y, th=3, ns=None):
    if ns is None:
        ns = rolling_sd_pd(y)
    return np.median(y[y<=np.median(y)+th*ns])


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
    

def std_median(v):
    N = float(len(v))
    md = np.median(v)
    return (np.sum((v-md)**2)/N)**0.5

def mad_std(v,axis=None):
    mad = np.median(abs(v-np.median(v,axis=axis)),axis=axis)
    return mad*1.4826

def closing_of_opening(m,s=None):
    return ndi.binary_closing(ndi.binary_opening(m,s),s)

def adaptive_median_filter(frames,th=5, tsmooth=1,ssmooth=5, keep_clusters=True):
    medfilt = ndi.median_filter(frames, (tsmooth,ssmooth,ssmooth))
    details = frames - medfilt
    #mdmap = np.median(details, axis=0)
    #sdmap = np.median(abs(details - mdmap), axis=0)*1.4826
    sdmap = mad_std(frames,axis=0)
    outliers = np.abs(details) > th*sdmap
    #s = np.zeros((3,3,3)); s[:,1,1] = 1
    s = np.array([[[0,0,0],[0,1,0],[0,0,0]]]*3)    
    #outliers[ndi.binary_closing(ndi.binary_opening(outliers,s),s)]=False
    if keep_clusters:
        outliers ^= closing_of_opening(outliers)
    return np.where(outliers, medfilt, frames)

def adaptive_median_filter_2d(img,th=5, smooth=5):
    medfilt = ndi.median_filter(img, smooth)
    details = img - medfilt
    md = np.median(details)
    sd = np.median(abs(details - md))*1.4826
    #sdmap = ucats.mad_std(frames,axis=0)
    outliers = np.abs(details-md) > th*sd
    #s = np.zeros((3,3,3)); 
    #s[:,1,1] = 1
    #s = np.array([[[0,0,0],[0,1,0],[0,0,0]]]*3)    
    #outliers[ndi.binary_closing(ndi.binary_opening(outliers,s),s)]=False
    outliers ^= closing_of_opening(outliers)
    return np.where(outliers, medfilt, img)


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
    
    rolling_median = lambda x: ndi.median_filter(x, 2*hw)
    
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
            rsd_smoother = lambda v_: ndi.median_filter(y,7)
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
        rsd_smoother = lambda v_: ndi.median_filter(y,7)
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
    return where(b>0,rec,0)

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

def svd_flip_signs(u,vh,medianw=None):
    L = len(u)
    if medianw is None:
        medianw = L//5
    for i,c in enumerate(u.T):
        sk = skew(c-ndi.median_filter(c,medianw))
        sg = np.sign(sk)
        u[:,i] *= sg
        vh[i] *= sg
    return u,vh


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
    base1 = calculate_baseline_pca(frames,smooth=smooth,npc=npc)
    base2 = calculate_baseline(frames-base1, pipeline=baseline_fn, pipeline_kw=baseline_kw)
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
def roticity_fft(data,period_low = 100, period_high=5,npc=6):
    """
    Look for local areas with oscillatory dynamics in TXY framestack
    """
    L = len(data)
    if ndim(data)>2:
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
        lm = np.array(extrema.locextr(p[:,i],x=nu,refine=True,output='max'))
        lm = lm[(lm[:,0]>1/period_low)*(lm[:,0]<1/period_high)]
        peak_ = np.amax(lm[:,1])/p[:,i][~nu_phys].mean()*s2[i]
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
    coll_enh = process_signals_parallel(coll,pipeline=pipeline, pipeline_kw=pipeline_kw)
    print('Time-signals processed, recombining to video...')
    out = combine_weighted_signals(coll_enh,frames.shape)
    fsx = fseq.from_array(out)
    print('Done')
    fsx.meta['channel']='-'.join(['newrec4',kind])
    return fsx

def svd_denoise_tslices(frames, twindow=50,
                        nhood=5,
                        npc=5,
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
        
    return out_s/counts[:,None,None],out_b/counts[:,None,None]


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
    return data[:,lims[1]:-lims[1],lims[0]:-lims[0]]




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
    fs_f0 = get_baseline_frames(frames[:],baseline_fn=baseline_fn, baseline_kw=baseline_kw)
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
    def __init__(self, frames, threshold=0.025,min_duration=3,min_area=9,peak_threshold=0.05):
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
