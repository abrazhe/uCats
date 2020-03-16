import numpy as np
from numpy import linalg

from sklearn import cluster as skcluster

import gzip
import pickle

from umap import UMAP


from tqdm.auto import tqdm

from imfun import components


from ..decomposition import (min_ncomp,
                             SVD_patch,
                             patch_tsvd_transform,
                             patch_tsvd_inverse_transform)

from ..patches import make_grid, slice_overlaps_square

from ..globals import _dtype_
from ..utils import mad_std


_nclusters_ = 32

# TODO:
# - [ ] Exponential-family PCA
# - [ ] Cut svd_signals into pieces before second-stage SVD
#       - alternatively, look at neighboring patches in time
# - [X] (***) Cluster svd_signals before SVD

from skimage.restoration import denoise_tv_chambolle

import sklearn.cluster as skcluster

# TODO: -- move the clustering code somewhere in ucats or imfun
from numba import jit
@jit
def calc_coincidents(labels):
    Npts = len(labels)
    out = np.zeros((Npts,Npts))
    for i in range(Npts):
        for j in range(i,Npts):
            out[i,j] = labels[i] == labels[j]
    return out

class DumbConsensusClusterer:

    def __init__(self, n_clusters, merge_threshold=0.85, n_clusterers=25, min_cluster_size=10, min_overlap=5):
        self.n_clusters = n_clusters
        self.merge_threshold = merge_threshold
        self.n_clusterers = n_clusterers
        self.min_cluster_size=min_cluster_size
        self.min_overlap = min_overlap

    def fit_predict(self, X):
        lbls = []
        for i in range(self.n_clusterers):
            clusterer = skcluster.MiniBatchKMeans(n_clusters=self.n_clusters*2)
            l = clusterer.fit_predict(X)
            lbls.append(l)
        lbls = np.array(lbls)
        return self.cluster_ensembles(lbls)

    def calc_consensus_matrix(self, labels):
        Npts = len(labels[0])
        out = np.zeros((Npts,Npts))
        for lab in labels:
            out += calc_coincidents(lab)
        return out/len(labels)

    def cluster_ensembles(self, labels,):
        clusters = []
        Cm = self.calc_consensus_matrix(labels)
        Npts =  len(Cm)
        for i in range(Npts):
            row = Cm[i]
            candidates = set([j for j in range(i,Npts) if (row[j]>=self.merge_threshold)])
            if not len(candidates):
                continue
            if not len(clusters):
                clusters.append(set(candidates))
            else:
                overlaps = [c for c in clusters if len(c.intersection(candidates)) >= self.min_overlap]
                non_overlapping = [c for c in clusters if not c in overlaps]
                if len(overlaps):
                    for c in overlaps:
                        candidates.update(c)
                    clusters = [candidates] + non_overlapping
                else:
                    clusters.append(candidates)

        clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        out_labels = np.zeros(Npts)
        for k,cx in enumerate(clusters):
            for i in cx:
                out_labels[i] = k + 1
        return out_labels


import hdbscan

class UMAP_Preprocessed:
    def __init__(self, *args, **kwargs):
        self.preprocessor = UMAP(n_neighbors=30, min_dist=0, n_components=2)
        #self.clusterer = skcluster.DBSCAN(**kwargs) # this is not *H*dbscan, is it?
        #self.clusterer = hdbscan.HDBSCAN()
    def fit_predict(self, X):
        X = self.preprocessor.fit_transform(X)
        return self.clusterer.fit_predict(X)

# this can be done better ..., e.g. using decorators for
# fit_predict methods for existing  clustering objects
class UMAP_MiniBatchKMeans(UMAP_Preprocessed):
    def __init__(self,nclust):
        super(UMAP_MiniBatchKMeans, self).__init__()
        self.clusterer = skcluster.MiniBatchKMeans(nclust)

class UMAP_KMeans(UMAP_Preprocessed):
    def __init__(self, nclust):
        super(UMAP_KMeans, self).__init__()
        self.clusterer = skcluster.KMeans(nclust)


class UMAP_HDBSCAN(UMAP_Preprocessed):
    def __init__(self, *args, **kwargs):
        super(UMAP_HDBSCAN, self).__init__()
        self.clusterer = hdbscan.HDBSCAN()


def separable_iterated_tv_chambolle(im, sigma_x=1, sigma_y=1, niters=5):
    if (sigma_x <= 0) and (sigma_y <= 0):
        return im
    im_w = np.copy(im)

    for i in range(niters):
        if sigma_y > 0 :
            im_w = np.array([denoise_tv_chambolle(cv, mad_std(cv)*sigma_y) for cv in im_w.T]).T # columns
        if sigma_x > 0:
            im_w = np.array([denoise_tv_chambolle(rv, mad_std(rv)*sigma_x) for rv in im_w])     # rows
    return im_w

# TODO: decide which clustering algorithm to use.
#       candidates:
#         - KMeans (sklearn)
#         - MiniBatchKMeans (sklearn)
#         - AgglomerativeClustering with Ward or other linkage (sklearn)
#         - SOM aka Kohonen (imfun)
#         - something else?
#       - clustering algorithm should be made a parameter
# TODO: good crossfade and smaller overlap

def second_stage_svd(collection,
                     fsh,
                     n_clusters=_nclusters_,
                     Nhood=100,
                     clustering_algorithm='MiniBatchKmeans',
                     mode='cluster',
                     **kwargs):
    out_signals = [np.zeros(c.signals.shape,_dtype_) for c in collection]
    out_counts = np.zeros(len(collection), np.int) # make crossafade here
    squares = make_grid(fsh[1:], Nhood, Nhood//2)
    tstarts = set(c.sq[0].start for c in collection)
    tsquares = [(t, sq) for t in tstarts for sq in squares]
    clustering_dispatcher = {
        'AgglomerativeWard'.lower(): lambda nclust : skcluster.AgglomerativeClustering(nclust),
        'KMeans'.lower() : lambda nclust: skcluster.KMeans(nclust),
        'MiniBatchKMeans'.lower(): lambda nclust: skcluster.MiniBatchKMeans(nclust),
        'UMAP_DBSCAN'.lower(): lambda nclust: UMAP_HDBSCAN(min_samples=15),
        'UMAP_KMeans'.lower(): lambda nclust: UMAP_KMeans(nclust),
        'UMAP_MiniBatchKMeans'.lower(): lambda nclust: UMAP_MiniBatchKMeans(nclust),
        'ConsensusKMeans'.lower(): lambda nclust: DumbConsensusClusterer(nclust)
    }
    def _is_local_patch(p, sqx):
        t0, sq = sqx
        tstart = p[0].start
        psq = p[1:]
        return (tstart==t0) & (slice_overlaps_square(psq, sq))

    for sqx in tqdm(tsquares, desc='Going through larger squares'):
        signals = [c.signals for c in collection if _is_local_patch(c.sq, sqx)]
        if not(len(signals)):
            print(sqx)
            for c in collection:
                print(sqx, c.sq, _is_local_patch(c.sq, sqx))

        nclust = min(n_clusters, len(signals))

        signals = np.vstack(signals)

        if mode.lower() == 'flatTV'.lower():
            mapper1d = UMAP(n_components=1, n_neighbors=30,min_dist=0,metric='euclidean')
            X1d = mapper1d.fit_transform(signals)[:,0]
            ksort = np.argsort(X1d)
            img = signals[ksort]
            ns = mad_std(img) # questionable
            # only do along Y axis (between signals, not along time)
            # rather slow though
            # niters should be a parameter
            img2 = separable_iterated_tv_chambolle(img.T, sigma_x=2, sigma_y=0, niters=5).T
            sqx_approx = np.zeros(signals.shape, _dtype_)
            for i,k in enumerate(ksort):
                sqx_approx[k] = img2[i]
        elif mode.lower() == 'cluster':
            # number of signals can be different in some patches due to boundary conditions
            clust = clustering_dispatcher[clustering_algorithm.lower()](nclust)
            if clustering_algorithm == "MiniBatchKMeans".lower():
                clust.batch_size  = min(clust.batch_size, len(signals))

            labels = clust.fit_predict(signals)

            sqx_approx = np.zeros(signals.shape, _dtype_)
            for i in np.unique(labels):
                group = labels==i
                u,s,vh = linalg.svd(signals[group],False)
                r = min_ncomp(s, (u.shape[0],vh.shape[1]))+1
                u = u[:,:r]
                #w = weight_components(all_svd_signals[group],vh,r)
                approx = u@np.diag(s[:r])@vh[:r]
                sqx_approx[group] = approx
        else:
            raise ValueError(f"Unknown mode {mode}, use 'flatTV' or 'cluster'")
        kstart=0
        for i,c in enumerate(collection):
            if _is_local_patch(c.sq, sqx):
                l =len(c.signals)
                out_signals[i] += sqx_approx[kstart:kstart+l]
                out_counts[i] += 1
                kstart+=l
    return [SVD_patch(x/(1e-7+cnt), *c[1:])
            for c,x,cnt in zip(collection, out_signals, out_counts)]

# def second_stage_svd(collection,
#                      fsh,
#                      n_clusters=_nclusters_,
#                      Nhood=100,
#                      clustering_algorithm='AgglomerativeWard'):
#     out_signals = [np.zeros(c.signals.shape, _dtype_) for c in collection]
#     out_counts = np.zeros(len(collection), np.int)    # make crossafade here
#     squares = make_grid(fsh[1:], Nhood, Nhood // 2)
#     tstarts = set(c.sq[0].start for c in collection)
#     tsquares = [(t, sq) for t in tstarts for sq in squares]
#     clustering_dispatcher = {
#         'AgglomerativeWard'.lower():
#         lambda nclust: skcluster.AgglomerativeClustering(nclust),
#         'KMeans'.lower(): lambda nclust: skcluster.KMeans(nclust),
#         'MiniBatchKMeans'.lower(): lambda nclust: skcluster.MiniBatchKMeans(nclust)
#     }
#
#     def _is_local_patch(p, sqx):
#         t0, sq = sqx
#         tstart = p[0].start
#         psq = p[1:]
#         return (tstart == t0) & (slice_overlaps_square(psq, sq))
#
#     for sqx in tqdm(tsquares, desc='Going through larger squares'):
#         #print(sqx, collection[0][3], slice_starts_in_square(collection[0][3], sqx))
#         signals = [c.signals for c in collection if _is_local_patch(c.sq, sqx)]
#         if not (len(signals)):
#             print(sqx)
#             for c in collection:
#                 print(sqx, c.sq, _is_local_patch(c.sq, sqx))
#         nclust = min(n_clusters, len(signals))
#         signals = np.vstack(signals)
#         # number of signals can be different in some patches due to boundary conditions
#         clust = clustering_dispatcher[clustering_algorithm.lower()](nclust)
#         if clustering_algorithm == "MiniBatchKMeans".lower():
#             clust.batch_size = min(clust.batch_size, len(signals))
#         labels = clust.fit_predict(signals)
#         sqx_approx = np.zeros(signals.shape, _dtype_)
#         for i in np.unique(labels):
#             group = labels == i
#             u, s, vh = linalg.svd(signals[group], False)
#             r = min_ncomp(s, (u.shape[0], vh.shape[1])) + 1
#             #w = weight_components(all_svd_signals[group],vh,r)
#             approx = u[:, :r] @ np.diag(s[:r]) @ vh[:r]
#             sqx_approx[group] = approx
#         kstart = 0
#         for i, c in enumerate(collection):
#             if _is_local_patch(c.sq, sqx):
#                 l = len(c.signals)
#                 out_signals[i] += sqx_approx[kstart:kstart + l]
#                 out_counts[i] += 1
#                 kstart += l
#     return [SVD_patch(x / (1e-7+cnt), *c[1:])
#             for c, x, cnt in zip(collection, out_signals, out_counts)]



def patch_svd_denoise_frames(frames,
                             do_second_stage=False,
                             save_coll=None,
                             tsvd_kw=None,
                             second_stage_kw=None,
                             inverse_kw=None):
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
    coll = patch_tsvd_transform(frames, **tsvd_kw)
    if do_second_stage:
        coll = second_stage_svd(coll, frames.shape, **second_stage_kw)
    if save_coll is not None:
        with gzip.open(save_coll, 'wb') as fh:
            pickle.dump((coll, frames.shape), fh)
    return patch_tsvd_inverse_transform(coll, frames.shape, **inverse_kw)
