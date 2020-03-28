"""
Some wrappers for clusterisation routines from scikit-learn and scipy
and a couple of original algorithms
"""

from numba import jit

import numpy as np
from numpy import linalg

from scipy import cluster as sp_clust
from scipy.cluster import hierarchy as sp_hierarchy

from sklearn import cluster as skclust

import hdbscan
from umap import UMAP

from .globals import _dtype_

_nclusters_ = 32

from numba import jit


@jit
def calc_coincidents(labels):
    Npts = len(labels)
    out = np.zeros((Npts, Npts))
    for i in range(Npts):
        for j in range(i, Npts):
            out[i, j] = labels[i] == labels[j]
    return out


def labels_to_sort(labels):
    return np.concatenate([np.where(labels == i)[0] for i in np.unique(labels)])


def aggward_clusterise(data, n_clusters=10, ncomp=3):
    u, s, vh = linalg.svd(data, False)
    u = u[:, :ncomp]
    clust = skclust.AgglomerativeClustering(n_clusters=n_clusters, compute_full_tree=True)
    labels = clust.fit_predict(u)
    return labels


def sort_by_clust(data, n_clusters=10, ncomp=3, output='labels'):
    """
    Given data (n_samples, n_features), reduce dimensionality by SVD and do
    Agglomerative clusterisation with ward linkage. If output is `sort`,
    return leaves of the clasterisation tree. If output is `labels`, convert to
    flat clusters and return labels.

    Input:
    -------
     - data: data points, 2D array (n_samples, n_features)
     - n_clusters [10]: if output is `labels` cut the agglomerative clustering tree
       at this number of clusters (based on cophenetic distances); if `None`, try
       find optimal number of clusters from Calinski-Harabasz criterion (slow)
     - ncomp [3]: number of SVD components to use in dimensionality reduction step
     - output [labels]: if output is `labels`, return flat clusters, if output is `sort`,
       return leaves of the agglomerative clustering tree as a 1D array.
    """
    u, s, vh = linalg.svd(data, False)
    u = u[:, :ncomp]
    nsamples = len(u)
    Z = sp_hierarchy.linkage(u, method='ward')
    if 'sort' in output:
        #Z = sp_clust.hierarchy.linkage(u, method='ward')
        return sp_hierarchy.leaves_list(Z)
    else:
        if n_clusters is not None:
            labels = sp_hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
        else:
            # this is just a dumb guess. must be tested though
            # gap statistic or CH index? (how to calculate in Python?)
            # i.e. sklearn.metrics.calinski_harabasz_score
            # calc labels for several n_clusters, find max CH score. (nclust>=2)
            nsignals_per_cluster = range(2, 50, 2)
            nc_acc = []
            ch_acc = []
            for nsc in nsignals_per_cluster:
                nc = np.int(np.ceil(nsamples / nsc))
                nc = max(2, nc)
                labels = sp_hierarchy.fcluster(Z, nc, criterion='maxclust')
                ch = skmetrics.calinski_harabasz_score(u, labels)
                ch_acc.append(ch)
                nc_acc.append(nc)
            k = np.argmax(ch_acc)
            labels = sp_hierarchy.fcluster(Z, nc_acc[k], criterion='maxclust')
            #dcoph = sp_hierarchy.cophenet(Z)
            #th = np.percentile(dcoph,5)
            #labels = sp_hierarchy.fcluster(Z,th,criterion='distance')
        return labels


# TODO:
# add clusterizer as an option (with default to MiniBatchKMeans)
class DumbConsensusClusterer:
    """
    Run MiniBatchKMeans several times and group points that tend to fall in the same cluster
    as a new 'meta-cluster'
    """
    def __init__(self,
                 n_clusters,
                 merge_threshold=0.85,
                 n_clusterers=25,
                 min_cluster_size=10,
                 min_overlap=5):
        self.n_clusters = n_clusters
        self.merge_threshold = merge_threshold
        self.n_clusterers = n_clusterers
        self.min_cluster_size = min_cluster_size
        self.min_overlap = min_overlap

    def fit_predict(self, X):
        labels = []
        for i in range(self.n_clusterers):
            clusterer = skclust.MiniBatchKMeans(n_clusters=self.n_clusters * 2)
            ll = clusterer.fit_predict(X)
            labels.append(ll)
        labels = np.array(labels)
        return self.cluster_ensembles(labels)

    def calc_consensus_matrix(self, labels):
        Npts = len(labels[0])
        out = np.zeros((Npts, Npts))
        for lab in labels:
            out += calc_coincidents(lab)
        return out / len(labels)

    def cluster_ensembles(self,  labels):
        clusters = []
        Cm = self.calc_consensus_matrix(labels)
        Npts = len(Cm)
        for i in range(Npts):
            row = Cm[i]
            candidates = set(
                [j for j in range(i, Npts) if (row[j] >= self.merge_threshold)])
            if not len(candidates):
                continue
            if not len(clusters):
                clusters.append(set(candidates))
            else:
                overlaps = [
                    c for c in clusters
                    if len(c.intersection(candidates)) >= self.min_overlap
                ]
                non_overlapping = [c for c in clusters if not c in overlaps]
                if len(overlaps):
                    for c in overlaps:
                        candidates.update(c)
                    clusters = [candidates] + non_overlapping
                else:
                    clusters.append(candidates)

        clusters = [c for c in clusters if len(c) >= self.min_cluster_size]

        out_labels = np.zeros(Npts)
        for k, cx in enumerate(clusters):
            for i in cx:
                out_labels[i] = k + 1
        return out_labels


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
    def __init__(self, nclust):
        super(UMAP_MiniBatchKMeans, self).__init__()
        self.clusterer = skclust.MiniBatchKMeans(nclust)


class UMAP_KMeans(UMAP_Preprocessed):
    def __init__(self, nclust):
        super(UMAP_KMeans, self).__init__()
        self.clusterer = skclust.KMeans(nclust)


class UMAP_HDBSCAN(UMAP_Preprocessed):
    def __init__(self, *args, **kwargs):
        super(UMAP_HDBSCAN, self).__init__()
        self.clusterer = hdbscan.HDBSCAN()


class UMAP_DumbConsensus(UMAP_Preprocessed):
    def __init__(self, *args, **kwargs):
        super(UMAP_DumbConsensus, self).__init__()
        self.clusterer = DumbConsensusClusterer()


clustering_dispatcher_ = {
    'AgglomerativeWard'.lower(): lambda nclust: skclust.AgglomerativeClustering(nclust),
    'KMeans'.lower(): lambda nclust: skclust.KMeans(nclust),
    'MiniBatchKMeans'.lower(): lambda nclust: skclust.MiniBatchKMeans(nclust),
    'UMAP_DBSCAN'.lower(): lambda nclust: UMAP_HDBSCAN(min_samples=15),
    'UMAP_KMeans'.lower(): lambda nclust: UMAP_KMeans(nclust),
    'UMAP_MiniBatchKMeans'.lower(): lambda nclust: UMAP_MiniBatchKMeans(nclust),
    'ConsensusKMeans'.lower(): lambda nclust: DumbConsensusClusterer(nclust)
}
