from collections import namedtuple

import numpy as np
from numpy import linalg

from scipy import ndimage as ndi
from scipy.stats import skew
import scipy as sp

from sklearn.feature_extraction.image import grid_to_graph


from tqdm.auto import tqdm



from . import scramble


from .anscombe import Anscombe
from .patches import make_grid
from .utils import adaptive_filter_1d, adaptive_filter_2d
from .cluster import clustering_dispatcher_
from .masks import cleanup_cluster_map
from sklearn import cluster as skclust


from .globals import _dtype_

_do_pruning_ = False


def lambda_star(beta):
    return np.sqrt(2 * (beta+1) + (8*beta) / (beta + 1 + np.sqrt(beta**2 + 14*beta + 1)))


def omega_approx(beta):
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82*beta + 1.43


def svht(sv, sh, sigma=None):
    "Gavish and Donoho 2014"
    m, n = sh
    if m > n:
        m, n = n, m
    beta = m / n
    omg = omega_approx(beta)
    if sigma is None:
        return omg * np.median(sv)
    else:
        return lambda_star(beta) * np.sqrt(n) * sigma


def min_ncomp(sv, sh, sigma=None):
    th = svht(sv, sh, sigma)
    return np.sum(sv >= th)


def pca_flip_signs(pcf, medianw=None):
    L = len(pcf.coords)
    if medianw is None:
        medianw = L // 5
    for i, c in enumerate(pcf.coords.T):
        sk = skew(c - ndi.median_filter(c, medianw))
        sg = np.sign(sk)
        #print(i, sk)
        pcf.coords[:, i] *= sg
        pcf.tsvd.components_[i] *= sg
    return pcf


def svd_flip_signs(u, vh, mode='v'):
    "flip signs of U,V pairs of the SVD so that either V or U have positive skewness"
    for i in range(len(vh)):
        if mode == 'v':
            sg = np.sign(skew(vh[i]))
        else:
            sg = np.sign(skew(u[:, i]))
        u[:, i] *= sg
        vh[i] *= sg
    return u, vh


def unfolding(k,X):
    sh = X.shape
    dimlist = list(range(len(sh)))
    dimlist[k],dimlist[0] = 0,k
    return np.transpose(X,dimlist).reshape(sh[k],-1)

#import tensorflow as tf
def modalsvd(k,X):
    kX = unfolding(k,X)
    return np.linalg.svd(kX, full_matrices=False)
    #return tf.linalg.svd(kX, full_matrices=False)


class HOSVD:
    def __init__(self):
        self.collapsed_ = False
    def fit_transform(self, X, r=None,min_ncomps=1,max_ncomps=None):
        Ulist = []
        S = X
        sh = X.shape

        if not np.iterable(r):
            r = [r]*len(sh)

        for i,ni in enumerate(X.shape):
            u,s,vh = modalsvd(i,X)
            # the following actually doesn't produce good results
            # and is not recommended
            rank = min_ncomp(s, (u.shape[0],vh.shape[1])) + 1
            #print('rank guess 1:', rank)
            rank = max(min_ncomps, rank)
            #print('rank guess 2:', rank)
            if max_ncomps is not None:
                rank = min(max_ncomps, rank)
                #print('rank guess 3:', rank)

            rank = rank if r[i] is None else r[i]
            #print('rank guess 4:', rank)
            u = u[:,:rank]
            Ulist.append(u)
            S = np.tensordot(S,u.T,axes=(0,1))
        self.S_ = S
        self.Ulist_ = Ulist
        self.ranks_ = r
        return S,Ulist

    def inverse_transform(self, S=None, Ulist=None):
        S = self.S_ if S is None else S
        Ulist = self.Ulist_ if Ulist is None else Ulist
        if not self.collapsed_:
            Xrec = S
            for u in Ulist:
                Xrec = np.tensordot(Xrec, u, (0,1))
        else:
            Xrec = Ulist[-1]
            for u in Ulist[-2::-1]:
                Xrec = np.tensordot(u, Xrec, (1,-3))
        return Xrec

    def collapse_2last_dimensions(self, S=None, Ulist=None):
        S = self.S_ if S is None else S
        Ulist = self.Ulist_ if Ulist is None else Ulist
        if not self.collapsed_:
            Ssh = S.shape
            Sndim = len(Ssh)

            C1 = np.tensordot(S, Ulist[-2],(Sndim-2,1))
            C1 = np.tensordot(C1, Ulist[-1],(Sndim-2,1))

            self.collapsed_ = True
            self.Ulist_ = list(Ulist[:-2]) + [C1]
        return self.Ulist_




SVD_patch = namedtuple('SVD_patch', "signals filters sigma center sq w_shape toverlap soverlap")
HOSVD_patch = namedtuple('HOSVD_patch', "hosvd center sq w_shape toverlap soverlap")


def simple_tSVD(signals, min_ncomps=1, max_ncomps=100, return_components=True):
    sh = signals.shape
    u, s, vh = np.linalg.svd(signals, False)
    r = min_ncomp(s, (u.shape[0], vh.shape[1])) + 1
    r = min(max_ncomps, max(r, min_ncomps))
    u, vh = svd_flip_signs(u[:,:r], vh[:r])
    return u,s[:r],vh

def superpixel_tSVD(signals,
                    Niter=3,
                    nclusters=5,
                    alpha=0.1,
                    grid_shape=None,
                    min_ncomps = 1,
                    max_ncomps = 100,
                    do_cleanup_label_maps=False,
                    return_components=True):
    approx = []
    sh = signals.shape
    connectivity_ward = None
    if grid_shape is not None:
        connectivity_ward = grid_to_graph(*grid_shape)

    labels = None # just to put this name into outer context
    comps = {}

    if connectivity_ward is None:
        clusterer = clustering_dispatcher_['minibatchkmeans'](nclusters)
        clusterer.batch_size = min(clusterer.batch_size, len(signals))
        if clusterer.init_size is None:
            clusterer.init_size=3*nclusters
        clusterer.init_size = max(3 * nclusters, clusterer.init_size)
    else:
        clusterer = skclust.AgglomerativeClustering(nclusters,connectivity=connectivity_ward)

    for k in (range(Niter)):
        # could also "improve" signals for labeling by smoothing or projection to low-rank spaces
        if nclusters >1 :
            label_signals = signals if k == 0 else np.mean(approx,0)#/i
            labels = clusterer.fit_predict(label_signals)
            if do_cleanup_label_maps:
                labels = cleanup_cluster_map(labels.reshape((len(labels),1)), min_neighbors=2, niter=10).ravel()
        else:
            labels = np.ones(signals.shape,dtype=np.int)
        #alpha = k/Niter
        update_signals = (1-alpha)*signals + alpha*np.mean(approx,0) if k > 0 else signals
        update = np.zeros_like(update_signals)
        comps = {}
        for ll in np.unique(labels):
            group = labels == ll
            u,s,vh = simple_tSVD(signals[group])
            comps[ll] = (u,s,vh)
            app = u @ np.diag(s) @ vh
            update[group] = app
        approx.append(update)

    if return_components:
        Ulist,Slist,Vhlist = [],[],[]
        for ll in comps:
            u,s,vh = comps[ll]
            Slist.append(s)
            ui = np.zeros((sh[0], len(s)))
            ui[labels==ll] = u
            Ulist.append(ui)
            Vhlist.append(vh)

        U = np.hstack(Ulist)
        S = np.concatenate(Slist)
        Vh = np.vstack(Vhlist)
        return U,S,Vh
    else:
        kstart = 1 if Niter > 1 else 0
        approx = np.mean(approx[kstart:],0)
        return approx


class Windowed_tSVD():
    def __init__(self,
                 patch_ssize:'spatial size of the patch'=8,
                 patch_tsize:'temporal size of the patch'=-1,
                 soverlap:'spatial overlap between patches'=4,
                 toverlap:'temporal overlap between patches'=100,
                 min_ncomps:'minimal number of SVD components to use'=1,
                 max_ncomps:'maximal number of SVD components'=100,
                 nclusters: 'number of clusters for superpixels' = 1,
                 use_connectivity: 'use grid connectivity for clustering'=True,
                 cluster_niterations:'number of superpixel iterations'=2,
                 do_pruning:'pruning of spatial coefficients'=_do_pruning_,
                 center_data:'subtract mean before SVD'=True,
                 tfilter:'window of adaptive median filter for temporal components'=3,
                 sfilter:'window of adaptive median filter for spatial components'=3,
                 verbose=False):

        self.patch_ssize = patch_ssize
        self.soverlap = soverlap

        self.patch_tsize = patch_tsize
        self.toverlap = toverlap

        self.min_ncomps = min_ncomps
        self.max_ncomps = max_ncomps

        self.center_data = center_data

        self.t_amf = tfilter
        self.s_amf = sfilter

        self.patches_ = None
        self.verbose = verbose

        self.nclusters = nclusters
        self.use_connectivity = use_connectivity
        self.cluster_niterations = cluster_niterations

        self.do_pruning = do_pruning
        self.fit_transform_ansc = Anscombe.wrap_input(self.fit_transform)
        self.inverse_transform_ansc = Anscombe.wrap_output(self.inverse_transform)

    def fit_transform(self, frames,):
        data = np.array(frames).astype(_dtype_)
        acc = []
        L = len(frames)

        if (self.patch_tsize <=0) or (self.patch_tsize > L):
            patch_tsize = L
        else:
            patch_tsize = self.patch_tsize

        if self.toverlap >= patch_tsize:
            self.toverlap = patch_tsize // 4

        squares = make_grid(np.shape(frames),
                            (patch_tsize, self.patch_ssize, self.patch_ssize),
                            (self.toverlap, self.soverlap, self.soverlap))

        tsmoother = lambda v:v
        ssmoother = lambda v:v

        if self.t_amf > 0:
            tsmoother = lambda v: adaptive_filter_1d(
                v, th=3, smooth=self.t_amf, keep_clusters=False)
        if self.s_amf > 0:
            ssmoother = lambda v: adaptive_filter_2d(v.reshape(self.patch_ssize, -1),
                                                     smooth=self.s_amf,
                                                     keep_clusters=False).reshape(v.shape)

        for sq in tqdm(squares, desc='superpixel truncSVD in patches', disable=not self.verbose):

            patch_frames = data[sq]
            L = len(patch_frames)
            w_sh = np.shape(patch_frames)

            # now each column is signal in one pixel
            patch = patch_frames.reshape(L,-1)
            #pnorm = np.linalg.norm(patch)
            patch_c = np.zeros(patch.shape[1])
            if self.center_data:
                patch_c = np.mean(patch, 0)
                patch = patch - patch_c

            # now each row is one pixel
            signals = patch.T
            grid_shape = w_sh[1:] if self.use_connectivity else None

            if self.nclusters > 1:
                u,s,vh = superpixel_tSVD(signals,
                                         Niter=self.cluster_niterations,
                                         nclusters=self.nclusters,
                                         grid_shape=grid_shape)
            else:
                u,s,vh = simple_tSVD(signals, min_ncomps=self.min_ncomps, max_ncomps=self.max_ncomps, )

            if self.do_pruning:
                w = weight_components(signals, vh)
            else:
                w = np.ones(u.shape)

            svd_signals, loadings = vh, u*w


            # How to make it a convenient option?
            svd_signals = svd_signals * s[:, None]**0.5
            loadings = loadings * s[None,:]**0.5
            s = np.ones(len(s))

            if self.t_amf > 0:
                svd_signals = np.array([tsmoother(v) for v in svd_signals])

            W = loadings.T
            if (self.s_amf > 0) and (patch.shape[1] == self.patch_ssize**2):
                W = np.array([ssmoother(v) for v in W])

            p = SVD_patch(svd_signals, W, s, patch_c, sq, w_sh, self.toverlap, self.soverlap)
            acc.append(p)
        self.patches_ = acc
        self.data_shape_ = np.shape(frames)
        return self.patches_

    def inverse_transform(self, patches=None, inp_data=None):
        if patches is None:
            patches = self.patches_

        out_data = np.zeros(self.data_shape_, dtype=_dtype_)
        counts = np.zeros(self.data_shape_, _dtype_)    # candidate for crossfade

        for p in tqdm(patches,
                      desc='truncSVD inverse transform',
                      disable=not self.verbose):

            L = p.w_shape[0]

            if (self.patch_tsize <= 0) or (self.patch_tsize >= L):
                patch_tsize = L
            else:
                patch_tsize = self.patch_tsize

            if patch_tsize >= L:
                t_crossfade = np.ones(L, _dtype_)
            else:
                t_crossfade = tanh_step(np.arange(L), L, p.toverlap, p.toverlap/2).astype(_dtype_)
            t_crossfade = t_crossfade[:, None, None]

            psize = np.max(p.w_shape[1:])
            scf = tanh_step(np.arange(psize), psize, p.soverlap, p.soverlap/2)
            scf = scf[:,None]
            w_crossfade = scf @ scf.T
            nr,nc = p.w_shape[1:]
            w_crossfade = w_crossfade[:nr, :nc].astype(_dtype_)
            w_crossfade = w_crossfade[None, :, :]

            counts[p.sq] += t_crossfade * w_crossfade

            #rnorm = np.linalg.norm(rec)
            #rec = rec*p.pnorm/rnorm
            sigma = np.diag(p.sigma)
            if inp_data is not None:
                pdata = inp_data[p.sq].reshape(L,-1)
                pdata_c =  pdata - p.center
                #sigma = np.linalg.pinv(p.signals.T) @ pdata_c @ np.linalg.pinv(p.filters)
                #sigma = p.signals @ pdata_c @ p.filters.T
                new_filters =  np.linalg.pinv(p.signals.T) @ pdata_c
                #new_filters =  p.signals @ pdata_c
                p = p._replace(filters = new_filters)
                sigma = np.diag(np.ones(len(sigma)))

            rec = (p.signals.T @ sigma @ p.filters).reshape(p.w_shape)
            rec += p.center.reshape(p.w_shape[1:])

            out_data[p.sq] += rec * t_crossfade * w_crossfade

        out_data /= (1e-12 + counts)
        out_data *= (counts > 1e-12)

        return out_data


class Windowed_tHOSVD():
    def __init__(self,
                 patch_ssize:'spatial size of the patch'=8,
                 patch_tsize:'temporal size of the patch'=-1,
                 soverlap:'spatial overlap between patches'=4,
                 toverlap:'temporal overlap between patches'=100,
                 min_ncomps:'minimal number of SVD components to use'=1,
                 max_ncomps:'maximal number of SVD components'=None,
                 ranks: 'ranks to use for HOSVD'=None,
                 center_data:'subtract mean before desomposition'=True,
                 tfilter:'window of adaptive median filter for temporal components'=3,
                 sfilter:'window of adaptive median filter for spatial components'=3,
                 Sth_percentile=0,
                 verbose=False):

        self.patch_ssize = patch_ssize
        self.soverlap = soverlap

        self.patch_tsize = patch_tsize
        self.toverlap = toverlap

        self.min_ncomps = min_ncomps
        self.max_ncomps = max_ncomps

        self.center_data = center_data

        self.t_amf = tfilter
        self.s_amf = sfilter
        self.use_collapsed_hosvd = False

        self.patches_ = None
        self.verbose = verbose
        self.Sth_percentile = Sth_percentile

        self.ranks = ranks


        self.fit_transform_ansc = Anscombe.wrap_input(self.fit_transform)
        self.inverse_transform_ansc = Anscombe.wrap_output(self.inverse_transform)

    def fit_transform(self, frames,):
        data = np.array(frames).astype(_dtype_)
        acc = []
        L = len(frames)

        if (self.patch_tsize <=0) or (self.patch_tsize > L) :
            patch_tsize = L
        else:
            patch_tsize = self.patch_tsize


        if self.toverlap >= patch_tsize:
            self.toverlap = patch_tsize // 4

        squares = make_grid(np.shape(frames),
                            (patch_tsize, self.patch_ssize, self.patch_ssize),
                            (self.toverlap, self.soverlap, self.soverlap))


        for sq in tqdm(squares, desc='tHOSVD in patches', disable=not self.verbose):

            patch = data[sq]
            L = len(patch)
            w_sh = np.shape(patch)
            #ranks = w_sh[0]//2,w_sh[1]//2,w_sh[2]//2 # fixed for testing
            ranks = None
            if self.ranks is None:
                ranks = (None,w_sh[1]//2,w_sh[2]//2) # fixed for testing
            else:
                ranks = self.ranks
            #print(ranks)

            patch_c = np.zeros(w_sh[1:])

            if self.center_data:
                patch_c = np.mean(patch,0)
                patch = patch - patch_c

            hosvd = HOSVD()
            S,Ulist = hosvd.fit_transform(patch, ranks, self.min_ncomps, self.max_ncomps)

            p = HOSVD_patch(hosvd, patch_c, sq, w_sh, self.toverlap, self.soverlap)
            acc.append(p)
        self.patches_ = acc
        self.data_shape_ = np.shape(frames)
        return self.patches_

    def inverse_transform(self, patches=None, inp_data=None):
        if patches is None:
            patches = self.patches_

        out_data = np.zeros(self.data_shape_, dtype=_dtype_)
        counts = np.zeros(self.data_shape_, _dtype_)    # candidate for crossfade

        tsmoother = lambda v: v
        ssmoother = lambda v: v

        if self.t_amf > 0:
            tsmoother = lambda v: adaptive_filter_1d(v, th=3, smooth=self.t_amf, keep_clusters=False)
        if self.s_amf > 0:
            ssmoother = lambda v: adaptive_filter_2d(v,
                                                     smooth=self.s_amf,
                                                     keep_clusters=False)

        for p in tqdm(patches,
                      desc='tHOSVD inverse transform',
                      disable=not self.verbose):

            L = p.w_shape[0]
            t_crossfade = tanh_step(np.arange(L), L, p.toverlap, p.toverlap/2).astype(_dtype_)
            t_crossfade = t_crossfade[:, None, None]

            psize = np.max(p.w_shape[1:])
            scf = tanh_step(np.arange(psize), psize, p.soverlap, p.soverlap/2)
            scf = scf[:,None]
            w_crossfade = scf @ scf.T
            nr,nc = p.w_shape[1:]
            w_crossfade = w_crossfade[:nr, :nc].astype(_dtype_)
            w_crossfade = w_crossfade[None, :, :]

            counts[p.sq] += t_crossfade * w_crossfade
            S, Ulist = p.hosvd.S_, p.hosvd.Ulist_

            if self.Sth_percentile > 0:
                thresh = np.percentile(np.abs(S), self.Sth_percentile)
                S = np.where(np.abs(S)>=thresh, S, 0)

            p.hosvd.S_ = S

            if self.t_amf > 0:
                # I rely here that first dimension is time, and HOSVD is 3D
                # this is not generalized to 4D imaging or bags of patches yet
                Utmp = np.array([tsmoother(v) for v in Ulist[0].T]).T
                p.hosvd.Ulist_[0] = Utmp

            if self.s_amf > 0:
                Usp = p.hosvd.collapse_2last_dimensions()[-1]
                Usp = np.array([ssmoother(m) for m in Usp])
                p.hosvd.Ulist_[-1] = Usp
                #print([u.shape for u in p.hosvd.Ulist_])

            rec = p.hosvd.inverse_transform()
            #rec = p.hosvd.inverse_transform(S, Ulist)
            rec += p.center
            out_data[p.sq] += rec * t_crossfade * w_crossfade

        out_data /= (1e-12 + counts)
        out_data *= (counts > 1e-12)

        return out_data

def weight_components(data, components, rank=None, Npermutations=100, clip_percentile=95):
    """
    For a collection of signals (each row of input matrix is a signal),
    try to decide if using projection to the principal or svd components should describe
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
    v_shuffled = (scramble.shuffle_signals(components[:rank])
                  for i in range(Npermutations))
    coefs_randomized = np.array([np.abs(data @ vt.T).T for vt in v_shuffled])
    coefs_orig = np.abs(data @ components[:rank].T).T
    w = np.zeros((len(data), len(components[:rank])), _dtype_)
    for j in np.arange(w.shape[1]):
        w[:, j] = coefs_orig[j] >= np.percentile(
            coefs_randomized[:, j, :], clip_percentile, axis=0)
    return w


def tanh_step(x, window, overlap, taper_k=None):
    overlap = max(1, overlap)
    taper_width = overlap / 2
    if taper_k is None:
        taper_k = overlap / 10
    A = np.tanh((x+0.5-taper_width) / taper_k)
    B = np.tanh((window-(x+0.5)-taper_width) / taper_k)
    return np.clip((1.01 + A*B)/2, 0, 1)


def DyCA(data, min_ncomp=2, eig_threshold = 0.98):
    "Given data of the form (time, sensors) returns the DyCA projection and projected data with eigenvalue threshold eig_threshold"
    derivative_data = np.gradient(data,axis=0,edge_order=1) #get the derivative of the data
    L = data.shape[0] #for time averaging
    #construct the correlation matrices
    C0 = data.T @ data / L
    C1 = derivative_data.T @ data / L
    C2 = derivative_data.T @ derivative_data / L

    try:
        eigvalues, eigvectors = sp.linalg.eig(C1 @ np.linalg.inv(C0) @ C1.T, C2)
        eigvalues = eigvalues.real
        sorted_eigvals = sorted(eigvalues, reverse=True)
        min_ncomp = min(min_ncomp, len(eigvalues))
        eig_threshold = min(sorted_eigvals[min_ncomp-1], eig_threshold)
        eigvectors = eigvectors[:,np.array(eigvalues > eig_threshold) &  np.array(eigvalues <= 1)] # eigenvalues > 1 are artifacts of singularity of C0
        if eigvectors.shape[1] > 0:
            #C3 = np.matmul(np.linalg.inv(C1), C2)
            C3 = np.linalg.inv(C1)@C2
            proj_mat = np.concatenate((eigvectors, np.apply_along_axis(lambda x: C3@x, 0, eigvectors)),axis=1)
        else:
            raise ValueError('No generalized eigenvalue fulfills threshold!')
        return proj_mat, data@proj_mat, eigvalues
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.zeros(data.shape[1])+np.nan
