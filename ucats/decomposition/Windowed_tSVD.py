from collections import namedtuple
import numpy as np
from numpy import linalg

from scipy import ndimage as ndi
from scipy.stats import skew
import scipy as sp

from tqdm.auto import tqdm

from ..patches import make_grid
from ..utils import adaptive_filter_1d, adaptive_filter_2d

from ..anscombe import Anscombe
from ..globals import _dtype_

from .svd_utils import min_ncomp, pca_flip_signs, svd_flip_signs, simple_tSVD
from .svd_utils import weight_components, tanh_step

_do_pruning_ = False


SVD_patch = namedtuple('SVD_patch', "signals filters sigma center sq w_shape toverlap soverlap")

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
