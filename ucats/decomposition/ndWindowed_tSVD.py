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

ndSVD_patch = namedtuple('ndSVD_patch', "signals filters sigma center sq w_shape overlap")

class ndWindowed_tSVD():
    def __init__(self,
                 patch_size:'spatial size of the patch'=(-1,-1,8,8), # all time and Z, 8x8 in YX
                 patch_overlap:'spatial overlap between patches'=(0,0,4,4),
                 min_ncomps:'minimal number of SVD components to use'=1,
                 max_ncomps:'maximal number of SVD components'=100,
                 nclusters: 'number of clusters for superpixels' = 1,
                 use_connectivity: 'use grid connectivity for clustering'=True,
                 cluster_niterations:'number of superpixel iterations'=2,
                 #do_pruning:'pruning of spatial coefficients'=_do_pruning_,
                 center_data:'subtract mean before SVD'=True,
                 tfilter:'window of adaptive median filter for temporal components'=3,
                 sfilter:'window of adaptive median filter for spatial components'=3,
                 verbose=False):

        self.patch_size = list(patch_size)
        self.patch_overlap = list(patch_overlap)


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

        self.do_pruning = False
        self.fit_transform_ansc = Anscombe.wrap_input(self.fit_transform)
        self.inverse_transform_ansc = Anscombe.wrap_output(self.inverse_transform)

    def fit_transform(self, frames,):
        data = np.array(frames).astype(_dtype_)
        acc = []

        sh = frames.shape

        L = sh[0]
        Z = sh[1]

        # T axis (volumes)
        if (self.patch_size[0] <=0) or (self.patch_size[0] > L):
            self.patch_size[0] = L

        if self.patch_overlap[0] >= self.patch_size[0]:
            self.patch_overlap[0] = self.patch_size[0] // 2


        # Z axis (planes)
        if (self.patch_size[1] <=0) or (self.patch_size[1] > Z):
            self.patch_size[1] = Z


        if self.patch_overlap[1] >= self.patch_size[1]:
            self.patch_overlap[1] = self.patch_size[1] // 2



        squares = make_grid(np.shape(frames), self.patch_size, self.patch_overlap)

        tsmoother = lambda v:v
        ssmoother = lambda v:v

        if self.t_amf > 0:
            tsmoother = lambda v: adaptive_filter_1d(
                v, th=3, smooth=self.t_amf, keep_clusters=False)

        if self.s_amf > 0:
            # note could alternatively do Z-planewise with adaptive_filter_2d
            ssmoother = lambda v: adaptive_median_filter(v.reshape(self.patch_size[1:]),
                                                     smooth=self.s_amf,
                                                     keep_clusters=False).reshape(v.shape)

        for sq in tqdm(squares, desc='4d truncSVD in patches', disable=not self.verbose):

            patch_volume = data[sq]

            L = len(patch_volume)
            w_sh = np.shape(patch_volume)
            #print('Patch shape:', w_sh)


            # now each column is signal in one pixel
            patch = patch_volume.reshape(L,-1)
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
            if (self.s_amf > 0) and (w_sh == self.patch_size[1:]):
                W = np.array([ssmoother(v) for v in W])

            p = ndSVD_patch(svd_signals, W, s, patch_c, sq, w_sh, self.patch_overlap)
            acc.append(p)
        self.patches_ = acc
        self.data_shape_ = np.shape(frames)
        return self.patches_

    def inverse_transform(self, patches=None, inp_data=None):
        if patches is None:
            patches = self.patches_


        vol_shape = self.data_shape_

        out_data = np.zeros(vol_shape, dtype=_dtype_)
        counts = np.zeros(vol_shape, _dtype_)    # candidate for crossfade

        for p in tqdm(patches,
                      desc='4d truncSVD inverse transform',
                      disable=not self.verbose):

            L,Z = vol_shape[:2]


            ndim = len(p.w_shape)

            #if (self.patch_tsize <= 0) or (self.patch_tsize >= L):
            #    patch_tsize = L
            #else:
            #    patch_tsize = self.patch_tsize

            patch_tsize = self.patch_size[0]
            patch_zsize = self.patch_size[1]

            toverlap = p.overlap[0]
            zoverlap = p.overlap[1]

            crossfades = np.ones(p.w_shape, dtype=_dtype_)
            for kd, (dim, psize, overlap) in enumerate(zip(vol_shape, p.w_shape, p.overlap)):
                if (psize >= dim) or overlap <= 0:
                    cfade = np.ones(psize, _dtype_)
                else:
                    cfade = tanh_step(np.arange(psize), psize, overlap, overlap/2).astype(_dtype_)

                csh = (tuple(1 for i in range(kd))
                       + (psize,)
                       + tuple(1 for i in range(ndim-kd-1)))
                crossfades *= np.reshape(cfade, csh)

                #crossfades.append(cfade.astype(_dtype_))


            #psize = np.max(p.w_shape[-2:])
            #scf = tanh_step(np.arange(psize), psize, p.soverlap, p.soverlap/2)
            #scf = scf[:,None]
            #w_crossfade = scf @ scf.T
            #nr,nc = p.w_shape[1:]
            #w_crossfade = w_crossfade[:nr, :nc].astype(_dtype_)
            #w_crossfade = w_crossfade[None, :, :]
            #counts[p.sq] += t_crossfade * w_crossfade
            counts[p.sq] += crossfades

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

            out_data[p.sq] += rec * crossfades

        out_data /= (1e-12 + counts)
        out_data *= (counts > 1e-12)

        return out_data
