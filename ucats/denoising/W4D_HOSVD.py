import numpy as np
from numpy import linalg

from collections import namedtuple

from numba import jit

from scipy import ndimage as ndi

from sklearn import cluster as skclust

import gzip
import pickle

from tqdm.auto import tqdm

from imfun import components

from ..anscombe import Anscombe

from ..cluster import sort_by_clust, clustering_dispatcher_

from ..decomposition import (min_ncomp, HOSVD_patch, HOSVD, Windowed_tHOSVD, tanh_step)

from ..globals import _dtype_

from ..patches import make_grid, slice_overlaps_square

from ..utils import mad_std

from skimage import transform as sktransform




Centered_patch = namedtuple('Centered_patch', 'center sq w_shape toverlap soverlap')

# consider subclassing this from Windowed_tHOSVD?
class NL_Windowed_HOSVD():
    def __init__(self,
                 sNhood=100,
                 tNhood=1,
                 patch_ssize:'spatial size of the patch'=8,
                 patch_tsize:'temporal size of the patch'=600,
                 soverlap:'spatial overlap between patches'=4,
                 toverlap:'temporal overlap between patches'=100,
                 center_data:'subtract mean before decomposition'=True,
                 Sth_percentile=75,
                 ranks=None,
                 verbose=False):

        self.patch_ssize = patch_ssize
        self.soverlap = soverlap

        self.patch_tsize = patch_tsize
        self.toverlap = toverlap

        self.center_data = center_data

        self.patches_ = None
        self.verbose = verbose

        self.nl_ranks = ranks
        self.Sth_percentile = Sth_percentile
        self.sNhood = sNhood
        self.tNhood = tNhood

        # Anscombe wrappers
        self.denoise_ansc = Anscombe.wrap(self.denoise)
        self.fit_transform_ansc = Anscombe.wrap_input(self.fit_transform)
        self.inverse_transform_ansc = Anscombe.wrap_output(self.inverse_transform)
        return


    def denoise(self, frames):
        coll = self.fit_transform(frames)
        return self.inverse_transform()


    def fit_transform(self, frames,):

        data = np.array(frames).astype(_dtype_)
        L = len(data)

        self.data_shape_ = np.shape(data)
        fsh = self.data_shape_

        self.patch_tsize = min(L, self.patch_tsize)
        if self.toverlap >= self.patch_tsize:
            self.toverlap = self.patch_tsize // 4

        local_squares = make_grid(np.shape(data),
                                  (self.patch_tsize, self.patch_ssize, self.patch_ssize),
                                  (self.toverlap, self.soverlap, self.soverlap))


        nl_squares = make_grid(fsh[1:], self.sNhood, self.sNhood // 2)
        tstarts = set(sq[0].start for sq in local_squares)            # haven't decided on whether to combine different times yet
        tsquares = [(t, sq) for t in tstarts for sq in nl_squares]

        patches1 = []
        for sq in local_squares:
            patch = data[sq]
            L = len(patch)
            w_sh = np.shape(patch)
            patch_c = np.zeros(w_sh[1:])
            if self.center_data:
                patch_c = np.mean(patch,0)
                #patch = patch - patch_c
            patches1.append(Centered_patch(patch_c, sq, w_sh, self.toverlap, self.soverlap))



        def _is_local_patch(p, sqx):
            t0, sq = sqx
            tstart = p[0].start
            psq = p[1:]
            #return (tstart == t0) & (slice_overlaps_square(psq, sq))
            return (np.abs(tstart - t0) <= self.tNhood) & (slice_overlaps_square(psq, sq))

        loop = tqdm(tsquares, desc=f'4D HOSVD', disable=not self.verbose)
        self.nl_processed_patches = []
        for sqx in loop:
            #samples = [getattr(p, field) for p in patches1 if _is_local_patch(c.sq, sqx)]
            aux_coll = [p for p in patches1 if _is_local_patch(p.sq, sqx)]
            patches_nl = np.array([px-p.center for px,p in ((data[p.sq],p) for p in aux_coll)])
            nl_shape4d = patches_nl.shape
            hosvd = HOSVD()
            ranks = None
            if self.nl_ranks is None:
                # this is just ad-hoc, and should probably be tested thoroughly
                ranks = np.round(np.array(nl_shape4d)/np.array([4,2,1,1])).astype(int)
            else:
                ranks = self.nl_ranks
            S,Ulist = hosvd.fit_transform(patches_nl, r=ranks) # second-pass HOSVD run, ranks can be misjudged
            Sthreshold = np.percentile(np.abs(S),self.Sth_percentile)
            hosvd.S_ = S*(np.abs(S) > Sthreshold)
            self.nl_processed_patches.append((hosvd, aux_coll))
            #patches_nl_rec = hosvd.inverse_transform(S*(np.abs(S)>Sthreshold), Ulist)
            #self.patches_nl_rec_ = patches_nl_rec

    def inverse_transform(self):

        output = np.zeros(self.data_shape_, dtype=_dtype_)
        counts = np.zeros(self.data_shape_, dtype=_dtype_)

        for hosvd,aux_coll in self.nl_processed_patches:
            patch_nl_rec = hosvd.inverse_transform()
            for p, prec in zip(aux_coll,patch_nl_rec):
                L = p.w_shape[0]
                t_crossfade = tanh_step(np.arange(L), L, p.toverlap, p.toverlap/2).astype(_dtype_)
                t_crossfade = t_crossfade[:, None, None]

                psize = np.max(p.w_shape[1:])
                scf = tanh_step(np.arange(psize), psize, p.soverlap, p.soverlap/2).astype(_dtype_)
                scf = scf[:,None]
                w_crossfade = scf @ scf.T
                nr,nc = p.w_shape[1:]
                w_crossfade = w_crossfade[:nr, :nc].astype(_dtype_)
                w_crossfade = w_crossfade[None, :, :]

                output[p.sq] += (prec + p.center)*t_crossfade*w_crossfade
                counts[p.sq] +=  t_crossfade * w_crossfade

        output /= counts + 1e-5
        output[counts<=0] = 0
        return output








class Multiscale_NL_Windowed_HOSVD():
    pass
