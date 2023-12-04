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

HOSVD_patch = namedtuple('HOSVD_patch', "hosvd center sq w_shape toverlap soverlap")


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
