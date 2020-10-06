import numpy as np
from numpy import linalg

from numba import jit

from scipy import ndimage as ndi

from sklearn import cluster as skclust

import gzip
import pickle

from tqdm.auto import tqdm

from imfun import components

from ..anscombe import Anscombe

from ..cluster import sort_by_clust, clustering_dispatcher_

from ..decomposition import (min_ncomp, SVD_patch, Windowed_tSVD)

from ..globals import _dtype_

from ..patches import make_grid, slice_overlaps_square

from ..utils import mad_std

_nclusters_ = 16
_large_nsamples = 4000

# TODO:
# - [ ] Exponential-family PCA
# - [ ] Cut svd_signals into pieces before second-stage SVD
#       - alternatively, look at neighboring patches in time
# - [X] (***) Cluster svd_signals before SVD

from skimage.restoration import denoise_tv_chambolle


def separable_iterated_tv_chambolle(im, sigma_x=1, sigma_y=1, niters=5):
    if (sigma_x <= 0) and (sigma_y <= 0):
        return im
    im_w = np.copy(im)

    for i in range(niters):
        if sigma_y > 0:
            im_w = np.array(
                [denoise_tv_chambolle(cv,
                                      mad_std(cv) * sigma_y)
                 for cv in im_w.T]).T    # columns
        if sigma_x > 0:
            im_w = np.array(
                [denoise_tv_chambolle(rv,
                                      mad_std(rv) * sigma_x) for rv in im_w])    # rows
    return im_w


def separable_iterated_tv_chambolle2(im, sigma_x=1, sigma_y=1, niters=5):
    if (sigma_x <= 0) and (sigma_y <= 0):
        return im
    nr, nc = im.shape
    im_w = np.copy(im)
    for i in range(niters):
        # columns
        if sigma_y > 0:
            for c in range(nc):
                cv = im_w[:, c]
                #yd = cv-signal.savgol_filter(cv, ucats.baselines.make_odd(len(cv)), 3)
                ns = mad_std(np.diff(cv))
                im_w[:, c] = denoise_tv_chambolle(cv, ns * sigma_y)
        # rows
        if sigma_x > 0:
            for r in range(nr):
                rv = im_w[r, :]
                #yd = rv-signal.savgol_filter(rv, ucats.baselines.make_odd(len(rv)), 3)
                ns = mad_std(np.diff(rv))
                im_w[r, :] = denoise_tv_chambolle(rv, ns * sigma_x)
    return im_w


@jit
def unsort(v, ksort):
    #z = np.arange(len(v))
    out = np.zeros(v.shape)
    for i, k in enumerate(ksort):
        out[k] = v[i]
    return out


def windowed_flat_tv(img,
                     window=50,
                     overlap=25,
                     samples_per_cluster=10,
                     mode='sorted',
                     tv_sigma=1,
                     tv_niters=3):
    nr, nc = img.shape
    counts = np.zeros(img.shape)
    out = np.zeros(img.shape)
    window = np.minimum(window, nc)
    tslices = [x[0] for x in make_grid((nc, 1), window, overlap)]
    tslices = [(slice(None), t) for t in tslices]

    if samples_per_cluster is not None:
        nclust = np.int(np.ceil(nr / samples_per_cluster))
    else:
        nclust = None

    for tslice in tslices:
        patch = img[tslice]

        if mode == 'sorted':
            ksort = sort_by_clust(patch, output='sorting')
            sorted_patch = patch[ksort]
            sorted_patch2 = separable_iterated_tv_chambolle2(sorted_patch,
                                                             sigma_x=0,
                                                             sigma_y=tv_sigma,
                                                             niters=tv_niters)
            out[tslice] += unsort(sorted_patch2, ksort)
        elif mode == 'means':
            labels = sort_by_clust(patch, nclust, output='labels')
            patch_approx = np.zeros(patch.shape)
            for i in np.unique(labels):
                patch_approx[labels == i] = patch[labels == i].mean(0)
            out[tslice] += patch_approx
        counts[tslice] += 1
    out = out / (1e-12 + counts)
    out = out * (counts > 0)
    return out


# TODO: decide which clustering algorithm to use.
#       candidates:
#         - KMeans (sklearn)
#         - MiniBatchKMeans (sklearn)
#         - AgglomerativeClustering with Ward or other linkage (sklearn)
#         - SOM aka Kohonen (imfun)
#         - something else?
#       - clustering algorithm should be made a parameter
# TODO: good crossfade and smaller overlap





def process_flat_collection(samples,
                            n_clusters=_nclusters_,
                            clustering_algorithm='MiniBatchKMeans',
                            mode='cluster-svd',
                            **kwargs):

    nclust = min(n_clusters, len(samples))
    samples_approx = np.zeros(samples.shape, _dtype_)

    clust = clustering_dispatcher_[clustering_algorithm.lower()](nclust)
    if clustering_algorithm == "MiniBatchKMeans".lower():
        clust.batch_size = min(clust.batch_size, len(samples))
        clust.init_size = max(3 * nclust, clust.init_size)

    if 'flattv' in mode.lower():
        out_kind = 'sorted' if 'sort' in mode.lower() else 'means'
        tv_sigma = kwargs['tv_sigma'] if 'tv_sigma' in kwargs else 1
        tv_niters = kwargs['tv_niters'] if 'tv_niters' in kwargs else 3

        Nsamples = len(samples)

        labels = np.ones(len(samples))
        sample_batches = [(1,samples)]

        if Nsamples > _large_nsamples:
            npartition = np.int(np.round(Nsamples / _large_nsamples)) + 1
            #clustx = clustering_dispatcher_['kmeans'](npartition)
            #clustx.batch_size = min(clustx.batch_size, Nsamples)
            #u,s,vh = np.linalg.svd(samples,False)
            #labels = clustx.fit_predict(u[:,:3])
            labels = np.random.randint(1,npartition+1,size=Nsamples)
            sample_batches = [(i,samples[labels==i]) for i in np.unique(labels)]

        for i,batch in sample_batches:
            if 'tv_samples_per_cluster' in kwargs:
                samples_per_cluster = kwargs['tv_samples_per_cluster']
            else:
                samples_per_cluster = np.int(np.round(len(batch) / nclust))

            #print(len(batch))
            if len(batch):
                batch_approx = windowed_flat_tv(batch,
                                                mode=out_kind,
                                                samples_per_cluster=samples_per_cluster,
                                                tv_sigma=tv_sigma,
                                                tv_niters=tv_niters)
                samples_approx[labels==i] = batch_approx


    elif 'cluster' in mode.lower():
        if nclust > 1:
            labels = clust.fit_predict(samples)
        else:
            labels = np.ones(len(samples))
        if 'svd' in mode.lower():
            for i in np.unique(labels):
                group = labels == i
                u, s, vh = linalg.svd(samples[group], False)
                r = min_ncomp(s, (u.shape[0], vh.shape[1])) + 1
                u = u[:, :r]
                approx = u @ np.diag(s[:r]) @ vh[:r]
                samples_approx[group] = approx
        else:
            for i in np.unique(labels):
                samples_approx[labels == i] = samples[labels == i].mean(0)
    return samples_approx


class NL_Windowed_tSVD(Windowed_tSVD):
    def __init__(self,
                 Nhood=100,
                 do_signals=True,
                 do_spatial=True,
                 n_clusters=_nclusters_,
                 temporal_mode='flatTV-means',
                 tv_samples_per_cluster=10,
                 **kwargs):
        super().__init__(**kwargs)

        self.Nhood = Nhood
        self.n_clusters = n_clusters
        self.temporal_mode = temporal_mode
        self.tv_samples_per_cluster = tv_samples_per_cluster
        self.tv_sigma = 1.5
        self.tv_niters = 3
        self.clustering_algorithm = 'MiniBatchKMeans'
        self.do_spatial = do_spatial
        self.do_signals = do_signals
        self.denoise_ansc = Anscombe.wrap(self.denoise)

    def denoise(self, frames):
        coll = self.fit_transform(frames)
        return self.inverse_transform(coll)

    def fit_transform(self, frames, do_signals=None, do_spatial=None):
        coll = super().fit_transform(frames)

        if do_signals is None: do_signals = self.do_signals
        if do_spatial is None: do_spatial = self.do_spatial

        if do_signals:
            self.patches_ = self.update_signals()
        if do_spatial:
            self.patches_ = self.update_spatial()

        return self.patches_

    def nl_update_components(self, collection=None, field='signals', **kwargs):
        fsh = self.data_shape_
        if collection is None:
            collection = self.patches_
        out_samples = [np.zeros(getattr(c, field).shape, _dtype_) for c in collection]
        out_counts = np.zeros(len(collection), np.int)

        squares = make_grid(fsh[1:], self.Nhood, self.Nhood // 2)
        tstarts = set(c.sq[0].start for c in collection)
        tsquares = [(t, sq) for t in tstarts for sq in squares]

        def _is_local_patch(p, sqx):
            t0, sq = sqx
            tstart = p[0].start
            psq = p[1:]
            return (tstart == t0) & (slice_overlaps_square(psq, sq))

        loop = tqdm(tsquares, desc=f'Updating SVD {field}', disable=not self.verbose)

        for sqx in loop:
            samples = [getattr(c, field)
                       for c in collection if _is_local_patch(c.sq, sqx)]
            flat_samples = np.vstack(samples)
            loop.set_description(f'Updating window with {len(flat_samples)} {field}')
            approx = process_flat_collection(flat_samples, **kwargs)
            kstart = 0
            for i, c in enumerate(collection):
                if _is_local_patch(c.sq, sqx):
                    l = len(getattr(c, field))
                    out_samples[i] += approx[kstart:kstart + l]
                    out_counts[i] += 1
                    kstart += l
        loop.close()

        self.patches_ = [
            c._replace(**{field: x / (1e-7+cnt)})
            for c, x, cnt in zip(collection, out_samples, out_counts)
        ]
        return self.patches_

    def update_signals(self, collection=None, temporal_mode=None, **kwargs):

        if collection is None:
            collection = self.patches_

        if temporal_mode is None:
            temporal_mode = self.temporal_mode

        kwargs = dict(field='signals',
                      clustering_algorithm=self.clustering_algorithm,
                      n_clusters=self.n_clusters,
                      mode=temporal_mode,
                      tv_samples_per_cluster=self.tv_samples_per_cluster,
                      tv_sigma=self.tv_sigma,
                      tv_niters=self.tv_niters,
                      **kwargs)
        return self.nl_update_components(collection, **kwargs)

    def update_spatial(self, collection=None):
        if collection is None:
            collection = self.patches_
        #kwargs = dict(field='filters', Nhood=self.Nhood, mode='cluster-svd', n_clusters=1)
        kwargs = dict(field='filters', Nhood=self.Nhood, mode='cluster-svd')
        #kwargs = dict(field='filters', Nhood=self.Nhood, mode='flatTV-means')#, n_clusters=1)
        return self.nl_update_components(collection, **kwargs)


from skimage import transform as sktransform

class Multiscale_NL_Windowed_tSVD(NL_Windowed_tSVD):
    def ms_denoise(self, frames, *args, **kwargs):
        colls = self.ms_fit_transform(frames, *args, **kwargs)
        return self.ms_inverse_transform(colls)

    def ms_denoise_ansc(self, frames, *args, **kwargs):
        frames_t = Anscombe.transform(frames)
        out = self.ms_denoise(frames_t, *args, **kwargs)
        return Anscombe.inverse_transform(out)

    def ms_fit_transform(self, frames, nscales=3):
        colls = []
        self.full_data_shape_ = frames.shape
        #if patch_tsizes is None:
        #    patch_tsizes = (self.patch_tsize,)*len(patch_ssizes)
        #loop = tqdm(patch_ssizes, desc='Going through scales')
        loop = tqdm(range(nscales), desc='tSVD at different spatial scales')
        Nhood_orig = self.Nhood
        for j in loop:
            scale = 2**j
            coll = self.fit_transform(frames)
            colls.append((coll, scale, frames.shape))
            rec = self.inverse_transform(coll)
            #frames = frames - rec
            frames = sktransform.downscale_local_mean(frames - rec, factors=(1, 2, 2))
            self.Nhood = max(self.patch_ssize*4, self.Nhood // 2)
        loop.close()
        self.Nhood = Nhood_orig
        self.ms_patches_ = colls
        return colls

    def ms_inverse_transform(self, collections=None):
        if collections is None:
            collections = self.ms_patches_
        rec = np.zeros(self.full_data_shape_)
        nr,nc = self.full_data_shape_[1:]
        for coll, scale, fsh in collections:
            self.data_shape_ = fsh
            update = self.inverse_transform(coll)
            if scale > 1:
                #update = sktransform.rescale(update, (1,scale,scale))
                update = sktransform.rescale(update, (1,scale,scale), multichannel=False)
            nur,nuc = update.shape[1:]
            nrx = min(nr,nur)
            ncx = min(nc, nuc)
            rec[:, :nrx, :ncx] += update[:, :nrx, :ncx]
            #rec[:, :nrx, :ncx] = rec[:, :nrx, :ncx] + update[:, :nrx, :ncx]
            #rec[:, :nrx, :ncx] = rec[:, :nrx, :ncx] + update[:, :nrx, :ncx]
        return rec

    def ms_fit_transform_ansc(self, frames, *args, **kwargs):
        frames_t = Anscombe.transform(frames)
        return self.ms_fit_transform(frames_t, *args, **kwargs)

    def ms_inverse_transform_ansc(self,*args,**kwargs):
        return Anscombe.inverse_transform(self.ms_inverse_transform(*args,**kwargs))
