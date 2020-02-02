import numpy as np

from scipy import ndimage as ndi


from skimage.feature import register_translation
from skimage import transform as skt
from imfun import core

#from fastdtw import fastdtw
from imfun import core

def apply_warp_path(v, path):
    path = np.array(path)
    return np.interp(np.arange(len(v)), path[:,0], v[path[:,1]])

def interpolate_path(path,L):
    return np.interp(np.arange(L), path[:,0],path[:,1])


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
