import numpy as np
import itertools as itt

from ..decomposition import min_ncomp


def dmdf_new(X, Y=None, r=None, sort_explained=False):
    if Y is None:
        Y = X[:, 1:]
        X = X[:, :-1]
    U, sv, Vh = np.linalg.svd(X, False)
    if r is None:
        r = min_ncomp(sv, X.shape) + 1
    sv = sv[:r]
    V = Vh[:r].conj().T
    Uh = U[:, :r].conj().T
    B = Y @ V @ (np.diag(1 / sv))

    Atilde = Uh @ B
    lam, W = np.linalg.eig(Atilde)
    Phi = B @ W

    #print(Vh.shape)
    # approx to b
    def _bPOD(i):
        alpha1 = np.diag(sv[:r]) @ Vh[:r, i]
        return np.linalg.lstsq(Atilde @ W, alpha1, rcond=None)[0]

    #bPOD = _bPOD(0)
    stats = (None, None)
    if sort_explained:
        #proj_dmd = Phi.T.dot(X)
        proj_dmd = np.array([_bPOD(i) for i in range(Vh.shape[1])])
        dmd_std = proj_dmd.std(0)
        dmd_mean = abs(proj_dmd).mean(0)
        stats = (dmd_mean, dmd_std)
        kind = np.argsort(dmd_std)[::-1]
    else:
        kind = np.arange(r)[::-1]    # from slow to fast
    Phi = Phi[:, kind]
    lam = lam[kind]
    #bPOD=bPOD[kind]
    return lam, Phi    #,bPOD,stats


def _patch_denoise_dmd(data,
                       stride=2,
                       nhood=5,
                       npc=None,
                       temporal_filter=None,
                       mask_of_interest=None):
    sh = data.shape
    L = sh[0]

    #if mask_of_interest is None:
    #    mask_of_interest = np.ones(sh[1:],dtype=np.bool)
    out = np.zeros(sh, _dtype_)
    counts = np.zeros(sh[1:], _dtype_)
    if mask_of_interest is None:
        mask = np.ones(counts.shape, bool)
    else:
        mask = mask_of_interest
    Ln = (2*nhood + 1)**2

    #preproc = lambda y: core.rescale(y)

    #tmp_signals = np.zeros()
    tv = np.arange(L)

    def _next_x_prediction(X, lam, Phi):
        Xn = X.reshape(-1, 1)
        b = lstsq(Phi, Xn, rcond=None)[0]
        Xnext = (Phi @ np.diag(lam) @ b.reshape(-1, 1)).real
        return Xnext

    #    return Xnext.T.reshape(f.shape)

    def _process_loc(r, c):
        sl = (slice(r - nhood, r + nhood + 1), slice(c - nhood, c + nhood + 1))
        tsl = (slice(None), ) + sl

        patch = data[tsl]

        X = patch.reshape(L, -1).T
        #print(patch.shape, X.shape)

        lam, Phi = dmdf_new(X, r=npc)

        rec = np.array([_next_x_prediction(f, lam, Phi) for f in X.T])

        #print(out[tsl].shape, patch.shape, rec.shape)
        out[tsl] += rec.reshape(*patch.shape)

        score = 1.0
        counts[sl] += score

    for r in itt.chain(range(nhood, sh[1] - nhood, stride), [sh[1] - nhood]):
        for c in itt.chain(range(nhood, sh[2] - nhood, stride), [sh[2] - nhood]):
            sys.stderr.write('\rprocessing location (%03d,%03d), %05d/%d' %
                             (r, c, r * sh[1] + c + 1, np.prod(sh[1:])))
            if mask[r, c]:
                _process_loc(r, c)
    out = out / (1e-12 + counts[None, :, :])
    for r in range(sh[1]):
        for c in range(sh[2]):
            if counts[r, c] == 0:
                out[:, r, c] = 0
    return out
