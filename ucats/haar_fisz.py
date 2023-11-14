import numpy as np

from numba import jit


@jit
def haar_step(v, out=None, with_fisz=False):
    L = len(v)
    H = L//2
    if out is None:
        out= np.zeros(len(v),dtype=np.float32)
    for i in range(H):
        d = (v[2*i]-v[2*i+1])/2
        s = (v[2*i+1]+v[2*i])/2
        if with_fisz:
            d = 0 if s == 0 else d/s**0.5
        out[i] = d
        out[H+i]=s
    return out

@jit
def ihaar_step(u, out=None, with_fisz=False):
    L = len(u)
    H = L//2
    if out is None:
        out = np.zeros(len(u), dtype=np.float32)
    for i in range(H):
        d,s = u[i], u[H+i]
        if with_fisz:
            d = d*s**0.5
        out[2*i] = s+d
        out[2*i+1] = s-d
    return out

@jit
def vec_copy(v,u):
    L = len(v)
    for i in range(L):
        u[i] = v[i]
    return

@jit
def haar(v, level=None, with_fisz=False):
    out = np.zeros_like(v)
    v_temp = np.zeros_like(v)
    vec_copy(v, v_temp)
    L = len(v)
    H = L

    # if using Fisz transform, clip values below 0
    if with_fisz:
        for i in range(L):
            if v_temp[i] < 0:
                v_temp[i] = 0


    if level is None:
        level = int(np.floor(np.log2(L)))

    for j in range(level):
        tmp = haar_step(v_temp[L-H:], out[L-H:], with_fisz=with_fisz)
        vec_copy(tmp, v_temp[L-H:])
        H = H//2
        if H < 2:
            break
    return out, j+1

@jit
def ihaar(v, level=None, with_fisz=False):
    out = np.zeros_like(v)
    v_temp = np.zeros_like(v)
    vec_copy(v, v_temp)
    L = len(v)

    # if using Fisz transform, clip values below 0
    if with_fisz:
        for i in range(L):
            if v_temp[i] < 0:
                v_temp[i] = 0


    if level is None:
        level = int(np.floor(np.log2(L)))

    for j in range(level):
        H = L//2**(level-j-1)
        tmp = ihaar_step(v_temp[L-H:], out[L-H:], with_fisz=with_fisz)
        vec_copy(tmp, v_temp[L-H:])

    return out

def fisz(v, level=None):
    return ihaar(haar(v, level, with_fisz=True))


def fisz(v, level=None):
    vhat, level = haar(v, level, with_fisz=True)
    return ihaar(vhat,level)

def ifisz(u, level=None):
    uhat,level = haar(u, level)
    return ihaar(uhat,level,with_fisz=True)

class HaarFisz:
    """
    Variance-stabilizing transformation via Haar-Fisz transform
    based on FryzlewiCz and Nason 2004, DOI:10.1198/106186004X2697
    """
    @staticmethod
    def transform(data):
        """Apply forward Haar-Fisz variance-stabilizing transform
        negative input values will be clipped at zero
        """
        return np.apply_along_axis(fisz, 0, data)

    @staticmethod
    def inverse_transform(data):
        return np.apply_along_axis(ifisz, 0, data)

    @staticmethod
    def wrap_input(func):
        def _wrapper(data, *args, **kwargs):
            data_t = HaarFisz.transform(data)
            return func(data_t, *args, **kwargs)
        return _wrapper

    @staticmethod
    def wrap_output(func):
        def _wrapper(*args, **kwargs):
            out_t = func(*args, **kwargs)
            return HaarFisz.inverse_transform(out_t)
        return _wrapper

    @staticmethod
    def wrap(func):
        return HaarFisz.wrap_output(HaarFisz.wrap_input(func))
