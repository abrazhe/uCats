import numpy as np

from numba import jit

_vmin_ = 2*np.sqrt(3/8)

@jit(nopython=True)
def approximate_inverse_transform(d):
    d = np.maximum(d, _vmin_ + 2e-16)
    dsq = d*d
    a = np.sqrt(3/2)
    return dsq/4 - 1/8  + (a/4)/d - (11/8)/dsq + a*(5/8)/(dsq*d)




class Anscombe:
    """Variance-stabilizing transformation"""

    vmin = _vmin_

    @staticmethod
    def transform(data):
        return 2 * (np.maximum(data,0) + 3/8)**0.5

    @staticmethod
    def inverse_transform(D):
        if np.iterable(D):
            return Anscombe.inverse_transform_jit(D)
        else:
            return approximate_inverse_transform(D)

    @staticmethod
    @jit(nopython=True)
    def inverse_transform_jit(Din, Dout=None):
        sh = Din.shape
        if Dout is None:
            Dout = np.zeros_like(Din)
        Dout = np.ravel(Dout)
        Din = np.ravel(Din)
        L = len(Din)
        for i in range(L):
            Dout[i] = approximate_inverse_transform(Din[i])
        return Dout.reshape(sh)


    @staticmethod
    def wrap_input(func):
        def _wrapper(data, *args, **kwargs):
            data_t = Anscombe.transform(data)
            return func(data_t, *args, **kwargs)
        return _wrapper

    @staticmethod
    def wrap_output(func):
        def _wrapper(*args, **kwargs):
            out_t = func(*args, **kwargs)
            return Anscombe.inverse_transform(out_t)
        return _wrapper

    @staticmethod
    def wrap(func):
        return Anscombe.wrap_output(Anscombe.wrap_input(func))
