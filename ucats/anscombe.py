import numpy as np


class Anscombe:
    """Variance-stabilizing transformation"""

    vmin = 2*np.sqrt(3/8)

    @staticmethod
    def transform(data):
        return 2 * (np.maximum(data,0) + 3/8)**0.5

    @staticmethod
    def inverse_transform(D):
        D = np.maximum(D, Anscombe.vmin + 2e-16)
        Dsq = D*D
        a = np.sqrt(3/2)
        return Dsq/4 - 1/8  + (a/4)/D - (11/8)/Dsq + a*(5/8)/(Dsq*D)

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
