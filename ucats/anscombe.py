import numpy as np


class Anscombe:
    "Variance-stabilizing transformation"
    vmin = 2*np.sqrt(3/8)
    @staticmethod
    def transform(data):
        return 2 * (data + 3/8)**0.5

    @staticmethod
    def inverse_transform(D):
        D = np.maximum(D, Anscombe.vmin + 2e-16)
        Dsq = D*D
        a = np.sqrt(3/2)
        return Dsq/4 - 1/8  + (a/4)/D - (11/8)/Dsq + a*(5/8)/(Dsq*D)
