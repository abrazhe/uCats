import numpy as np


class Anscombe:
    "Variance-stabilizing transformation"
    vmin = 2*np.sqrt(3/8)
    @staticmethod
    def transform(data):
        return 2 * (data + 3/8)**0.5

    @staticmethod
    def inverse_transform(D):
        if D <= self.vmin:
            return 0
        else:
            Dsq = D**2
            a = np.sqrt(3/2)
            return Dsq/4 - 1/8  + 0.25*a/D - 11/8/Dsq + (5/8)*a/Dsq/D
