import numpy as np


class Anscombe:
    """Variance-stabilizing transformation"""

    @staticmethod
    def transform(data):
        return 2 * (data + 3 / 8) ** 0.5

    @staticmethod
    def inverse_transform(tdata):
        tdata_squared = tdata ** 2
        return tdata_squared / 4 + np.sqrt(3 / 2) * (
                    1 / (4 * tdata) + 5 / (8 * tdata ** 3)) - 11 / 8 / tdata_squared - 1 / 8
