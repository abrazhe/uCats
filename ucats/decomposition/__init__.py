import numpy as np

from scipy import ndimage as ndi

import itertools as itt

#from ..detection1d import simple_pipeline_
from ..masks import threshold_object_size
from ..utils import (avg_filter_greater, mad_std, smoothed_medianf, find_bias)

from .svd_utils import min_ncomp, pca_flip_signs, svd_flip_signs, tanh_step
from ..globals import _dtype_

from .Windowed_tSVD import superpixel_tSVD, Windowed_tSVD, SVD_patch
#from .ndWindowed_tSVD import ndWindowed_tSVD, ndSVD_patch
from .HOSVD import HOSVD, Windowed_tHOSVD, HOSVD_patch

from . import ndWindowed_tSVD

def DyCA(data, min_ncomp=2, eig_threshold = 0.98, vebose=True):
    "Given data of the form (time, sensors) returns the DyCA projection and projected data with eigenvalue threshold eig_threshold"
    derivative_data = np.gradient(data,axis=0,edge_order=1) #get the derivative of the data
    L = data.shape[0] #for time averaging
    #construct the correlation matrices
    C0 = data.T @ data / L
    C1 = derivative_data.T @ data / L
    C2 = derivative_data.T @ derivative_data / L

    try:
        eigvalues, eigvectors = sp.linalg.eig(C1 @ np.linalg.inv(C0) @ C1.T, C2)
        eigvalues = np.abs(eigvalues).real
        #print('Any eigenvalues > 1? ', np.sum(eigvalues > 1))
        eigvectors = eigvectors[:,eigvalues <= 1]
        eigvalues = eigvalues[eigvalues <= 1] # eigenvalues > 1 are artifacts of singularity of C0
        if not len(eigvalues):
            return np.nan, np.nan, np.zeros(data.shape[1])+np.nan
        ksort = np.argsort(eigvalues)[::-1]
        eigvalues = eigvalues[ksort]
        eigvectors = eigvectors[:,ksort]
        #print(min_ncomp, len(eigvalues), len(eigvectors))
        min_ncomp = min(min_ncomp, len(eigvalues))

        eig_threshold = min(eigvalues[min_ncomp-1], eig_threshold)
        eigvectors = eigvectors[:,np.array(eigvalues > eig_threshold)]
        if eigvectors.shape[1] > 0:
            #C3 = np.matmul(np.linalg.inv(C1), C2)
            C3 = np.linalg.inv(C1)@C2
            proj_mat = np.concatenate((eigvectors, np.apply_along_axis(lambda x: C3@x, 0, eigvectors)),axis=1)
        else:
            raise ValueError('No generalized eigenvalue fulfills threshold!')
        return proj_mat, data@proj_mat, eigvalues
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.zeros(data.shape[1])+np.nan
