import numpy as np

from scipy import ndimage as ndi
from sklearn import linear_model

from matplotlib import pyplot as plt

from imfun.core import extrema
from .patches import extract_random_column

from .globals import _dtype_


def _simple_stats(x):
    "Just return mean and variance of a sample"
    return (x.mean(), x.var())
    #mu = x.mean()
    #sigmasq = np.var(x[np.abs(x-mu)<3*np.std(x)])
    #return mu, sigmasq


# TODO: may be scrambled data are the best for robust gain,offset estimates???
def estimate_offset2(frames, smooth=None, nsteps=100, with_plot=False):
    mu = np.median(frames)
    sigma = np.std(np.concatenate((frames[frames <= mu], mu - frames[frames <= mu])))
    print('mu', mu, 'sigma', sigma)
    biases = np.linspace(mu - sigma/4, mu + sigma/4, nsteps)
    db = biases[1] - biases[0]
    v = np.array([np.mean(frames < n) for n in biases])
    if smooth is not None:
        dv = ndi.gaussian_filter1d(v, smooth, order=1)
        offset = biases[np.argmax(dv)]
    else:
        for smooth in np.arange(0.1 * db, 100 * db, 0.5 * db):
            dv = ndi.gaussian_filter1d(v, smooth, order=1)
            peaks = extrema.locextr(dv, x=biases, output='max')
            if len(peaks) < 1:
                continue
            offset = peaks[0][0]
            if len(peaks) <= 1:
                break
        if not len(peaks):
            offset = biases[np.argmax(dv)]
    if with_plot:
        plt.figure()
        plt.plot(biases, v, '.-')
        plt.plot(biases, (dv - dv.min()) / (dv.max() - dv.min() + 1e-7))
        plt.axvline(offset, color='r', ls='--')

    return offset


# TODO: eventually move to μCats
def estimate_gain_and_offset(frames,
                             patch_width=10,
                             npatches=int(1e5),
                             ntries=200,
                             with_plot=False,
                             save_to=None,
                             return_type='mean',
                             phigh=95,
                             verbose=False):
    """
    Estimage gain and offset parameters used by the imaging system
    For a given stack of frames, extract many small cubic patches and then fit a line
    as variance ~ mean. The slope of the line is the gain, the intercept of the X axis
    is the offset or system bias.

    Parameters:
     - frames : TXY 3D array-like, a stack of frames to analyze
     - patch_width: scalar int, side of a cubic patch to randomly sample from frames (default: 20)
     - npatches: scalar int, number of random patches to draw from the frames (default: 100,000)
     - ntries: scalar int, number of patch ensembles to fit at a time, next parameters will be pooled (default: 100)
     - with_plot: bool, whether to make a plot of the mean-variance line fit
     - save_to: string, name of a file to save the plot to, has no effect if with_plot is False (default: None, i.e. don't save)
     - return_type: {'mean','min','median','ransac'} -- which estimate of the line fit to return
    """

    pxr = np.array([
        _simple_stats(extract_random_column(frames, patch_width)) for i in range(npatches)
    ])

    x_uniq = np.unique(pxr[:,0])
    cut_uniq = x_uniq[min(100, len(x_uniq)-1)]
    cut_perc = np.percentile(pxr[:, 0], phigh)

    cut = max(cut_uniq, cut_perc)

    pxr = pxr[pxr[:, 0] < cut]
    vm, vv = pxr.T

    gains = np.zeros(ntries, _dtype_)
    offsets = np.zeros(ntries, _dtype_)

    for i in range(ntries):
        vmx, vvx = np.random.permutation(pxr, )[:npatches // 10].T
        p = np.polyfit(vmx, vvx, 1)
        #regressor = linear_model.RANSACRegressor()
        #regressor.fit(vmx[:,None], vvx)
        #re = regressor.estimator_
        gain, intercept = p
        offset = -intercept / gain
        gains[i] = gain
        offsets[i] = offset

    regressorg = linear_model.RANSACRegressor()
    regressorg.fit(vm[:, None], vv)
    gainx = regressorg.estimator_.coef_
    interceptx = regressorg.estimator_.intercept_
    offsetx = -interceptx / gainx
    results = {
        'min': (np.amin(gains), np.amin(offsets)),
        'mean': (np.mean(gains), np.mean(offsets)),
        'median': (np.median(gains), np.median(offsets)),
        'ransac': (gainx, offsetx)
    }

    if verbose:
        print('RANSAC: Estimated gain %1.2f and offset %1.2f' % (results['ransac']))
        print('ML: Estimated gain %1.2f and offset %1.2f' % (results['mean']))
        print('Med: Estimated gain %1.2f and offset %1.2f' % (results['median']))

    min_gain, min_offset = np.amin(gains), np.amin(offsets)

    if with_plot:
        fmt = ' (%1.2f, %1.2f)'

        f, axs = plt.subplots(1,
                              3,
                              figsize=(12, 4),
                              gridspec_kw=dict(width_ratios=(2, 1, 1)))
        h = axs[0].hexbin(vm, vv, bins='log', cmap='viridis', mincnt=5)
        xlow, xhigh = vm.min(), np.percentile(vm, 99)
        ylow, yhigh = vv.min(), np.percentile(vv, 99)
        xfit = np.linspace(vm.min(), xhigh)
        linefit = lambda gain, offset: gain * (xfit-offset)
        axs[0].axis((xlow, xhigh, ylow, yhigh))
        line_fmts = [('--', 'skyblue'), ('-', 'g'), ('--', 'm')]
        hist_kw = dict(density=True, bins=25, color='slategray')
        axs[1].hist(gains, **hist_kw)
        axs[2].hist(offsets, **hist_kw)
        plt.setp(axs[1], title='Gains')
        plt.setp(axs[2], title='Offsets')
        for key, lp in zip(('min', 'mean', 'ransac'), line_fmts):
            gain, offset = results[key]
            axs[0].plot(xfit,
                        linefit(gain, offset),
                        ls=lp[0],
                        color=lp[1],
                        label=key + ': ' + fmt % (gain, offset))
            axs[1].axvline(gain, color=lp[1], ls=lp[0])
            axs[2].axvline(offset, color=lp[1], ls=lp[0])

        axs[0].legend(loc='upper left')
        plt.setp(axs[0],
                 xlabel='Mean',
                 ylabel='Variance',
                 title='Mean-Variance for small patches')
        plt.colorbar(h, ax=axs[0])
        if save_to is not None:
            f.savefig(save_to)
    return results[return_type]

def estimate_clip_level(frames, max_use_frames=5000):
    gain, offset = estimate_gain_and_offset(frames, phigh=99, ntries=300)
    u_raw = np.unique(frames[:min(len(frames),max_use_frames)])
    return min(u_raw[u_raw > offset])


def convert_from_varstab(df, b):
    "convert fluorescence signals separated to fchange and f baseline from 2*√f space"
    bc = b**2 / 4
    dfc = (df**2 + 2*df*b) / 4
    return dfc, bc
