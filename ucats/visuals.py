from pylab import *
from collections import namedtuple

import pickle
from tqdm.auto import tqdm


from scipy import ndimage as ndi
from scipy import stats

import seaborn as sns

from skimage import feature as skfeature


import ucats
from . import utils as uutils
from . import baselines as ubase
from . import masks as umasks

#sys.path.append('/home/brazhe/proj/uCats')
#import astrocats as acats

#import powerlaw
from imfun import fseq, ui
from imfun import core
from imfun.filt import l1spline, l2spline

def make_seethrough_colormap(base_name='plasma', gamma=1.5,kcut=5):
    cm_base = plt.cm.get_cmap(base_name)
    cmx = cm_base(np.linspace(0,1,256))
    v = np.linspace(0,1,256)
    cmx[:,-1] = np.clip(2*(v/(0.15 + v))**gamma,0,1)
    cmx[:kcut,-1] = 0
    cdict = dict(red=np.array((np.arange(256)/255, cmx[:,0], cmx[:,0])).T,
                 green=np.array((np.arange(256)/255, cmx[:,1], cmx[:,1])).T,
                 blue= np.array((np.arange(256)/255, cmx[:,2], cmx[:,2])).T,
                 alpha= np.array((np.arange(256)/255, cmx[:,3], cmx[:,3])).T)
    cm = mpl.colors.LinearSegmentedColormap(base_name + '-x',  cdict, 256)
    return cm

def multi_savefig(fig, name, formats =('svg', 'png'), **kwargs):
    if 'bbox_inches' not in kwargs:
        kwargs['bbox_inches'] = 'tight'
    for f in formats:
        fig.savefig('.'.join([name,f]), **kwargs)


def viz_baseline_result(data, baselines, loc, ax=None):
    if ax is None:
        f,ax = subplots(1,1,figsize=(18,6))
    v = data[:,loc[0],loc[1]]
    b = baselines[:,loc[0],loc[1]]
    ax.plot(v,'+-', color='gray'); ax.plot(b, lw=2)
    ax.legend(('data', 'baseline'), )

def show_baseline_with_residuals(y, b, figsize=(18,6)):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=figsize,
                                  gridspec_kw=dict(width_ratios=(5,1), left=0.05, right=1-0.05, wspace=0.05))

    #bias = find_bias_mode(y[~ynan]-b, top_cut=85, smooth_factor=3)

    ynan = np.isnan(y)
    ax1.plot(y, color=(0.15,)*3, lw=0.5)
    ax1.plot(b, lw=2)
    ax2.hist(y[~ynan]-b[~ynan], 200, color=(0.33,)*3, orientation='horizontal');
    ax2.axhline(0, color='skyblue')
    return fig

def cdf_plot(y,ax=None,is_loglog=False, complementary=False):
    if ax is None:
        f,ax = subplots(1,1)
    yrank = np.sort(y)
    cp = np.arange(len(yrank),0,-1)/len(yrank)

    if not complementary:
        cp = 1-cp

    ax.step(yrank, cp, alpha=0.75)
    #title('Complementary CDF')
    if is_loglog:
        ax.set(xscale='log',yscale='log')


def recolor_axis(ax, color, spines='left'):
    plt.setp(ax.spines[spines], visible=True, color=color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y', colors=color)

def make_stained_area_mask(F0, th=0.25, margin=5,nlargest=5):
    pmask = umasks.opening_of_closing(np.mean(F0,0) >= th)
    pmask = umasks.threshold_object_size(pmask,50)
    #stained_area = ndi.binary_dilation(stained_area, iterations=2)
    #pmask = umasks.largest_region(pmask)# + stained_area #todo replace by N largest or filter size
    pmask = umasks.n_largest_regions(pmask, nlargest)
    #pmask = ndi.binary_fill_holes(uc.masks.largest_region(pmask))# + stained_area
    #stained_area = largest_region(stained_area)# + stained_area

    # to avoid artifacts due to motion correction, also clip 5px marings:
    mask = np.zeros(pmask.shape)
    mask[margin:-margin,margin:-margin]=True
    return (mask*pmask).astype(bool)


def add_scalebar(ax,length=25, height=1, scale=0.1,xy=None, unit='μm',color='w',
                 with_text=True, fontsize=None, xmargin=0.2):
    l = length/scale
    h = height/scale
    ax.set( xticks=[],yticks=[],frame_on=False)
    if xy is None:
        sh = ax.images[0].get_size()
        x = sh[1] - l - xmargin*sh[1]
        y = sh[0] - h - 0.1*sh[0]
        xy= x,y
    r = plt.Rectangle(xy,l,h, color=color )
    if with_text:
        ax.text(xy[0]+l/2,xy[1],s='{} {}'.format(length,unit),color=color,
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=fontsize)
    ax.add_patch(r)


def get_avg_positive_values(frames, th=0):
    return array([f[f>th].mean() if any(f>th) else 0 for f in frames])


def get_covered_area_per_frame(labels):
    return array([(ll>0).sum() for ll in labels])

def get_number_events_per_frame(labels):
    return array([len(unique(ll))-1 for ll in labels])

def nobjs(fm):
    labels, nlab = ndi.label(fm)
    return nlab


def avg_contiguous_area(fm):
    labels, nlab = ndi.label(fm)
    objs = ndi.find_objects(labels)
    areas = [np.sum(labels[o]==k+1) for k,o in enumerate(objs)]
    #areas = [a for a in areas if a > min_area]
    if len(areas):
        return np.mean(areas)
    else:
        return 0
    
    

def minute_maps(fs_enh, framerate, ncols=None, interval=60,
                vmax=0.5,vmin=0.025,cmap='plasma',fn=np.max, unit='s'):
    L = len(fs_enh)
    kstep = int(round(interval*framerate))
    kstarts = arange(0,L,kstep)
    maps = [fn(fs_enh[k:k+kstep],0) for k in kstarts]
    ui.group_maps(maps,ncols,
                  titles=[f'{k/framerate:0.0f}:{k/framerate+interval:0.0f} {unit}'
                          for k in kstarts],
                  figscale=3,
                  samerange=False,
                  imkw=dict(clim=(vmin, vmax), cmap=cmap),)
    tight_layout()
    return maps

def arrays_to_fstackcoll(*arrays, titles=None):
    fscoll =  fseq.FStackColl([fseq.from_array(x,) for x in arrays])
    if titles is not None:
        for stack,title in zip(fscoll.stacks, titles):
            stack.meta['channel'] = title
    return fscoll

def show_mean_frame_and_hist(frames):
    f, axs = plt.subplots(1,2,figsize=(12,5))
    mf = np.mean(frames,0)

    h = axs[0].imshow(ucats.clip_outliers(mf),  cmap='gray')
    plt.colorbar(h, ax=axs[0])
    axs[1].hist(ravel(frames), 100, log=True,histtype='step')
    axs[0].set_title('mean projection')
    axs[1].set_title('intensity distribution')

    return f

def show_mean_frame_and_dynamics(frames, show_contour=False):
    fig, ax = plt.subplots(1, 2,
                           gridspec_kw=dict(width_ratios=(1, 3)),
                           figsize=(12, 3))
    mf = np.mean(frames, axis=0)
    masks = ucats.utils.masks_at_levels(mf)
    #bright_mask  = mf > np.percentile(mf, 50)
    #v = np.array([f[bright_mask].mean() for f in frames])
    #lines = np.array([np.percentile(f, (25,50,75)) for f in frames])
    lines = np.array([[f[m].mean() for m in masks] for f in frames]).T
    ax[0].imshow(ucats.utils.clip_outliers(mf), cmap='gray')
    if show_contour:
        ax[0].contour(bright_mask, levels=[0.5],colors='g')
    for line in lines:
        ax[1].plot(line)
    plt.grid(True)
    return fig


def show_side_projs(frames, control_variable=None, nfaces=2, dt=1.0, dx=1.0,
                    cmap='plasma',
                    vmin=0, vmax=100*5,
                    aspect=5,
                    figsize=(12,4),
                    control_color = 'cyan'):

    proj_rows, proj_cols = [np.max(frames, axis=ax).T for ax in (1,2)]

    tv_full = np.arange(len(frames))*dt

    nrows = nfaces+ (1 if control_variable is not None else 0)

    height_ratios=(2,)*nfaces
    if control_variable is not None:
        height_ratios = height_ratios + (1,)

    cmap = plt.colormaps.get_cmap(cmap)

    fig, axs = plt.subplots(nrows,  1,sharex=True,
                            figsize=figsize,
                            gridspec_kw = dict(height_ratios=height_ratios,
                                               hspace=0.05))

    for ax, proj, lab in zip(axs[:nfaces], (proj_rows, proj_cols), ('Y', 'X')):
        h = ax.imshow(100*proj, cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
                      aspect='auto',
                      extent=(tv_full[0], tv_full[-1], proj.shape[0]*dx, 0))
        #ax.axis('off')
        ax.set(ylabel=lab + ' dim, μm')
        ui.plots.lean_axes(ax, hide=('top', 'right',))
        cb = plt.colorbar(h, ax=ax, label='%ΔF/F')

    if nfaces > 1:
        cb.ax.remove()

    if control_variable is not None:
        axs[-1].plot(tv_full[:len(control_variable)],
                     control_variable, color=control_color)
        ui.plots.lean_axes(axs[-1], hide=('top', 'right'))
        axs[-1].set(ylabel='speed, cm/s')
        # dirty trick to aling axes
        cb = plt.colorbar(h,ax=axs[-1])
        cb.ax.set_visible(False)

    axs[-1].set(xlabel='time, s')

    return fig

def get_clims(arr, pmin, pmax):
    #pmax_v = np.linspace(pmax, 100, 10)
    #levels = np.percentile(arr, [pmin]+list(pmax_v))
    vmin,vmax = np.percentile(arr[arr>0], (pmin, pmax))
    unique_levels = np.unique(arr[:min(len(arr),100)])
    uique_levels = unique_levels[unique_levels>0]
    if (vmin==vmax) or ((vmax-vmin) <= (unique_levels[1]-unique_levels[0])):
        vmin = vmin
        vmax = unique_levels[len(unique_levels)//2]

    return vmin, vmax






def save_as_movie(fs, video_name, percentile_clims=(1,99.5),colormap='gray'):
    from imfun import fseq, ui
    if isinstance(fs, fseq.FrameStackMono):
        fs = fseq.FStackColl([fs])

    #pmin, pmax = percentile_clims
    clims = [get_clims(stack.data, *percentile_clims) for stack in fs.stacks]
    print('Clims:', clims)
    p = ui.Picker(fs)
    p.clims = clims
    p.start()
    p.cmap=colormap
    ui.pickers_to_movie([p],
                        video_name,
                        writer=_writer,
                        codec=_codec,
                        fps=_fps,
                        cmap=colormap)
    plt.close('all')
    return

def trim_axes(ax, level=1, is_twin=False,
              dropped=True,
              hide = ('top', 'right')):
    """plot only x and y axis, not a frame for subplot ax"""

    if dropped:
        for key in ('top', 'right','bottom', 'left'):
            ax.spines[key].set_position(('outward', 6))

    for key in hide:
        ax.spines[key].set_visible(False)


    ax.get_xaxis().tick_bottom()
    if not is_twin:
        ax.get_yaxis().tick_left()
    else:
        ax.get_yaxis().tick_right()
    sides = [ax.get_xaxis(),]#ax.get_yaxis()
    if level > 1:
        for t in ax.get_xaxis().get_ticklabels():
            t.set_visible(False)
    if level > 2:
        for t in ax.get_yaxis().get_ticklabels():
            t.set_visible(False)
    if  level > 3:
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

def framewise_time_info_plot(dfof,
                             th1=1,
                             th_seg=(10,15,20),
                             framerate=1,
                             return_data=False,
                             suff='%ΔF/F',
                             min_area=10,
                             smoothing=3,
                             show_event_density=True,
                             line_colors=None,
                             show_means_instead_of_percentiles=False,
                             dx = None,
                             additional_data = None,
                             show_suprathreshold_means=False,
                             sum_active_for_mean = 100,
                             labels=None,
                             palette=None,
                          ):

    nrows = 4 if show_event_density else 3
    if additional_data is not None:
        nrows += 1

    #fig_height = 9 if show_event_density else 8
    fig_height = nrows*2 + 1

    gridspec_kw = dict(top=1-0.02, bottom=0.01,left=0.01,right=1-0.01,hspace=0.01,wspace=0.01)
    figh,axs = plt.subplots(nrows,1,sharex=True, figsize=(len(dfof)/600+3,fig_height), gridspec_kw=gridspec_kw)#,

    def _get_percentiles(f):
        if np.any(f>th1):
            return np.percentile(f[f>th1], (5,25,50,75,95))
        else:
            return np.nan*np.zeros(5)

    dfof_percentiles = np.array([_get_percentiles(f) for f in dfof]).T
    dfof_means = np.array([np.mean(f) for f in dfof])


    stained_area= np.sum(dfof.max(0) > 0)

    if dx is not None:
        stained_area_mm = stained_area*dx*dx/1e6 # stained area in mm^2
    else:
        dx = 1
        stained_area_mm = None

    print(stained_area, stained_area_mm, dx)
    if not np.iterable(th_seg):
        th_seg = [th_seg]

    if labels is None:
        #simple_labels = [np.array([umasks.threshold_object_size(f > th, min_area) for f in dfof]) for th in th_seg]
        simple_labels = [umasks.framewise_refine_masks(dfof>th,min_area) for th in th_seg]
        #simple_labels = [umasks.refine_mask_percentile()]
    else:
        simple_labels = labels

    def _get_active_area(lframe):
        return 100*np.sum(lframe)/stained_area

    def _get_event_area(lframe):
        return avg_contiguous_area(lframe)*dx*dx

    def _get_event_density(lframe):
        ed = nobjs(lframe)/stained_area
        if dx is not None:
            ed /= dx*dx/1e6 # convert to events/mm^2
        return ed



    areas = [np.array([_get_active_area(lf) for lf in ll]) for ll in simple_labels]
    event_areas = [np.array([_get_event_area(lf) for lf in ll]) for ll in simple_labels]
    event_density = [np.array([_get_event_density(lf) for lf in ll]) for ll in simple_labels] # in events/mm^2
    #event_numbers = [np.array([nobjs(lf) for lf in ll]) for ll in simple_labels]
    event_means = [np.array([np.mean(f[lx]) if np.sum(lx) > sum_active_for_mean else 0
                             for f,lx in zip(dfof, ll)]) for ll in simple_labels]

    L = len(dfof)
    tv = np.arange(L)/framerate

    if palette is None:
        palette = {'5-95': [(0.9,0.8,0.3), 0.2],
                   '25-75': ['orange', 0.2],
                   '50':[(0.5,0,0), 1]}


    if show_means_instead_of_percentiles:
        axs[0].plot(tv, dfof_means, 'k-', label='mean')
        if show_suprathreshold_means:
            for emv,th, lc in zip(event_means, th_seg, line_colors):
                axs[0].plot(tv, emv, 'm-', label=f'$\\langle$%ΔF/F$\\rangle$ | {suff} > {th}')
        #axs[0].legend(loc='upper center')
        axs[0].legend(loc='upper right')
    else:
        axs[0].fill_between(tv, dfof_percentiles[0],dfof_percentiles[-1],
                            color=palette['5-95'][0],
                            alpha=palette['5-95'][1],
                            label='5–95%')
        axs[0].fill_between(tv, dfof_percentiles[1],dfof_percentiles[-2],
                            color=palette['25-75'][0],
                            alpha=palette['25-75'][1],
                            label='25–75%')
        axs[0].plot(tv,dfof_percentiles[2],
                    color=palette['50'][0],
                    alpha=palette['50'][1],
                    lw=1,label='median')
        #axs[0].legend(loc='upper center',ncol=3)
        axs[0].legend(loc='upper right',ncol=3)

    #title('ΔF/F per frame'+suff)

    #title('ΔF/F per frame'+suff)
    smoother = lambda v: v
    if smoothing > 0:
        smoother = lambda v: l1spline(v, smoothing)

    if line_colors is None:
        #line_colors = np.linspace(0,1,len(th_seg))
        line_colors = [None]*len(th_seg) if len(th_seg) > 1 else 'k-'

    for av,th,lc in zip(areas,th_seg,line_colors):
        axs[1].plot(tv,smoother(av),color=lc,label=f'{th:1.1f}')

    #for av,th in zip(areas,th_seg):
    #    axs[1].plot(tv,av,label=f'{th:1.1f}')

    #for ea,th in zip(event_areas,th_seg):
    #    axs[2].plot(tv,ea,label=f'{th:1.1f}')

    for ea,th,lc in zip(event_areas,th_seg,line_colors):
        axs[2].plot(tv,smoother(ea),label=f'{th:1.1f}',color=lc)

    if show_event_density:
        for en,th,lc in zip(event_density,th_seg,line_colors):
            axs[3].plot(tv,smoother(en),label=f'{th:1.1f}',color=lc)
        density_ylabel = f'# segments' + '' if dx is None else '/mm^2'
        axs[3].set_ylabel(density_ylabel)
        axs[3].legend(title=f'segment density | {suff} > th',loc='upper right', ncol=len(th_seg))

    axs[0].set_ylabel(suff)
    axs[1].set_ylabel(f'% active pixels')
    axs[1].legend(title=f'% pixels | {suff} > th',loc='upper right', ncol=len(th_seg))

    axs[2].set_ylabel(f'seg. area (px)')
    axs[2].legend(title=f'$\\langle$segment area$\\rangle$ | {suff} > th',loc='upper right', ncol=len(th_seg))


    #axs[3].set_ylabel(f' # seg.')
    #axs[3].legend(title=f'# segments | {suff} > th',loc='upper right', ncol=len(th_seg))

    if additional_data is not None:
        ad = additional_data
        axs[-1].plot(ad['time'], ad['value'], label=ad['label'], color=ad['color'])
        axs[-1].legend(loc='upper right')

    #axs[2].set_ylabel('number of segmented events')
    axs[-1].set_xlabel('seconds')
    axs[-1].set_xlim(tv[0],tv[-1])
    for ax in axs:
        ui.plots.lean_axes(ax, hide=('top','right'))

    #plt.subplots_adjust(top=1-0.02, bottom=0.01,left=0.01,right=1-0.01,hspace=0.01,wspace=0.01)

    if not return_data:
        return figh
    else:
        return figh, dfof_percentiles, areas, event_areas, event_density


def plot_fw_stats(fw_stats,
                  tslice=None,
                  zero_time=True,
                  do_trim=True,
                  variables = ('avg ΔF/F', 'active area [%]',
                               'avg segment area [um^2]', 'segments/mm^2',
                               'speed [cm/s]')):

    variables = [v for v in variables if v in fw_stats.columns]
    nrows = len(variables)

    if tslice is not None:
        fw_stats = fw_stats.iloc[tslice]

    if zero_time:
        fw_stats = fw_stats.copy()
        fw_stats['time (s)'] -= fw_stats['time (s)'].iat[0]


    fig, axs = plt.subplots(nrows,1,
                            sharex=True,
                            gridspec_kw=dict(hspace=0.5, right=1-0.05),
                            figsize=(nrows*16/2.5/5,19/2.5))

    tv = fw_stats['time (s)']

    for ax, yv in zip(axs, variables):
        ax.plot('time (s)', yv, data=fw_stats, color=palette[yv])
        ax.set_ylabel(yv,fontsize=9)


    axs[-1].set_xlabel('time, s')

    if do_trim:
        sns.despine(fig, trim=True )
    return fig
