"""
Higher-level functions based on the rest of the code.
Can serve as usage examples, possibly outdated.
"""

def make_enh4(frames, pipeline=simple_pipeline_,
              labeler=percentile_label,
              kind='pca', nhood=5, stride=2, mask_of_interest=None,
              pipeline_kw=None,
              labeler_kw=None):
    from imfun import fseq
    #coll = signals_from_array_pca_cluster(frames,stride=2,dbscan_eps=0.05,nhood=5,walpha=0.5)
    if kind.lower()=='corr':
        coll = signals_from_array_correlation(frames,stride=stride,nhood=nhood,mask_of_interest=mask_of_interest)
    elif kind.lower()=='pca':
        coll = signals_from_array_pca_cluster(frames,stride=stride,nhood=nhood,
                                              mask_of_interest=mask_of_interest,
                                              ncomp=2,
        )
    else:
        coll = signals_from_array_avg(frames,stride=stride,patch_size=nhood*2+1,mask_of_interest=mask_of_interest)
    print('\nTime-signals, grouped,  processing (may take long time) ...')
    if pipeline_kw is None:
        pipeline_kw = {}
    pipeline_kw.update(labeler=labeler,labeler_kw=labeler_kw)
    coll_enh = process_signals_parallel(coll,pipeline=pipeline, pipeline_kw=pipeline_kw,)
    print('Time-signals processed, recombining to video...')
    out = combine_weighted_signals(coll_enh,frames.shape)
    fsx = fseq.from_array(out)
    print('Done')
    fsx.meta['channel']='-'.join(['newrec4',kind])
    return fsx


def make_enh5(dfof, twindow=50, nhood=5, stride=2, temporal_filter=3, verbose=False):
    from imfun import fseq
    amask = activity_mask_median_filtering(dfof, nw=7,verbose=verbose)
    nsf = mad_std(dfof, axis=0)
    dfof_denoised = svd_denoise_tslices(dfof,twindow, mask_of_interest=amask, temporal_filter=temporal_filter, verbose=verbose)
    mask_active = dfof_denoised > nsf
    mask_active = opening_of_closing(mask_active) + ndi.median_filter(mask_active,3)>0
    dfof_denoised2 = np.array([avg_filter_greater(f, 0) for f in dfof_denoised*mask_active])
    fsx = fseq.from_array(dfof_denoised2)
    if verbose:
        print('Done')
    fsx.meta['channel']='-'.join(['newrec5'])
    return fsx

def process_framestack(frames,min_area=9,verbose=False,
                       do_dfof_denoising = True,
                       baseline_fn = multi_scale_simple_baseline,
                       baseline_kw = dict(smooth_levels=(10,20,40,80)),
                       pipeline=simple_pipeline_,
                       labeler=percentile_label,
                       labeler_kw=None):
    """
    Default pipeline to process a stack of frames containing Ca fluorescence to find astrocytic Ca events
    Input: F(t): temporal stack of frames (Nframes x Nx x Ny)
    Output: Collection of three frame stacks containting ΔF/F0 signals, one thresholded and one denoised, and a baseline F0(t):
            fseq.FStackColl([fsx, dfof_filtered, F0])
    """
    from imfun import fseq
    if verbose:
        print('calculating baseline F0(t)')
    #fs_f0 = get_baseline_frames(frames[:],baseline_fn=baseline_fn, baseline_kw=baseline_kw)
    fs_f0 = calculate_baseline_pca_asym(frames[:],verbose=True,niter=20)
    fs_f0 = fseq.from_array(fs_f0)
    fs_f0.meta['channel'] = 'F0'

    dfof= frames/fs_f0.data - 1

    if do_dfof_denoising:
        if verbose:
            print('filtering ΔF/F0 data')
        dfof = patch_pca_denoise2(dfof, spatial_filter=3, temporal_filter=1, npc=5)
    fs_dfof = fseq.from_array(dfof)
    fs_dfof.meta['channel'] = 'ΔF_over_F0'

    if verbose:
        print('detecting events')
    ## todo: decide, whether we actually need the cleaning step.
    ## another idea: use dfof for detection to avoid FP, use dfof_cleaned for reconstruction because of better SNR?
    ##               but need to show that FP is lower, TP is OK and FN is low for this combo
    ## motivation:   using filters in dfof_cleaned introduces spatial correlations, which may lead to higher FP
    ##               (with low amplitude though). Alternative option would be to guess a correct amplitude threshold
    ##               afterwards
    ## note: but need to test that on real data, e.g. on slices with OGB and gcamp
    fsx = make_enh4(dfof,nhood=2,kind='pca',pipeline=pipeline,labeler=labeler,labeler_kw=labeler_kw)
    coll_ = EventCollection(fsx.data,min_area=min_area)
    meta = fsx.meta
    fsx = fseq.from_array(fsx.data*(coll_.to_filtered_array()>0),meta=meta)
    fscoll = fseq.FStackColl([fsx, fs_dfof, fs_f0])
    return fscoll


#from imfun.cluster import som

from matplotlib import animation
from skimage.feature import peak_local_max
from scipy import ndimage

def make_denoising_animation(frames, yhat,f0,  movie_name, start_loc=None,path=None):
    figh = plt.figure(figsize=(10,10))
    axleft = plt.subplot2grid((2,2), (0,0))
    axright = plt.subplot2grid((2,2), (0,1))

    L = len(frames)

    for ax in (axleft,axright):
        plt.setp(ax,xticks=[],yticks=[])
    axbottom = plt.subplot2grid((2,2), (1,0),colspan=2)
    fsh = frames[0].shape


    if (start_loc is None) :
        f = ndimage.gaussian_filter(np.max(yhat-f0,0),3)
        k = argmax(ravel(f))
        nrows,ncols = frames[0].shape
        loc = (k//ncols, k%ncols )
    else:
        loc = start_loc

    axleft.set_title('Raw (x10 speed)',size=14)
    axright.set_title('Denoised (x10 speed)',size=14)
    axbottom.set_title('Signals at point',size=14)
    low,high = np.percentile(yhat, (0.5, 99.5))
    low,high = 0.9*low,1.1*high
    h1 = axleft.imshow(frames[0],clim=(low,high),cmap='gray',animated=True)
    h2 = axright.imshow(yhat[0],clim=(low,high),cmap='gray',animated=True)
    axbottom.set_ylim(yhat.min(), yhat.max())
    plt.tight_layout()

    lhc = axright.axvline(loc[1], color='y',lw=1)
    lhr = axright.axhline(loc[0], color='y',lw=1)
    lhb = axbottom.axvline(0,color='y',lw=1)
    if path is None:
        locs = (loc + np.cumsum([(0,0)] + [np.random.randint(-1,2,size=2)*0.5+(1.25,1.25)
                                           for i in range(L)],axis=0)).astype(int)
    else:
        locs = []
        current_loc = array(start_loc)
        apath = asarray(path)
        keypoints = apath[:,2]
        for kf in range(L):
            ktarget = argmax(keypoints>=kf)
            target = apath[ktarget,:2][::-1] # switch from xy to rc
            kft = keypoints[ktarget]
            #print(target, current_loc)
            if kft == kf:
                v = 0
            else:
                v = (target-current_loc)#/(ktarget-kf)
                #vl = norm(v)
                v = v/(kft-kf)

            current_loc = current_loc + v
            locs.append(current_loc.copy())

    loc = locs[0].astype(int)
    xsl = (slice(None), loc[0], loc[1])
    lraw = axbottom.plot(frames[xsl], color='gray',label='Fraw')[0]#,animated=True)
    ly = axbottom.plot(yhat[xsl], color='royalblue',label=r'$\hat F$')[0]#,animated=True)
    lb = axbottom.plot(f0[xsl], color=(0.2,0.8,0.5),lw=2,label='F0')[0]#,animated=True)

    axbottom.legend(loc='upper right')
    nrows,ncols = fsh

    nn = np.concatenate([np.diag(ones(2,np.int)),-np.diag(ones(2,np.int))])

    loc = [loc]
    def _animate(frame_ind):
        #loc += randint(-1,2,size=2)

        #loc = locs[frame_ind]

        #f = ndimage.gaussian_filter(yhat[frame_ind]/f0[frame_ind],3)
        #lmx = peak_local_max(f)
        #labels, nlab = ndi.label(lmx)
        #objs = ndi.find_objects(labels)
        #peak = argmax([f[o].mean() for o in objs])
        #loc =  [(oi.start+oi.stop)/2 for oi in objs[peak]]

        #f = ndimage.gaussian_filter(yhat[frame_ind],3)
        #k = argmax([f[n[0]] for n in loc[0]+nn])
        #k = argmax(ravel(f))
        #loc[0] = (k//ncols, k%ncols )
        #loc[0] = loc[0] + nn[k]
        loc[0] = locs[frame_ind]
        loc[0] = asarray((loc[0][0]%nrows,loc[0][1]%ncols),np.int)
        xsl = (slice(None), loc[0][0], loc[0][1])
        h1.set_data(frames[frame_ind])
        h2.set_data(yhat[frame_ind])
        lraw.set_ydata(frames[xsl])
        ly.set_ydata(yhat[xsl])
        lb.set_ydata(f0[xsl])
        lhc.set_xdata(loc[0][1])
        lhr.set_ydata(loc[0][0])
        lhb.set_xdata(frame_ind)
        return [h1,h2,lraw,ly,lb,lhc,lhr]
    anim = animation.FuncAnimation(figh, _animate, frames=int(L), blit=True)
    Writer = animation.writers.avail['ffmpeg']
    w = Writer(fps=10,codec='libx264',bitrate=16000)
    anim.save(movie_name)
    return locs


from imfun.core import extrema
from numpy import fft
def roticity_fft(data,period_low = 100, period_high=5,npc=6):
    """
    Look for local areas with oscillatory dynamics in TXY framestack
    """
    L = len(data)
    if np.ndim(data)>2:
        data = data.reshape(L,-1)
    Xc = data.mean(0)
    data = data-Xc
    npc = min(npc, data.shape[-1])
    u,s,vh = svd(data,full_matrices=False)
    s2 = s**2/(s**2).sum()
    u = (u-u.mean(0))[:,:npc]
    p = (abs(fft.fft(u,axis=0))**2)[:L//2]
    nu = fft.fftfreq(len(data))[:L//2]
    nu_phys = (nu>1/period_low)*(nu<period_high)
    peak = 0
    sum_peak = 0
    for i in range(npc):
        pi = p[:,i]
        pbase = smoothed_medianf(pi, 5, 50)
        psmooth = smoothed_medianf(pi, 1,5)
        #pi = pi/psmooth-1
        lm = np.array(extrema.locextr(psmooth,x=nu,refine=True,output='max'))
        lm = lm[(lm[:,0]>1/period_low)*(lm[:,0]<1/period_high)]
        #peak_ = np.amax(lm[:,1])/psmooth[~nu_phys].mean()*s2[i]
        k = np.argmax(lm[:,1])
        nuk = lm[k,0]
        kx = np.argmin(np.abs(nu-nuk))
        peak_ = lm[k,1]/pbase[kx]*s2[i]
        #peak_ = np.amax(lm[:,1])#*s2[i]
        #print(amax(lm[:,1]),std(p[:,i]),peak_)
        sum_peak += peak_
        peak = max(peak, peak_)
    return sum_peak
