#!/usr/bin/env python

## TODO: add second pass (correction) as an option

import os,sys
import h5py

import argparse
import pickle
import json

from functools import partial,reduce
import operator as op

import numpy as np
from numpy import *

from scipy import ndimage,signal
from scipy import ndimage as ndi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from skimage.external import tifffile

import pandas as pd

from imfun import fseq,core,ui
from imfun import multiscale
from imfun import ofreg
from imfun.ofreg import stackreg, imgreg
from imfun.external import czifile

from imfun.filt.dctsplines import l2spline, l1spline

from imfun.multiscale import atrous
from imfun import components



import μCats as ucats

import socket
def my_hostname():
    return socket.gethostname()

_hostname_ = my_hostname()

if _hostname_ in ['gamma','xps','delta']:
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'




def main():
    parser = argparse.ArgumentParser(description="""Astrocats: process a Ca imaging record file, detect Ca transients""")
    argdict =  {
        "name": dict(nargs='?'),
        '-j': ('--json', dict(default=None, help="json file with default parameters")),
        '-p': ('--pretend', dict(action='store_true',help="Pretend mode: don't do anything, dry-run")),
        '-m': ('--stab-model', dict(action='append', metavar = ("STABMODEL","PARAMETER"),
                               nargs = '+',
                               #choices = ['shifts', 'softmesh', 'affine', 'Greenberg-Kerr'],
                               help='add movement model to use for stabilization;\
                               available models: {shifts, mslkp, msclg, affine, Greenberg-Kerr, homography}')),
        '-n': ('--ncpu', dict(default=4, type=int, help="number of CPU cores to use")),
        #'--record': dict(default=None, help='record within file to use (where applicable)'),
        '-v': ('--verbose', dict(action='count', default=0, help='increment verbosity level')),
        '--morphology-channel': dict(default=0,type=int, help='color channel to use for motion correction (if several)'),
        '--ca-channel':dict(default=1, type=int, help='color channel with Ca-dependent fluorescence'),
        '--with-motion-movies': dict(action='store_true'),
        '--with-detection-movies': dict(action='store_true'),
        '--suff': dict(default='', help="optional suffix to append to saved registration recipe"),
        '--fps': dict(default=25,type=float,help='fps of exported movie'),
        '--pca-denoise': dict(action='store_true'),
        '--codec': dict(default='libx264', help='movie codec'),
        '--writer': dict(default='ffmpeg', help='movie writer'),
        '--no-save-enh': dict(action='store_true'),
        #'--save-denoised-dfof': dict(action='store_true'),
        '--no-events': dict(action='store_true'),
        '--skip-existing': dict(action='store_true'),
        '--no-skip-dark-areas':dict(action='store_false'),
        #'--do-oscillations': dict(action='store_true'),
        '--detection-do-adaptive-median-filter': dict(action='store_true',
                                                      help='whether to pre-filter data with an adaptive median filter'),
        '--detection-low-percentile':dict(default=5, type=float,
                                          help='lower values detect less FPs, higher values make more detections'),
        '--detection-spatial_filter':dict(default=3, type=int, help='median smooth size for spatial SVD components'),
        #'--detection-tau-smooth':dict(default=2., type=float,
        #                              help='smoothing in detection, make larger for less false positives,\
        #                              make smaller for detection of smaller events'),
        '--detection-loc-nhood':dict(default=5,type=int),
        '--detection-loc-stride':dict(default=2,type=int),
        '--detection-use-clusters':dict(default=False, type=bool),
        '--detection-temporal-window':dict(default=600,type=int),
        '--detection-smoothed-reconstruction':dict(default=1,type=bool),
        '--detection-min-components': dict(default=3,type=int,help='minimum number of SVD components to analyse'),
        '--detection-no-variance-stabilization':dict(action='store_true'),
        '--detection-no-second-pass':dict(action='store_true'),
        '--baseline-smoothness': dict(default=150,type=float),
        '--event-segmentation-threshold':dict(default=0.025, type=float, help='ΔF/F level at which separate nearby events'),
        #'--event-segmentation-threshold':dict(default=0.05, type=float, help='ΔF/F level at which separate nearby events'),
        '--event-peak-threshold':dict(default=0.085, type=float,help='event must contain a peak with at least this ΔF/F value'),
        '--event-min-duration':dict(default=3, type=int,help='event must be at least this long (frames)'),
        '--event-min-area':dict(default=16, type=int, help='event must have at least this projection area (pixels)'),
        '--bitrate':dict(default=32000,type=float, help='bitrate of exported movie'),
        }

    for arg,kw in list(argdict.items()):
        if isinstance(kw, dict):
            parser.add_argument(arg,  **argdict[arg])
        else:
            parser.add_argument(arg, kw[0], **kw[1])

    args = parser.parse_args()

    # some stupid workaround for store_true store_false
    args.detection_do_second_pass = not args.detection_no_second_pass
    args.detection_do_variance_stabilization = not args.detection_no_variance_stabilization
    args.do_save_enh = not args.no_save_enh

    print('Will do second pass?', args.detection_do_second_pass)


    if args.stab_model is None:
        args.stab_model = ['msclg']
    for m in args.stab_model:
        if not isinstance(m, str) and len(m)>2:
            params = json.loads(m[2])
            m[2] = params


    # override everything if json parameter is given
    if args.json :
        with open(args.json,'rt') as jsonfile:
            pars = json.load(jsonfile)
            for key in pars:
                setattr(args, key, pars[key])

    if args.name.endswith('.lif'):
        process_lif_file(args.name,args)
    else:
        print("Processing", args.name)
        # I.   Load record(s)
        fs = load_record(args.name, ca_channel=args.ca_channel)
        process_record(fs, args.name, '', args)
    print('processing done')
    return # from main


import gc
def process_lif_file(fname, args,min_frames=600):
    ###
    # pip install javabridge
    # pip install python-bioformats
    # conda install xmltodict
    ###
    import io_lif
    print('------------------------------------------------------------------------------------------')
    print('Processing file',fname)
    metaxml,lif_recs = io_lif.load_meta_records(fname) # load record descriptions
    for rec in lif_recs:
        print(rec)
        if rec.get_size('T') > min_frames:
            print('Will process %s:%s'%(fname, rec.name))
            safename = rec.name.replace('/','_')
            fs = rec.load_timelapse(fname)
            fs.meta['file_path'] = fname
            try:
                process_record(fs,fname,safename,args)
            except Exception as e_:
                print("Coulnd't process %s because of "%safename, e_)
            del fs
            plt.close('all')
            gc.collect()
    io_lif.javabridge.kill_vm()

# def convert_from_varstab(df,b):
#     "convert fluorescence signals separated to fchange and f baseline from 2*√f space"
#     bc = b**2/4
#     dfc =  (df**2 + 2*df*b)/4
#     return dfc, bc

def process_record(fs, fname, series, args):
    nametag = '-'.join((fname,series,args.suff))
    print('nametag is:', nametag)
    h5f  = None
    # II.  Stabilize motion artifacts
    fsc,_ = stabilize_motion(fs, args,nametag)

    if isinstance(fsc, fseq.FStackColl) and len(fsc.stacks) > 1:
        fsc = fsc.stacks[args.ca_channel]

    if args.no_events:
        return

    # I. -- Adaptive median filtering input data --
    frames = fsc.data.astype(float32)
    if args.detection_do_adaptive_median_filter:
        print("Performing adaptive median filtering of the raw fluorescence signal")
        frames = ucats.adaptive_median_filter(fsc.data.astype(float32),
                                              th=5, tsmooth=1, ssmooth=5)
        frames = frames.astype(float32)
    else:
        frames = ucats.clip_outliers(frames,0.05, 99.95).astype(np.float32)


    # II. Process data
    benh = None
    detected_name = nametag+'-detected.h5'
    baseline_name = nametag+'-baseline.pickle'
    if os.path.exists(detected_name):
        #h5f = h5py.File(detected_name,'r')
        fsx = fseq.from_hdf5(detected_name)
        h5f = fsx.h5file
        print('loading existing results of event detection:', detected_name)
        if os.path.exists(baseline_name):
            print('loading existing fluorescence baseline frames')
            benh = ucats.load_baseline_pickle(baseline_name)
        else:
            print('calculating baseline fluorescence')
            #benh = ucats.calculate_baseline_pca_asym(frames, smooth=300, niter=20, verbose=args.verbose)
            _, benh = ucats.block_svd_denoise_and_separate(frames, nhood=16, stride=16, min_comps=3,
                                                           baseline_smoothness=args.baseline_smoothness,
                                                           spatial_filter=3,
                                                           correct_spatial_components=False,
                                                           with_clusters=False)
            print('storing baseline fluorescence estimate')
            ucats.store_baseline_pickle(baseline_name,benh)

        benh = fseq.from_array(benh)
        benh.meta['channel'] = 'Fbaseline'

    else:
        print('Denoising frames and separating background via block-SVD in sqrt-ed data')
        #todo: take parameters from arguments to the script
        print('Going in ~%d time-slices'% (2*np.ceil(len(frames)/args.detection_temporal_window)-1))
        if args.detection_do_variance_stabilization:
            xt = 2*np.sqrt(frames)
        else:
            xt = frames
        _process_kwargs = dict(nhood=args.detection_loc_nhood,
                               stride=args.detection_loc_stride,
                               spatial_filter=args.detection_spatial_filter,
                               min_comps=args.detection_min_components,
                               with_clusters = args.detection_use_clusters,
                               svd_detection_plow = args.detection_low_percentile*2,
                               baseline_smoothness=args.baseline_smoothness,
                               spatial_min_cluster_size=5)
        fdelta, fb = multiscale_process_frames(xt, twindow=args.detection_temporal_window, **_process_kwargs )
        #fdelta, fb = ucats.block_svd_separate_tslices(xt,twindow=args.detection_temporal_window, **_process_kwargs)

        # III. Calculate ΔF/F
        print('Calculating relative fluorescence changes')
        th1 = ucats.percentile_th_frames(fdelta,2.0)
        print('Fdelta dynamic range:', fdelta.min(), fdelta.max())
        print('Fbase  dynamic range:', fb.min(), fb.max())
        print('mad std dynamic range:', ucats.mad_std(xt-fdelta-fb, axis=0).min(), ucats.mad_std(xt-fb-fdelta, axis=0).max())
        correction_bias = ucats.find_bias_frames(xt-fdelta-fb,3,ucats.mad_std(xt-fdelta-fb,axis=0))
        print('Correction bias dynamic range:', np.min(correction_bias), np.max(correction_bias))
        #if any(np.isnan(correction_bias)):
        #    correction_bias = np.zeros(correction_bias.shape)
        correction_bias[np.isnan(correction_bias)] = 0 ## can't find out so far why there are nans in some cases there. there shouldn't be
        fdelta = ucats.adaptive_median_filter(fdelta,ssmooth=3,keep_clusters=True) # todo: make this optional
        if args.detection_do_variance_stabilization:
            frames_dn,benh = ucats.convert_from_varstab(fdelta, fb + 0*correction_bias)
        else:
            frames_dn, benh = fdelta, fb
        print('storing baseline fluorescence estimate')
        ucats.store_baseline_pickle(baseline_name,benh)

        #nsdt = ucats.std_median(fdelta,axis=0)
        mask_pipeline = lambda m: ucats.threshold_object_size(ucats.expand_mask_by_median_filter(m,niter=3),9)
        mask1 = fdelta >= th1
        mask2 = frames_dn/benh >= 0.01 # 1% change from baseline
        mask_final = mask_pipeline(ucats.select_overlapping(mask1,mask2))
        #mask = ucats.opening_of_closing((fdelta > th)*(fdelta>nsdt)*(fdelta/fb > 0.025))
        #mask = ucats.opening_of_closing((fdelta > th)*(fdelta/fb > 0.025))
        frames_dn *= mask_final
        dfofx = frames_dn/benh
        dfofx[np.abs(benh)<1e-5] = 0
        print('Baseline dynamic range:', np.min(benh), np.max(benh))
        #benh =  0.25*fb**2
        #frames_dn = 0.25*fdelta**2 + 0.5*fdelta*fb
        del fb, fdelta,mask_final

        coll_ = ucats.EventCollection(dfofx,
                                      threshold=args.event_segmentation_threshold,
                                      min_area=args.event_min_area,
                                      min_duration=args.event_min_duration,
                                      peak_threshold=args.event_peak_threshold)
        dfofx = dfofx*(coll_.to_filtered_array()>0)

        if args.detection_do_second_pass:
            print('Doing second pass...')
            dfof = (frames/benh).astype(float32)  - 1
            diff = dfof - dfofx
            mindiff = min(0, np.min(diff))

            if args.detection_do_variance_stabilization:
                diff = 2*np.sqrt(diff - mindiff)

            fcorr, b2 = ucats.block_svd_denoise_and_separate(diff, stride=2,nhood=5,baseline_smoothness=150)

            if args.detection_do_variance_stabilization:
                offset = 2*(-mindiff)**0.5 if mindiff < 0 else 0
                fcorr,b2 = ucats.convert_from_varstab(fcorr, b2 + offset)

            dfofx2 = dfofx + fcorr

            coll_ = ucats.EventCollection(dfofx2,
                                          threshold=args.event_segmentation_threshold,
                                          min_area=args.event_min_area,
                                          min_duration=args.event_min_duration,
                                          peak_threshold=args.event_peak_threshold)
            dfofx2 = dfofx2*(             coll_.to_filtered_array()>0)

            mx = ucats.select_overlapping(dfofx2>=0.025,dfofx[:]>=0.01)
            dfofx2[~mx] = 0
            channel_name = 'newrec8a'
        else:
            dfofx2 = dfofx
            channel_name = 'newrec8'

        benh = fseq.from_array(benh)
        benh.meta['channel'] = 'Fbaseline'

        #if args.no_skip_dark_areas:
        #    print('calculating well-stained and poorly-stained areas')
        #    colored_mask = dark_area_mask(benh.data.mean(0))
        #else:
        #    print('no color mask asked for')
        #    colored_mask = np.ones(benh[0].shape, np.bool)

        #f,ax = plt.subplots(1,1,figsize=(8,8));
        #ax.imshow(benh.data.mean(0),cmap='gray')
        #ax.imshow(ui.plots.mask4overlay2(colored_mask,alpha=0.5))
        #plt.tight_layout()
        #f.savefig(nametag+'-colored_mask.png')
        #plt.close(f)
        fsx = fseq.from_array(dfofx2)
        fsx.meta['channel'] = channel_name

        #meta = fsx.meta
        #fsx = fseq.from_array(fsx.data*(coll_.to_filtered_array()>0))
        #fsx.meta = meta
        #   del coll_
        print('--->Done')
        if args.do_save_enh:
            fsx.to_hdf5(detected_name, compress_level=3)


    # VI.  Make movies
    if args.verbose: print('Making movies of detected activity')
    #fsout = fseq.FStackColl([fsc,  fsx])
    #frames_out = benh.data*(asarray(fsx.data,float32)+1)
    frames_out = benh.data
    #frames_out = benh.data*(dfof_cleaned + 1)
    fsout = fseq.FStackColl([benh,  fsx])
    p = ui.Picker(fsout); p.start()
    p0 = ui.Picker(fseq.FStackColl([fsc]))
    p0._ccmap=dict(b=None,i=None,r=None,g=0)
    bgclim = np.percentile(frames_out,(1,99))
    bgclim[1] *= 1.25
    p0.clims[0] = bgclim
    p.clims[0] = bgclim
    #p.clims[1] = (0.025,0.25)
    mip = np.max(fsx.data,0)
    p.clims[1] = (args.event_segmentation_threshold, np.mean(mip[mip>0]))
    print('Testing clims: ',p.clims[1])
    p._ccmap = dict(b=None,i=None,r=1,g=0)
    #ui.pickers_to_movie([p],name+'-detected.mp4',writer='ffmpeg')
    ui.pickers_to_movie([p0, p],nametag+'-b-detected.mp4', titles=('raw','processed'),
                        codec=args.codec,
                        bitrate=args.bitrate,
                        writer=args.writer)

    print('segmenting and animating events')
    events = ucats.EventCollection(asarray(fsx.data,dtype=np.float32), dfof_frames=fsc.data/benh.data-1)
    if len(events.filtered_coll):
        events.to_csv(nametag+'-events.csv')
    #animate_events(fsc.data, events,name+'-events-new4.mp4')
    animate_events(frames_out, events,args, nametag+'-c-events.mp4')
    print('All done')
    if h5f:
        h5f.close()
    return # from process_record()

#todo: move to ucats?
def downsample_stack(frames):
    return np.array([ucats.downsample_image(f) for f in frames], dtype=ucats._dtype_)

#todo: move to ucats?
def upsample_stack(frames, target=None):
    if target is None:
        target = np.array([ucats.upsample_image(f) for f in frames], dtype=ucats._dtype_)
    else:
        for k,f in enumerate(frames):
            copy_to_larger_cpad(ucats.upsample_image(f), target[k])
    return target



#todo: move to ucats?
def copy_to_larger_cpad(source, destination):
    nr,nc = source.shape
    nrd,ncd = destination.shape
    crop = (slice(min(nr,nrd)), slice(min(nc,ncd)))
    destination[crop] = source[crop]
    if np.any(np.array(destination.shape)>source.shape):
        destination[nr:,:nc] = source[nr-1,:][None,:]
        destination[:nr,nc:] = source[:,nc-1][:,None]
        destination[nr:,nc:] = source[-1,-1]

def multiscale_process_frames(frames, twindow, **_process_kwargs):
    frames_ds1 = downsample_stack(frames)
    frames_ds2 = downsample_stack(frames_ds1)

    fdelta2, fb2 = ucats.block_svd_separate_tslices(frames_ds2, twindow,  **_process_kwargs)
    fdelta2 = upsample_stack(fdelta2, np.zeros_like(frames_ds1))
    fb2 = upsample_stack(fb2, np.zeros_like(frames_ds1))

    frames_ds1 = frames_ds1 - fb2 - fdelta2
    fdelta1, fb1 = ucats.block_svd_separate_tslices(frames_ds1, twindow,  **_process_kwargs)

    fdelta1 = upsample_stack(fdelta1 + fdelta2, np.zeros_like(frames))
    fb1 = upsample_stack(fb1 + fb2, np.zeros_like(frames))

    fdelta, fb = ucats.block_svd_separate_tslices(frames-fdelta1-fb1, twindow,  **_process_kwargs)
    return fdelta + fdelta1, fb + fb1





def load_record(name, channel_name = 'fluo', with_plot=True,ca_channel=1):
    name_low = name.lower()
    read_as_array = True
    if endswith_any(name_low, ('.tif', '.tiff', '.lsm')):
        reader = tifffile
    elif endswith_any(name_low, ('.czi',)):
        reader = czifile
    elif endswith_any(name_low, ('.mes',)):

        reader = fseq.from_mes
        read_as_array = False
    else:
        # todo: raise exception
        print("Can't find appropriate reader for input file format")
        return

    if read_as_array:
        frames = squeeze(reader.imread(name))
        fs = fseq.from_array(frames)
    else:
        fs = reader(name)
    fs.meta['channel'] = channel_name
    fs.meta['file_path'] = name

    if with_plot:
        fig,ax = plt.subplots(1,2, gridspec_kw=dict(width_ratios=(1,3)),figsize=(12,3))
        if isinstance(fs, fseq.FStackColl) and len(fs.stacks)>1:
            fsca = fs.stacks[ca_channel]
            fsca.meta['channel']=channel_name
            fsca.meta['file_path'] = name
        else:
            fsca = fs

        mf_raw = fsca.mean_frame()

        mf = simple_rescale(mf_raw)
        bright_mask = mf > np.percentile(mf, 50)
        v = array([np.mean(f[bright_mask]) for f in fs])
        ax[0].imshow(mf,cmap='gray')
        ax[1].plot(v)
        ax[1].set_title(os.path.basename(name)+': mean fluorescence')
        ax[1].set_xlabel('frame #')
        plt.grid(True)
    return fs

imgreg_dispatcher_ = {'affine':imgreg.affine,
                      'homography':imgreg.homography,
                      'shifts':imgreg.shifts,
                      'Greenberg-Kerr':imgreg.greenberg_kerr,
                      'mslkp':imgreg.mslkp,
                      'msclg':imgreg.msclg}

def apply_warps_and_crop(fs, warps, verbose, njobs):
    mx_warps = ucats.max_shifts(warps, verbose)
    fsc = ofreg.warps.map_warps(warps, fs, njobs=njobs)
    fsc.meta['file_path']=fs.meta['file_path']
    fsc.meta['channel'] = fs.meta['channel']+'-sc'

    if isinstance(fs,fseq.FStackColl):
        for stack in fsc.stacks:
            stack.data = ucats.crop_by_max_shift(stack.data,warps,mx_warps)
    else:
        fsc.data = ucats.crop_by_max_shift(fsc.data,warps,mx_warps)
    return fsc


def stabilize_motion(fs, args, nametag='',suff=None):
    "Try to remove motion artifacts by image registratio"
    morphology_channel = args.morphology_channel
    if isinstance(fs, fseq.FStackColl) and len(fs.stacks) > 1:
        fsm = fs.stacks[morphology_channel]
    else:
        fsm = fs

    if suff is None: suff = ''
    models = [isinstance(m,str) and m or m[0] for m in args.stab_model]
    suff = suff+'-' + '-'.join(models) + '-ch-%d'%(args.morphology_channel)
    #warps_name = fs.meta['file_path']+suff+'.npy'
    #warps_name = nametag+suff+'-warps.npy'
    warps_name = '-'.join((nametag, suff, 'warps.npy'))
    print('warps name:', warps_name)
    fsm_filtered = None
    newframes = None

    if os.path.exists(warps_name) and (not args.with_motion_movies):
        final_warps = ofreg.warps.from_dct_encoded(warps_name)
        fsc = apply_warps_and_crop(fs, final_warps, args.verbose, args.ncpu)
        return fsc, final_warps



    # If reusing previously calculating warps and want to re-make the movie
    if os.path.exists(warps_name) and (not args.with_motion_movies):
        final_warps = ofreg.warps.from_dct_encoded(warps_name)
        fsc = apply_warps_and_crop(fs, final_warps, args.verbose, args.ncpu)
        return fsc, final_warps


    if args.verbose:
        print('Filtering data')

    # Median filter. TODO: make optional via arguments
    #fsm.frame_filters = [partial(ndimage.median_filter, size=3)]
    #fsm_filtered = fseq.from_array(fsm[:])
    #fsm_filtered = ndi.median_filter(fsm[:], size=(5,3,3)).astype(ucats._dtype_)
    fsm_filtered = ucats.clip_outliers(fsm[:],0.05, 99.95).astype(ucats._dtype_)
    # Removing global trend
    #fsm_filtered = fsm_filtered - fsm_filtered.mean(axis=(1,2))[:,None,None]


    #fsm.frame_filters = []
    #if args.verbose > 1: print('done spatial median filter')

    from imfun.core import fnutils
    if args.pca_denoise:
        pcf = components.pca.PCA_frames(2*fsm_filtered**0.5, npc=len(fsm)//50+5)
        vh_frames = pcf.vh.reshape(-1, *pcf.sh)

        #process_spatial_component = fnutils.flcompose(
        #    lambda f: ucats.adaptive_median_filter_2d(f,th=2,smooth=25,reverse=True),
        #    lambda f: ucats.adaptive_median_filter_2d(f,th=5,smooth=3),
        #    lambda f: ucats.l2spline(f,0.5))
        process_spatial_component = fnutils.flcompose(
            lambda f: ucats.adaptive_filter_2d(f,th=1.5,smooth=25,reverse=True, keep_clusters=False,
                                               smoother=lambda f_,smooth: ucats.smoothed_medianf(f_,5,smooth)),
            lambda f: ucats.l2spline(f,1.0),
            lambda f: f-ucats.l2spline(f, 30),
        )
        #smooth_and_detrend = lambda f_: l2spline(f_,1.0)-l2spline(f_,30)
        coords_s = array([ucats.smoothed_medianf(v,0.5,3) for v in pcf.coords.T]).T
        pcf.vh = array([[process_spatial_component(f) for f in vh_frames]]).reshape(pcf.vh.shape)
        pcf.tsvd.components_ = pcf.vh
        fsm_filtered = pcf.tsvd.inverse_transform(coords_s).reshape(len(fsm_filtered),*pcf.sh) + process_spatial_component(pcf.mean_frame)
        fsm_filtered = fsm_filtered.astype(ucats._dtype_)
        if args.verbose>1: print('done PCA-based denoising')
    else: pcf = None

    fsm_filtered = fsm_filtered.astype(ucats._dtype_)


    #fsm_filtered.frame_filters.append(lambda f: l2spline(f,1.5)-l2spline(f,30))
    #fsm_filtered = fseq.from_array(fsm_filtered[:])
    #if args.verbose>1: print('done flattening')
    if args.verbose: print('Done filtering')

    if os.path.exists(warps_name):
        print('Loading pre-calculated movement correction:', warps_name)
        final_warps = ofreg.warps.from_dct_encoded(warps_name)
    else:
        if args.verbose:
            print('No existing movement correction found, calculating...')
        operations = args.stab_model
        warp_history = []
        newframes = fsm_filtered
        for movement_model in operations:
            if not isinstance(movement_model,str):
                if len(movement_model)>1:
                    model, stab_type, model_params = movement_model
                else:
                    model, stab_type, model_params = movement_model[0], 'updated_template', {}
            else:
                model = movement_model
                model_params = {}
                stab_type = 'updated_template'

            if args.verbose > 1:
                print('correcting for {} using {} with params: {}'.format(model, stab_type, model_params))
            template = newframes[:10].mean(0)
            if stab_type == 'template':
                warps = stackreg.to_template(newframes, template, regfn=imgreg_dispatcher_[model],
                                             njobs=args.ncpu, **model_params)
            elif stab_type == 'updated_template':
                warps = stackreg.to_updated_template(newframes, template, njobs=args.ncpu,
                                                     regfn=imgreg_dispatcher_[model], **model_params)
            elif stab_type in ['multi', 'multi-templates', 'pca-templates']:
                templates, affs = fseq.frame_exemplars_pca_som(newframes,npc=len(fsm)//100+5)
                warps = stackreg.to_templates(newframes, templates, affs, regfn=imgreg_dispatcher_[model],
                                              njobs=args.ncpu,
                                              **model_params)
            warp_history.append(warps)
            newframes = ofreg.warps.map_warps(warps, newframes, njobs=args.ncpu).astype(ucats._dtype_)
            mx_warps = ucats.max_shifts(warps, args.verbose)

        final_warps = [reduce(op.add, warpchain) for warpchain in zip(*warp_history)]
        ofreg.warps.to_dct_encoded(warps_name, final_warps)
        del warp_history, final_warps
        final_warps = ofreg.warps.from_dct_encoded(warps_name)
        # end else

    #mx_warps = ucats.max_shifts(final_warps, args.verbose)
    #fsc = ofreg.warps.map_warps(final_warps, fs, njobs=args.ncpu)
    #fsc.meta['file_path']=fs.meta['file_path']
    #fsc.meta['channel'] = fs.meta['channel']+'-sc'

    fsc = apply_warps_and_crop(fs, final_warps, args.verbose, args.ncpu)

    if isinstance(fs, fseq.FStackColl) and len(fs.stacks)>1:
        stacks = [fs.stacks[morphology_channel], fs.stacks[args.ca_channel]]
        stacks_c = [fsc.stacks[morphology_channel], fsc.stacks[args.ca_channel]]

        fs_show = fseq.FStackColl(stacks)
        fsc_show = fseq.FStackColl(stacks_c)

    else:
        fs_show = fs
        fsc_show = fsc


    if args.with_motion_movies:
        p1 = ui.Picker(fs_show)
        p2 = ui.Picker(fsc_show)
        clims = ui.harmonize_clims([p1,p2])
        p1.clims = clims
        p2.clims = clims
        pickers_list = [p1,p2]

        if (isinstance(fs, fseq.FStackColl) and len(fs.stacks) > 1) or (fsm_filtered is not None):
            if fsm_filtered is None:
                p3 = ui.Picker(fs.stacks[morphology_channel])
                newframes  = fsc.stacks[morphology_channel]
            else:
                p3 = ui.Picker(fseq.from_array(fsm_filtered))
                newframes = ofreg.warps.map_warps(final_warps, fsm_filtered)
            #print('------------------------------- New frames:', newframes)
            p4 = ui.Picker(fseq.from_array(newframes))
            #clims = ui.harmonize_clims([p3,p4])
            clims = [np.percentile(p3.frame_coll.stacks[0].data, (5,99.5))]
            p3.clims = clims
            p4.clims = clims
            pickers_list.extend([p3,p4])

        ui.pickers_to_movie(pickers_list, nametag+'-a-stabilization-%s.mp4'%suff,
                            codec=args.codec, writer=args.writer,titles=('raw', 'stabilized'))

    return fsc, final_warps # from stabilize_motion


def simple_rescale(m):
    low,high = percentile(m, (1,99))
    return clip((m-low)/(high-low),0,1)

def prep_mean_frame(fs):
    mfs = [simple_rescale(stack.mean_frame()) for stack in fs.stacks]
    if len(mfs) < 3:
        z = zeros_like(mfs[0])
        mfs += [z]*(3-len(mfs))
    return dstack(mfs[:3])

def endswith_any(s, suff_list):
    return reduce(op.or_, (s.endswith(suff) for suff in suff_list))

def remove_small_regions(mask, min_size=200):
    labels, nlab = ndimage.label(mask)
    for i in range(1,nlab+1):
        cond = labels==i
        if np.sum(cond) < min_size:
            labels[cond]=0
    return labels > 0

from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening
def dark_area_mask(mf,phigh=99.5, th_scale=0.1):
    mask = mf > np.percentile(mf,phigh)*th_scale
    return remove_small_regions(binary_opening(binary_closing(mask)))

from matplotlib import animation

def animate_events(frames, ev_coll, args,
                   movie_name='test.mp4',
                   min_event_show_size=50):
    pl,ph = percentile(frames, (1,99))

    header_add = 0.4
    figsize=(4.5,4.5+header_add)
    f,ax = plt.subplots(1,1,figsize=figsize)

    L =len(frames)
    titlestr = 'f: %03d'


    out = np.zeros(frames.shape+(4,),dtype=float32)#+avg.reshape((1,)+avg.shape).astype(float32)
    for d in ev_coll.filtered_coll:
        k = d['idx']
        o = ev_coll.objs[k]
        #color = np.random.uniform(size=4)
        color = list(plt.cm.tab20b(np.random.rand()))
        color[-1] = 0.5
        color = tuple(color)
        cond = ev_coll.labels[o]==k+1
        if np.sum(cond) > min_event_show_size:
            out[o][cond] = np.asarray(color)

    #np.save(movie_name+'-colored.npy',out)

    #out = ma.masked_less_equal(out,0)


    hb = ax.imshow(frames[0],clim=(pl,ph),cmap='gray',interpolation='nearest')
    hf = ax.imshow(out[0])
    ts  = ax.set_title(titlestr%(0),size='small')
    #sb = Rectangle((10,fsg[0].shape[1]-10), scalebar/dx, 3, color='g',ec='none')
    #ax.add_patch(sb)
    plt.setp(ax, frame_on=False, xticks=[],yticks=[])
    plt.tight_layout()

    def _animate(fk):
        hb.set_data(frames[fk])
        hf.set_data(out[fk])
        ts.set_text(titlestr%(fk))
        return [hb,hf]

    anim = animation.FuncAnimation(f, _animate, frames=int(L), blit=True)
    Writer = animation.writers.avail[args.writer]
    w = Writer(fps=args.fps,codec=args.codec,bitrate=args.bitrate)
    anim.save(movie_name, writer=w)


def make_mask(dfof, nsd0):
    th = ucats.percentile_th_frames(dfof,2.5)
    mask_simple1 = (dfof1 > th)*(dfof1>0.05)
    mask_simple1 = mask_simple1 + ndi.median_filter(mask_simple1, 3)
    mask_simple1 = np.array([ucats.cleanup_mask(m, 2,5) for m in mask_simple1])

    mask_simple2 = (dfof>nsd0)
    mask_simple2 = mask_simple2 + ndi.median_filter(mask_simple2, 3)
    mask_simple2 = np.array([ucats.cleanup_mask(m, 3,5) for m in mask_simple2])
    return ucats.select_overlapping(mask_simple2, mask_simple1)

if __name__ == '__main__':
    main()
    print('returned from main')
