#!/usr/bin/env python

import os,sys
import h5py

import argparse

import json

from functools import partial,reduce
import operator as op

import numpy as np
from numpy import *

from scipy import ndimage,signal

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
        '-v': ('--verbose', dict(action='count', help='increment verbosity level')),
        '--morphology-channel': dict(default=0,type=int, help='color channel to use for motion correction (if several)'),
        '--ca-channel':dict(default=1, type=int, help='color channel with Ca-dependent fluorescence'),
        '--with-movies': dict(action='store_true'),
        '--suff': dict(default='', help="optional suffix to append to saved registration recipe"),
        '--fps': dict(default=25,type=float,help='fps of exported movie'),
        '--pca-denoise': dict(action='store_true'),
        '--codec': dict(default='libx264', help='movie codec'),
        '--writer': dict(default='ffmpeg', help='movie writer'),
        '--no-save-enh': dict(action='store_false'),
        '--save-denoised-dfof': dict(action='store_true'),
        '--no-events': dict(action='store_true'),
        '--skip-existing': dict(action='store_true'),
        '--do-oscillations': dict(action='store_true'),        
        '--bitrate':dict(default=32000,type=float, help='bitrate of exported movie'),
        }

    for arg,kw in list(argdict.items()):
        if isinstance(kw, dict):
            parser.add_argument(arg,  **argdict[arg])
        else:
            parser.add_argument(arg, kw[0], **kw[1])

    args = parser.parse_args()

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
    
    print("Processing", args.name)
    # I.   Load record(s)
    fs = load_record(args.name, ca_channel=args.ca_channel)
    # II.  Stabilize motion artifacts
    fsc = stabilize_motion(fs, args)

    if isinstance(fsc, fseq.FStackColl) and len(fsc.stacks) > 1:
        fsc = fsc.stacks[args.ca_channel]
    


    # III. Calculate baseline
    print('Calculating dynamic fluorescence baseline for motion-corrected data')
    smooth,mw = len(fsc)//10, len(fsc)//5
    #benh = ucats.calculate_baseline_pca(fsc.data, smooth=smooth,medianw=mw)
    benh = ucats.get_baseline_frames(fsc.data,smooth=smooth)
    h5f = None
    detected_name = args.name+'-detected-%s.h5'%args.suff
    colored_mask = dark_area_mask(benh.data.mean(0))

    # IV. Process data
    if os.path.exists(detected_name):
        print('loading existing results of event detection:', detected_name)
        #h5f = h5py.File(detected_name,'r')
        fsx = fseq.from_hdf5(detected_name)
    else:
        print("Calculating 'augmented' data")
        print(' - denoising ΔF/F frames')
        dfof_cleaned = ucats.patch_pca_denoise2(fsc.data/benh.data-1,npc=5,spatial_filter=5,
                                                mask_of_interest=colored_mask)
        if args.save_denoised_dfof:
            fs_tmp_ = fseq.from_array(dfof_cleaned)
            fs_tmp_.meta['channel'] = 'dfof_denoised'
            fs_tmp_.to_hdf5(args.name + '-dfof-denoised-%s.h5'%args.suff)
        print(' - Detecting and cleaning up events...')
        fsx = ucats.make_enh4(dfof_cleaned,kind='pca', nhood=2, mask_of_interest=colored_mask)
        coll_ = ucats.EventCollection(fsx.data,min_area=16)
        meta = fsx.meta
        fsx = fseq.from_array(fsx.data*(coll_.to_filtered_array()>0))
        fsx.meta = meta
        del coll_
        print('--->Done')
        if args.no_save_enh:
            fsx.to_hdf5(detected_name)

    
    # VI.  Make movies
    if args.verbose: print('Making movies of detected activity')        
    #fsout = fseq.FStackColl([fsc,  fsx])
    frames_out = benh.data*(asarray(fsx.data,float32)+1)
    #frames_out = benh.data*(dfof_cleaned + 1)
    fsout = fseq.FStackColl([fseq.from_array(frames_out),  fsx])        
    p = ui.Picker(fsout); p.start()
    p0 = ui.Picker(fseq.FStackColl([fsc]))
    p0._ccmap=dict(b=None,i=None,r=None,g=0)
    bgclim = np.percentile(frames_out,(1,99))
    p0.clims[0] = bgclim
    p.clims[0] = bgclim
    p.clims[1] = (0.025,0.25)
    p._ccmap = dict(b=1,i=None,r=1,g=0)
    #ui.pickers_to_movie([p],name+'-detected.mp4',writer='ffmpeg')
    ui.pickers_to_movie([p0, p],args.name+'-detected-%s.mp4'%args.suff, titles=('raw','processed'),
                        codec=args.codec,
                        bitrate=args.bitrate,
                        writer=args.writer)
    
    print('segmenting and animating events')
    events = ucats.EventCollection(asarray(fsx.data,dtype=np.float32))
    if len(events.filtered_coll):
        events.to_csv(args.name+'-events-%s.csv'%args.suff)
    #animate_events(fsc.data, events,name+'-events-new4.mp4')
    animate_events(frames_out, events,args, args.name+'-events-%s.mp4'%args.suff)
    if h5f:
        h5f.close()

    return # from main()


def load_record(name, channel_name = 'fluo', with_plot=True,ca_channel=1):
    name_low = name.lower()
    if endswith_any(name_low, ('.tif', '.tiff', '.lsm')):
        reader = tifffile
    elif endswith_any(name_low, ('.czi',)):
        reader = czifile
    else:
        # todo: raise exception
        print("Can't find appropriate reader for input file format")
        return

    frames = squeeze(reader.imread(name))
    fs = fseq.from_array(frames)
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


def stabilize_motion(fs, args,suff=''):
    "Try to remove motion artifacts by image registratio"
    morphology_channel = args.morphology_channel
    if isinstance(fs, fseq.FStackColl) and len(fs.stacks) > 1:
        fsm = fs.stacks[morphology_channel]
    else:
        fsm = fs

    if suff is None: suff = ''
    models = [isinstance(m,str) and m or m[0] for m in args.stab_model]
    suff = suff+'-' + '-'.join(models) + '-ch-%d'%(args.morphology_channel)
    warps_name = fs.meta['file_path']+suff+'.npy'
    if os.path.exists(warps_name):
        print('Loading pre-calculated movement correction:', warps_name)
        final_warps = ofreg.warps.from_dct_encoded(warps_name)
    else:
        if args.verbose:
            print('No existing movement correction found, calculating...')            
            print('Filtering data')

        # Median filter. TODO: make optional via arguments
        fsm.frame_filters = [partial(ndimage.median_filter, size=3)]
        fsm_filtered = fseq.from_array(fsm[:])
        fsm.frame_filters = []
        if args.verbose > 1: print('done spatial median filter')

        if args.pca_denoise:
            pcf = components.pca.PCA_frames(fsm_filtered, npc=len(fsm)//100+10)
            if args.verbose>1: print('done truncated PCA projection')
            fsm_filtered.frame_filters = [pcf.approx]
            fsm_filtered = fseq.from_array(fsm_filtered[:])
            if args.verbose>1: print('done PCA-based denoising')

        # Additional smoothing and removing trend
        fsm_filtered.frame_filters.append(lambda f: l2spline(f,1.5)-l2spline(f,30))
        fsm_filtered = fseq.from_array(fsm_filtered[:])
        if args.verbose>1: print('done flattening')
        if args.verbose: print('Done filtering')

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
                warps = stackreg.to_template(newframes, template, regfn=imgreg_dispatcher_[model], **model_params)
            elif stab_type == 'updated_template':
                warps = stackreg.to_updated_template(newframes, template, regfn=imgreg_dispatcher_[model], **model_params)
            elif stab_type in ['multi', 'multi-templates', 'pca-templates']:
                templates, affs = fseq.frame_exemplars_pca_som(newframes)
                warps = stackreg.to_templates(newframes, templates, affs, regfn=imgreg_dispatcher_[model],**model_params)
            warp_history.append(warps)
            newframes = ofreg.warps.map_warps(warps, newframes, njobs=args.ncpu)
            mx_warps = max_shifts(warps, args.verbose)            

        final_warps = [reduce(op.add, warpchain) for warpchain in zip(*warp_history)]
        ofreg.warps.to_dct_encoded(warps_name, final_warps)
        # end else
    mx_warps = max_shifts(final_warps, args.verbose)
    fsc = ofreg.warps.map_warps(final_warps, fs)
    fsc.meta['file_path']=fs.meta['file_path']
    fsc.meta['channel'] = fs.meta['channel']+'-sc'
    if isinstance(fs,fseq.FStackColl):
        for stack in fsc.stacks:
            stack.data = crop_by_max_shift(stack.data,final_warps,mx_warps)
    else:
        fsc.data = crop_by_max_shift(fsc.data,final_warps,mx_warps)
    if args.with_movies:
        p1 = ui.Picker(fs)
        p2 = ui.Picker(fsc)
        clims = ui.harmonize_clims([p1,p2])
        p1.clims = clims
        p2.clims = clims
        ui.pickers_to_movie([p1, p2],fs.meta['file_path']+'-stabilization-%s.mp4'%suff,
                            codec=args.codec, writer=args.writer,titles=('raw', 'stabilized'))
    
    return fsc # from stabilize_motion


def simple_rescale(m):
    low,high = percentile(m, (1,99))
    return clip((m-low)/(high-low),0,1)

def prep_mean_frame(fs):
    mfs = [simple_rescale(stack.mean_frame()) for stack in fs.stacks]
    if len(mfs) < 3:
        z = zeros_like(mfs[0])
        mfs += [z]*(3-len(mfs))
    return dstack(mfs[:3])

def max_shifts(shifts,verbose=0):
    #ms = np.max(np.abs([s.fn_((0,0)) if s.fn_ else s.field[...,0,0] for s in shifts]),axis=0)
    ms = np.max([np.array([np.percentile(f,99) for f in np.abs(w.field)]) for w in shifts],0)
    if verbose: print('Maximal shifts were (x,y): ', ms)
    return ceil(ms).astype(int)

def crop_by_max_shift(data, shifts, mx_shifts=None):
    if mx_shifts is None:
        mx_shifts = max_shifts(shifts)
    lims = 2*mx_shifts
    return data[:,lims[1]:-lims[1],lims[0]:-lims[0]]


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
def dark_area_mask(mf):
    mask = mf > np.percentile(mf,99.5)*0.1
    return binary_fill_holes(remove_small_regions(binary_opening(binary_closing(mask))))

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


if __name__ == '__main__':
    main()
