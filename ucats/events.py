import numpy as np
import pandas as pd

from scipy import ndimage as ndi

from .globals import _dtype_


def quantify_events(rec, labeled, dt=1):
    "Collect information about transients for a 1D reconstruction"
    acc = []
    idx = np.arange(len(rec))
    for i in range(1, np.max(labeled) + 1):
        mask = labeled == i
        cut = rec[mask]
        ev = dict(start=np.min(idx[mask]),
                  stop=np.max(idx[mask]),
                  peak=np.max(cut),
                  time_to_peak=np.argmax(cut),
                  vmean=np.mean(cut))
        acc.append(ev)
    return acc


def segment_events(dataset, threshold=0.01):
    labels, nlab = ndi.label(np.asarray(dataset, dtype=_dtype_) > threshold)
    objs = ndi.find_objects(labels)
    return labels, objs


class EventCollection:
    def __init__(self,
                 frames,
                 threshold=0.025,
                 dfof_frames=None,
                 gf_sigma=(0.5, 2, 2),
                 min_duration=3,
                 min_area=9,
                 peak_threshold=0.05):

        self.min_duration = min_duration
        self.labels, self.objs = segment_events(frames, threshold)



        self.coll = [
            dict(start=self.objs[k][0].start,
                 duration=self.event_duration(k),
                 peak_area_frame = self.event_peak_area_time(k),
                 peak_area = self.event_peak_area(k),
                 area=self.event_area(k),
                 volume=self.event_volume(k),
                 peak=self.data_value(k, frames),
                 avg=self.data_value(k, frames, np.mean),
                 idx=k) for k in range(len(self.objs))
        ]

        self.filtered_coll = [c for c in self.coll
                              if c['duration']>min_duration \
                              and c['peak']>peak_threshold\
                              and c['area']>min_area]
        if dfof_frames is not None:
            dfofx = ndi.gaussian_filter(dfof_frames, sigma=gf_sigma,
                                        order=(1, 0, 0))  # smoothed first derivatives in time
            nevents = len(self.coll)
            for (k, event), obj in zip(enumerate(self.coll), self.objs):
                vmask = self.event_volume_mask(k)
                areas = [np.sum(m) for m in vmask]
                area_diff = ndi.gaussian_filter1d(areas, 1.5, order=1)
                event['mean_area_expansion_rate'] = area_diff[area_diff > 0].mean()\
                 if any(area_diff > 0) else 0
                event['mean_area_shrink_rate'] = area_diff[area_diff < 0].mean()\
                 if any(area_diff < 0) else 0
                dx = dfofx[obj] * vmask
                flatmask = np.sum(vmask, 0) > 0
                event['mean_peak_rise'] = (dx.max(axis=0)[flatmask]).mean()
                event['mean_peak_decay'] = (dx.min(axis=0)[flatmask]).mean()
                event['max_peak_rise'] = (dx.max(axis=0)[flatmask]).max()
                event['max_peak_decay'] = (dx.min(axis=0)[flatmask]).min()

    def event_duration(self, k):
        o = self.objs[k]
        return o[0].stop - o[0].start

    def event_volume_mask(self, k):
        return self.labels[self.objs[k]] == k + 1

    def project_event_mask(self, k):
        return np.max(self.event_volume_mask(k), axis=0)

    def event_peak_area_time(self, k):
        vm = self.event_volume_mask(k)
        return np.argmax(np.sum(vm, axis=(1,2)))

    def event_area(self, k):
        return np.sum(self.project_event_mask(k).astype(int))

    def event_peak_area(self, k):
        vm = self.event_volume_mask(k)
        return  np.max(np.sum(vm, axis=(1,2)))

    def event_volume(self, k):
        return np.sum(self.event_volume_mask(k))

    def data_value(self, k, data, fn=np.max):
        o = self.objs[k]
        return fn(data[o][self.event_volume_mask(k)])

    def to_DataFrame(self, use_filtered_coll=True):
        coll = self.filtered_coll if use_filtered_coll else self.coll
        return pd.DataFrame(coll)

    def flatmask_fullframe(self, k):
        o = self.objs[k]
        mask = self.project_event_mask(k)
        frame_shape = self.labels.shape[1:]
        out = np.zeros(frame_shape)
        frame_crop = o[1:]
        out[frame_crop] = mask
        return out

    def to_csv(self, name):
        df = self.to_DataFrame()
        df.to_csv(name)

    def to_filtered_array(self):
        sh = self.labels.shape
        out = np.zeros(sh, dtype=int)
        for d in self.filtered_coll:
            k = d['idx']
            o = self.objs[k]
            cond = self.labels[o] == k + 1
            out[o][cond] = k
        return out
