import bioformats
import javabridge

import numpy as np
import xmltodict

from imfun import core, fseq

# Can only start this once:
javabridge.start_vm(class_path=bioformats.JARS)


def load_meta_records(name):
    meta = bioformats.get_omexml_metadata(name)
    dmeta = xmltodict.parse(meta)
    return meta, [LIF_record_meta(im) for im in dmeta['OME']['Image']]


class LIF_record_meta:
    def __init__(self, image_meta):
        self.meta = image_meta  # bioformats.get_omexml_metadata(name)
        d = self.meta
        self.id = d['@ID']
        self.index = int(self.id.split(':')[1])
        self.name = d['@Name']
        self.date = None
        if 'AcqusitionDate' in d:
            self.date = d['AcquisitionDate']
        self.dp = d['Pixels']
        # self.dXspan = self.get_physical_size('X')
        #
        # self.axes = core.units.alist_to_scale([])

    def get_size(self, axis):
        key = '@Size' + axis.upper()
        if key in self.dp:
            return int(self.dp[key])

    def get_axes(self):
        alist = [self.get_dt()] + [self.get_physical_size(ax) for ax in 'XYZ']
        alist = [a for a in alist if a is not None]
        self.axes = core.units.alist_to_scale(alist)
        return self.axes

    def get_physical_size(self, axis):
        key = '@PhysicalSize' + axis.upper()
        key_unit = '@PhysicalSize' + axis.upper() + 'Unit'
        if key in self.dp and key_unit in self.dp:
            return float(self.dp[key]), self.dp[key_unit]

    def get_dt(self):
        planes = self.dp['Plane']
        p = planes[2]
        if '@DeltaT' in p:
            return (float(p['@DeltaT']), p['@DeltaTUnit'])
        else:
            return (0, '_')

    def load_timelapse(self, name):
        "Loads a TXY record"
        numframes = self.get_size('T')
        axes = self.get_axes()
        reader = bioformats.ImageReader(name)
        kw = dict(series=self.index, rescale=False)
        images = np.array([reader.read(t=i, **kw) for i in range(1, numframes)])
        out = fseq.from_array(images)
        out.meta['axes'] = axes
        return out

    def __repr__(self):
        messg = 'LIF record %s, with name: %s\n' % (self.id, self.name)
        if self.date:
            messg += 'Recorded:  %s' % self.date
        messg += '# samples: {'
        for ax in 'TXYZ':
            messg += '%s: %d, ' % (ax, self.get_size(ax))
        messg += '}'
        return messg
