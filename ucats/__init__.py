"""
μCats -- a set of routines for detection and analysis of Ca-transients
"""

from . import denoising
from . import decomposition
from . import baselines
from . import events
from . import detection1d
from . import events
#from . import io_lif
from . import masks
from . import patches
from . import scramble
from . import utils
from . import pmt

from .pmt import  estimate_gain_and_offset
from .anscombe import Anscombe
from .utils import clip_outliers, mad_std


import numpy as np
_dtype_ = np.float32
