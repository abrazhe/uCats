"""
μCats -- a set of routines for detection and analysis of Ca-transients
"""

from . import denoising
from . import decomposition
from . import baselines
from . import events
from . import detection1d
from . import events
from . import io_lif
from . import masks
from . import patches
from . import scramble
from . import utils
from . import exponential_family


impot numpy as np
_dtype_ = np.float32
