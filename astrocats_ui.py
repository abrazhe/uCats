from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlButton
from pyforms.controls import ControlCheckBoxList
from pyforms.controls import ControlFile
from pyforms.controls import ControlLabel
from pyforms.controls import ControlNumber
from pyforms.controls import ControlText


class AstrocatsGUI(BaseWidget):
    def __init__(self, *args, **kwargs):
        super().__init__('Astrocat GUI')
        self._input_file = ControlFile('Imaging record (*.lsm or *.tiff)')
        # self._json_file = ControlFile('Parameter JSON file')
        self._flags = ControlCheckBoxList('Flags')
        self._morphology_channel = ControlNumber("Morphology channel", minimum=0, maximum=5, default=0)
        self._ca_channel = ControlNumber("Ca channel", minimum=0, maximum=5, default=1)
        self._suff = ControlText("Name suffix", default='astrocat004')
        self._fps = ControlNumber("Movie fps", default=25)
        self._codec = ControlText('Movie codec', default='libx264')

        self._detection_label = ControlLabel('Detection')
        self._detection_loc_nhood = ControlNumber('Nhood', default=5)
        self._detection_loc_stride = ControlNumber('Stride', default=2)
        self._detection_spatial_filter = ControlNumber('Spatial filter', default=3)

        self._run_button = ControlButton('Process')
        self._run_button.value = self.__run_button_fired

        self._formset = ['_input_file',
                         '=',
                         ['_detection_label', '_detection_loc_nhood', '_detection_loc_stride'],
                         ('_codec', '_fps'),
                         ('_run_button')]

    def __run_button_fired(self):
        pass


if __name__ == '__main__':
    from pyforms import start_app

    start_app(AstrocatsGUI, geometry=(200, 200, 400, 400))
