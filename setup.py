from setuptools import setup

setup(name='ucats',
      version = '0.0.1',
      requires = ['numpy','scipy','image_funcut'],
      py_modules=[u'μCats','io_lif', 'astrocats'],
      scripts=['astrocats.py'],
)
