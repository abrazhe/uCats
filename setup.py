from setuptools import setup

setup(name='ucats',
      version = '0.0.2',
      requires = ['numpy','scipy','image_funcut','tqdm'],
      package_dir = {'ucats':'ucats'},
      packages =['ucats',
                  'ucats.denoising'],
      scripts=['astrocats.py'],
)
