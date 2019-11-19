from setuptools import setup, find_packages

setup(name='deepar',
      version='0.0.1',
      description='DeepAR tensorflow 2 implementation',
      author='Jeffrey Gleason',
      author_email='jeffrey.gleason@yonder.co',
      url='https://github.com/NewKnowledge/deepar/tree/master',
      install_requires=['tensorflow==2.0.0', 'numpy', 'pandas', 'sklearn'],
      packages=find_packages()
     )