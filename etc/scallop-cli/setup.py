from distutils.core import setup

setup(
  name='scallop',
  version='0.2.0',
  packages=[
    'scallop',
    'scallop.config',
    'scallop.stdlib',
    'scallop.stdlib.ff',
    'scallop.stdlib.fp',
    'scallop.stdlib.attr',
  ],
  requires=[
    'scallopy'
  ],
  entry_points = {
    'console_scripts': [
      'scallop=scallop.cli:main'
    ],
  }
)
