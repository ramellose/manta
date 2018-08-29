from setuptools import setup

setup(name='manca',
      version='0.1.0',
      packages=['manca'],
      entry_points={
          'console_scripts': [
              'manca = manca.__main__:main'
          ]
      },
      )