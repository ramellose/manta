from setuptools import setup

setup(name='manta',
      version='0.2.0',
      packages=['manta'],
      description='Microbial association network clustering',
      author='Lisa Röttjers',
      author_email='lisa.rottjers@kuleuven.be',
      url='https://github.com/ramellose/manta',
      license='Apache-2.0',
      include_package_data=True,
      package_data={'manta': ['test/demo.graphml']},
      summary='Clustering and centrality algorithms for weighted and undirected networks.',
      entry_points={
          'console_scripts': [
              'manta = manta.manta:main'
          ]
      },
      )
