from setuptools import setup

setup(name='manca',
      version='0.1.0',
      packages=['manca', 'manca.bootstrap_centrality',
                'manca.cluster_sparse', 'manca.cyjson_utils', 'manca.layout_graph'],
      description='Microbial association network clustering',
      author='Lisa Röttjers',
      author_email='lisa.rottjers@kuleuven.be',
      url='https://github.com/ramellose/manca',
      license='Apache-2.0',
      summary='Clustering and centrality algorithms for weighted and undirected networks.',
      entry_points={
          'console_scripts': [
              'manca = manca.manca:main'
          ]
      },
      )
