#!/usr/bin/env python

"""
manta: microbial association network clustering toolbox.
The script takes a weighted and undirected network as input
and uses this to generate network clusters.
Moreover, it can generate a Cytoscape-compatible layout (with optional taxonomy input).
Detailed explanations are available in the headers of each file.

The arguments for *manta* can be separated into 4 groups:

* Arguments for importing and exporting data.
*  Arguments for network clustering.
* Arguments for network clustering on flip-flopping networks.
* Arguments for network centralities.

The arguments for importing and exporting data include:

* -i Filepath to input network.
* -o Filepath to output network
* -f Filetype for output network
* -tax Filepath to taxonomy table.
* --layout If flagged, a layout is generated
* -dir If a directed network is imported, setting this to True converts the network to undirected.
* -b If flagged, all edge weights are converted to 1 and -1.
manta uses the file extension to import networks. Taxonomy tables should be given as tab-delimited files.
These tables can be used to generate a layout for cyjson files.
Other file formats do not export layout coordinates.

The arguments for network clustering include:

* -min Minimum cluster number
* -max Maximum cluster number
* -mc Minimum cluster size
* -limit Error limit until convergence is considered reached
* -iter Number of iterations to keep going if convergence is not reached

manta uses agglomerative clustering on a scoring matrix to assign cluster identities.
The scoring matrix is generated through a procedure involving network flow.
Nodes that cluster separately are removed and combined with identified clusters later on.
Hence, manta will not identify clusters of only 1 node.
It is highly likely that networks will not converge neatly.
In that case, manta will apply the network flow procedure on a subset of the network.

The arguments for network clustering on flip-flopping networks include:

* -perm Number of permutations on network subsets
* -ratio Ratio of edges that need to be positive or negative to consider the edges stable through permutations.
* -scale Threshold for shortest path products to oscillators.

The network flow procedure relies on the following assumption:
positions in the scoring matrix that are mostly positive throughout permutations, should have only positive values added.
The same is assumed for negative positions.
The ratio defines which positions are considered mostly positive or mostly negative.

For demo purposes, we included a network generated from data
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
import sys
import os
import argparse
import manta
from manta.cluster import cluster_graph
from manta.reliability import perm_clusters
from manta.cyjson import write_cyjson, read_cyjson
from manta.layout import generate_layout
import numpy as np
import logging.handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# handler to sys.stdout
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# handler to file
# only handler with 'w' mode, rest is 'a'
# once this handler is started, the file writing is cleared
# other handlers append to the file
logpath = "\\".join(os.getcwd().split("\\")[:-1]) + '\\manta.log'
# filelog path is one folder above manta
# pyinstaller creates a temporary folder, so log would be deleted
fh = logging.handlers.RotatingFileHandler(maxBytes=500,
                                          filename=logpath, mode='a')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def set_manta():
    """This parser gets input settings for running the *manta* clustering algorithm.
    Apart from the parameters specified by cluster_graph,
    it requires an input format that can be read by networkx."""
    parser = argparse.ArgumentParser(
        description='Run the microbial association network clustering algorithm.'
                    'If --central is added, centrality is calculated. '
                    'Exporting as .cyjs allows for import into Cytoscape with '
                    'a cluster- and phylogeny-informed layout.')
    parser.add_argument('-i', '--input_graph',
                        dest='graph',
                        help='Input network file. The format is detected based on the extension; \n'
                             'at the moment, .graphml, .txt (weighted edgelist), .gml and .cyjs are accepted. \n'
                             'If you set -i to "demo", a demo dataset will be loaded.',
                        default=None,
                        required=True)
    parser.add_argument('-o', '--output_graph',
                        dest='fp',
                        help='Output network file. Specify full file path without extension.',
                        default=None, required=True)
    parser.add_argument('-f', '--file_type',
                        dest='f',
                        help='Format of output network file. Default: cyjs.',
                        choices=['gml', 'edgelist',
                                 'graphml', 'adj', 'cyjs'],
                        default='cyjs')
    parser.add_argument('-tax', '--taxonomy_table',
                        dest='tax',
                        help='Filepath to tab-delimited taxonomy table. '
                             'This table is used to calculate edge weights during layout calculation. '
                             'If the taxonomy table is already included as node properties in the input network,'
                             'these node properties are used instead. ',
                        default=None)
    parser.add_argument('--layout', dest='layout', action='store_true',
                        help='With this flag, layout coordinates are calculated for the network. '
                             'Only compatible with .cyjs output. ', required=False),
    parser.set_defaults(layout=False)
    parser.add_argument('-min', '--min_clusters',
                        dest='min', type=int,
                        required=False,
                        help='Minimum number of clusters. Default: 2.',
                        default=2)
    parser.add_argument('-ms', '--min_size',
                        dest='ms', type=float,
                        required=False,
                        help='Minimum cluster size as fraction of network size. Default: 0.1.',
                        default=0.1)
    parser.add_argument('-max', '--max_clusters',
                        dest='max', type=int,
                        required=False,
                        help='Maximum number of clusters. Default: 4.',
                        default=4)
    parser.add_argument('-limit', '--convergence_limit',
                        dest='limit', type=float,
                        required=False,
                        help='The limit defines the minimum percentage decrease in error per iteration.'
                             ' If iterations do not decrease the error anymore, the matrix is considered converged. '
                             'Default: 2.',
                        default=2)
    parser.add_argument('-iter', '--iterations',
                        dest='iter', type=int,
                        required=False,
                        help='Number of iterations to repeat if convergence is not reached. Default: 20.',
                        default=20)
    parser.add_argument('-perm', '--permutation',
                        dest='perm', type=int,
                        required=False,
                        help='Number of permutation iterations for '
                             'network subsetting during partial iterations. Default: 100.',
                        default=100)
    parser.add_argument('-ratio', '--stability_ratio',
                        dest='ratio', type=float,
                        required=False,
                        help='Fraction of scores that need to be positive or negative'
                             'for edge scores to be considered stable. Default: 0.8.',
                        default=0.8)
    parser.add_argument('-scale', '--edgescale',
                        dest='edgescale', type=float,
                        required=False,
                        help='Edge scale used to separate out weak cluster assignments. '
                             'The larger the edge scale, the larger the weak cluster. Default: 0.8.',
                        default=0.8)
    parser.add_argument('-cr, --cluster_reliability', dest='cr',
                        action='store_true',
                        help='With this flag, reliability of cluster assignment is computed. ', required=False)
    parser.set_defaults(cr=False)
    parser.add_argument('-rel', '--reliability_permutations',
                        dest='rel', type=int,
                        required=False,
                        help='Number of permutation iterations for reliability estimates. Default: 100.',
                        default=100)
    parser.add_argument('-e', '--error',
                        dest='error', type=int,
                        required=False,
                        help='Fraction of edges to rewire for reliability tests. ',
                        default=0.1)
    parser.add_argument('-dir', '--direction',
                        dest='direction',
                        action='store_true',
                        required=False,
                        help='If flagged, directed graphs are not converted to undirected after import. ',
                        default=False)
    parser.add_argument('-b', '--binary',
                        dest='bin',
                        action='store_true',
                        required=False,
                        default=False,
                        help='If flagged, edge weights are converted to 1 and -1.')
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='Provides additional details on progress. ',
                        default=False)
    return parser


def main():
    args = set_manta().parse_args(sys.argv[1:])
    args = vars(args)
    if args['graph'] != 'demo':
        filename = args['graph'].split(sep=".")
        extension = filename[len(filename)-1]
        try:
            if extension == 'graphml':
                network = nx.read_graphml(args['graph'])
            elif extension == 'txt':
                network = nx.read_weighted_edgelist(args['graph'])
            elif extension == 'gml':
                network = nx.read_gml(args['graph'])
            elif extension == 'cyjs':
                network = read_cyjson(args['graph'])
            else:
                logger.warning('Format not accepted.')
                exit()
        except Exception:
            logger.error('Could not import network file!', exc_info=True)
            exit()
        # first need to convert network to undirected
    else:
        path = os.path.dirname(manta.__file__)
        path = path + '\\demo.graphml'
        network = nx.read_graphml(path)
    if args['direction']:
        if extension == 'txt':
            logger.warning('Directed networks from edge lists not supported, use graphml or cyjs! ')
            exit()
    else:
        network = nx.to_undirected(network)
    if args['bin']:
        for edge in network.edges:
            network.edges[edge]['weight'] = np.sign(network.edges[edge]['weight'])
    results = cluster_graph(network, limit=args['limit'], max_clusters=args['max'],
                            min_clusters=args['min'], min_cluster_size=args['ms'],
                            iterations=args['iter'],
                            ratio=args['ratio'], edgescale=args['edgescale'],
                            permutations=args['perm'], verbose=args['verbose'])
    graph = results[0]
    if args['cr']:
        perm_clusters(graph=graph, limit=args['limit'], max_clusters=args['max'],
                      min_clusters=args['min'], iterations=args['iter'],
                      ratio=args['ratio'], min_cluster_size=args['ms'],
                      partialperms=args['perm'], relperms=args['rel'],
                      error=args['error'], verbose=args['verbose'])
    layout = None
    if args['layout']:
        layout = generate_layout(graph, args['tax'])
    if args['f'] == 'graphml':
        nx.write_graphml(graph, args['fp'] + '.graphml')
    elif args['f'] == 'edgelist':
        nx.write_weighted_edgelist(graph, args['fp'] + '.txt')
    elif args['f'] == 'gml':
        nx.write_gml(graph, args['fp'] + '.gml')
    elif args['f'] == 'adj':
        nx.write_multiline_adjlist(graph, args['fp'] + '.txt')
    elif args['f'] == 'cyjs':
        write_cyjson(graph=graph, filename=args['fp'] + '.cyjs', layout=layout)
    logger.info('Wrote clustered network to ' + args['fp'] + '.')
    exit(0)


if __name__ == '__main__':
    main()
