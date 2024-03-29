#!/usr/bin/env python

"""
manta: microbial association network clustering toolbox.
The script takes a weighted and undirected network as input
and uses this to generate network clusters.
Moreover, it can generate a Cytoscape-compatible layout (with optional taxonomy input).
Detailed explanations are available in the headers of each file.

manta uses the file extension to import networks. Taxonomy tables should be given as tab-delimited files.
These tables can be used to generate a layout for cyjson files.
Other file formats do not export layout coordinates.

manta generates a scoring matrix and uses agglomerative clustering to assign cluster identities.
The scoring matrix is generated through a procedure involving network flow.
Nodes that cluster separately are removed and combined with identified clusters later on.
It is highly likely that networks will not converge neatly, as most real-world networks are unbalanced.
In that case, manta will apply the network flow procedure on a subset of the network.

The network flow procedure relies on the following assumption:
positions in the scoring matrix that are mostly positive throughout permutations, should have only positive values added.
The same is assumed for negative positions.
The ratio defines which positions are considered mostly positive or mostly negative.

Default numeric parameters:
-min    Minimum number of clusters. Default: 2.
-ms     Minimum cluster size as fraction of network size. Default: 0.2.
-max    Maximum number of clusters. Default: 4.
-limit  The limit defines the minimum percentage decrease in error per iteration.
        If iterations do not decrease the error anymore, the matrix is considered converged. Default: 2.
-perm   Number of permutation iterations for network subsetting during partial iterations. Default: number of nodes.
-subset     Fraction of edges that are used for subsetting if the input graph is not balanced. Default: 0.8.
-ratio  Fraction of scores that need to be positive or negative for edge scores to be considered stable. Default: 0.8.
-scale  Edge scale used to separate out weak cluster assignments.
        The larger the edge scale, the larger the weak cluster. Default: 0.8.
-rel    Number of permutation iterations for reliability estimates.
        By default, this number is estimated from the number of dyadic pairs.
-e      Fraction of edges to rewire for reliability tests. Default: 0.1.

For demo purposes, we included a network generated from oral samples of bats.
This data was downloaded from QIITA: https://qiita.ucsd.edu/study/description/11815
Lutz, H. L. et al. (2018). Associations between Afrotropical bats, parasites, and microbial symbionts. bioRxiv, 340109.
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
import pandas as pd
import logging.handlers
from pbr.version import VersionInfo

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
                    'Exporting as .cyjs allows for import into Cytoscape with '
                    'a cluster- and phylogeny-informed layout.')
    parser.add_argument('-i', '--input_graph',
                        dest='graph',
                        help='Input network file. The format is detected based on the extension; \n'
                             'at the moment, .graphml, .txt (weighted edgelist), .gml and .cyjs are accepted. \n'
                             'If you set -i to "demo", a demo dataset will be loaded.',
                        default=None,
                        required=False)
    parser.add_argument('-o', '--output_graph',
                        dest='fp',
                        help='Output network file. Specify full file path without extension.',
                        default=None, required=False)
    parser.add_argument('-f', '--file_type',
                        dest='f',
                        help='Format of output network file. Default: cyjs.\n'
                             'The csv format exports cluster assignments as a csv table.',
                        choices=['gml', 'graphml', 'cyjs', 'csv'],
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
                        help='Minimum cluster size as fraction of network size divided by cluster number. Default: 0.2.',
                        default=0.2)
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
                             'network subsetting during partial iterations. Default: number of nodes.',
                        default=None)
    parser.add_argument('-subset', '--subset_fraction',
                        dest='subset', type=float,
                        required=False,
                        help='Fraction of edges that are used for subsetting'
                             ' if the input graph is not balanced. Default: 0.8.',
                        default=0.8)
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
    parser.add_argument('-cr', '--cluster_reliability', dest='cr',
                        action='store_true',
                        help='If flagged, reliability of cluster assignment is computed. ', required=False)
    parser.set_defaults(cr=False)
    parser.add_argument('-rel', '--reliability_permutations',
                        dest='rel', type=int,
                        required=False,
                        help='Number of permutation iterations for reliability estimates. \n '
                             'By default, this is 20. \n',
                        default=20)
    parser.add_argument('-e', '--error',
                        dest='error', type=int,
                        required=False,
                        help='Fraction of edges to rewire for reliability tests. Default: 0.1.',
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
                        help='If flagged, edge weights are converted to 1 and -1. ')
    parser.add_argument('-seed', '--seed',
                        dest='seed',
                        required=False,
                        help='Specify seed. The default value (11111) means no seed is used.',
                        type=int,
                        default=11111)
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False,
                        action='store_true',
                        help='If flagged, rovides additional details on progress. ',
                        default=False)
    parser.add_argument('-version', '--version',
                        dest='version',
                        required=False,
                        help='Version number.',
                        action='store_true',
                        default=False)
    return parser


def main():
    args = set_manta().parse_args(sys.argv[1:])
    args = vars(args)
    if args['version']:
        info = VersionInfo('manta')
        logger.info('Version ' + info.version_string())
        exit(0)
    if args['graph'] != 'demo':
        filename = args['graph'].split(sep=".")
        extension = filename[len(filename)-1]
        # see if the file can be detected
        # if not, try appending current working directory and then read.
        if not os.path.isfile(args['graph']):
            if os.path.isfile(os.getcwd() + '/' + args['graph']):
                args['graph'] = os.getcwd() + '/'
            else:
                logger.error('Could not find the specified file. Is your file path correct?')
                exit()
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
                logger.warning('Format not accepted. '
                               'Please specify the filename including extension (e.g. test.graphml).', exc_info=True)
                exit()
        except Exception:
            logger.error('Could not import network file!', exc_info=True)
            exit()
        # first need to convert network to undirected
    elif args['graph'] == 'demo':
        path = os.path.dirname(manta.__file__)
        path = path + '//demo.graphml'
        network = nx.read_graphml(path)
    if args['direction']:
        if extension == 'txt':
            logger.warning('Directed networks from edge lists not supported, use graphml or cyjs! ')
            exit()
    else:
        network = nx.to_undirected(network)
    if args['bin']:
        orig_edges = dict()
        # store original edges for export
        for edge in network.edges:
            orig_edges[edge] = network.edges[edge]['weight']
            network.edges[edge]['weight'] = np.sign(network.edges[edge]['weight'])
    if sum(value == 0 for value in
           np.any(nx.get_edge_attributes(network, 'weight').values())) > 0:
        logger.error("Some edges in the network have a weight of exactly 0. \n"
                     "Such edges cannot be clustered. Try converting weights to 1 and -1. ")
    weight_properties = nx.get_edge_attributes(network, 'weight')
    if len(weight_properties) == 0:
        logger.error("The imported network has no 'weight' edge property. \n"
                     "Please make sure you are formatting the network correctly. ")
    results = cluster_graph(network, limit=args['limit'], max_clusters=args['max'],
                            min_clusters=args['min'], min_cluster_size=args['ms'],
                            iterations=args['iter'], subset=args['subset'],
                            ratio=args['ratio'], edgescale=args['edgescale'],
                            permutations=args['perm'], seed=args['seed'],
                            verbose=args['verbose'])
    graph = results[0]
    if args['cr']:
        perm_clusters(graph=graph, limit=args['limit'], max_clusters=args['max'],
                      min_clusters=args['min'], min_cluster_size=args['ms'],
                      iterations=args['iter'], ratio=args['ratio'],
                      partialperms=args['perm'], relperms=args['rel'], subset=args['subset'],
                      error=args['error'], verbose=args['verbose'])
    layout = None
    if args['bin']:
        for edge in network.edges:
            network.edges[edge]['weight'] = orig_edges[edge]
    if args['layout']:
        layout = generate_layout(graph, args['tax'])
    if args['fp']:
        if args['f'] == 'graphml':
            nx.write_graphml(graph, args['fp'] + '.graphml')
        elif args['f'] == 'csv':
            node_keys = graph.nodes[list(graph.nodes)[0]].keys()
            properties = {}
            for key in node_keys:
                properties[key] = nx.get_node_attributes(graph, key)
            data = pd.DataFrame(properties)
            data.to_csv(args['fp'] + '.csv')
        elif args['f'] == 'gml':
            nx.write_gml(graph, args['fp'] + '.gml')
        elif args['f'] == 'cyjs':
            write_cyjson(graph=graph, filename=args['fp'] + '.cyjs', layout=layout)
        logger.info('Wrote clustered network to ' + args['fp'] + '.' + args['f'])
    else:
        logger.error('Could not write network to disk, no file path given.')
    exit(0)


if __name__ == '__main__':
    main()
