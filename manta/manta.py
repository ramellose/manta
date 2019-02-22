#!/usr/bin/env python

"""
manta: microbial association network clustering toolbox.
The script takes a weighted and undirected network as input
and uses this to generate network clusters and calculate network centrality.
Moreover, it can generate a Cytoscape-compatible layout (with optional taxonomy input).
Detailed explanations are available in the headers of each file.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
import sys
import argparse
from manta.cluster import cluster_graph
from manta.centrality import central_edge, central_node
from manta.cyjson import write_cyjson, read_cyjson
from manta.layout import generate_layout


def set_manta():
    """This parser gets input settings for running the manta clustering algorithm.
    Apart from the parameters specified by cluster_graph,
    it requires an input format that can be read by networkx."""
    parser = argparse.ArgumentParser(
        description='Run the microbial association network clustering algorithm.'
                    'If --central is added, centrality is calculated. '
                    'Exporting as .cyjs allows for import into Cytoscape with '
                    'a cluster- and phylogeny-informed layout.')
    parser.add_argument('-i', '--input_graph',
                        dest='graph',
                        help='Input network file.',
                        default=None, required=True)
    parser.add_argument('-o', '--output_graph',
                        dest='fp',
                        help='Output network file.',
                        default=None, required=True)
    parser.add_argument('-f', '--file_type',
                        dest='f',
                        help='Format of output network file. Default is set to cyjs.',
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
    parser.add_argument('-min', '--min_clusters',
                        dest='min', type=int,
                        required=False,
                        help='Minimum number of clusters. ',
                        default=2)
    parser.add_argument('-max', '--max_clusters',
                        dest='max', type=int,
                        required=False,
                        help='Maximum number of clusters. ',
                        default=4)
    parser.add_argument('--layout', dest='layout', action='store_true',
                        help='With this flag, layout coordinates are calculated for the network. '
                             'Only compatible with .cyjs output. ', required=False),
    parser.set_defaults(layout=False)
    parser.add_argument('-limit', '--convergence_limit',
                        dest='limit', type=float,
                        required=False,
                        help='The limit defines the minimum percentage decrease in error per iteration.'
                             ' If iterations do not decrease the error anymore, the matrix is considered converged. ',
                        default=2)
    parser.add_argument('-iter', '--iterations',
                        dest='iter', type=int,
                        required=False,
                        help='Number of iterations to repeat if convergence is not reached. ',
                        default=10)
    parser.add_argument('-c, --central', dest='central', action='store_true',
                        help='With this flag, centrality values are calculated for the network. ', required=False)
    parser.set_defaults(central=False)
    parser.add_argument('-rel', '--reliability_permutations',
                        dest='rel', type=int,
                        required=False,
                        help='Number of permutation iterations for centrality estimates. ',
                        default=100)
    parser.add_argument('-p', '--percentile',
                        dest='p', type=int,
                        required=False,
                        help='Percentile of central edges to return. For example, '
                             ' a percentile of 10 returns edges below the 10th and '
                             'edges above the 90th percentile. ',
                        default=10)
    parser.add_argument('-e', '--error',
                        dest='error', type=int,
                        required=False,
                        help='Fraction of edges to rewire for reliability tests. ',
                        default=0.1)
    parser.add_argument('-perm', '--permutation',
                        dest='perm', type=int,
                        required=False,
                        help='Number of permutation iterations for '
                             'network subsetting during partial iterations. ',
                        default=100)
    parser.add_argument('-ratio', '--stability_ratio',
                        dest='ratio', type=float,
                        required=False,
                        help='Fraction of scores that need to be positive or negative'
                             'for edge scores to be considered stable. ',
                        default=0.7)
    parser.add_argument('-scale', '--edgescale',
                        dest='edgescale', type=int,
                        required=False,
                        help='Edge scale used to separate out fuzzy clusters. '
                             'The larger the edge scale, the larger the fuzzy cluster.',
                        default=0.5)
    parser.add_argument('-dir', '--direction',
                        dest='direction',
                        required=False, type=bool,
                        help='If set to False, directed graphs are converted to undirected after import. ',
                        default=False)
    parser.add_argument('-v', '--verbose',
                        dest='verbose',
                        required=False, type=bool,
                        help='Provides additional details on progress. ',
                        default=True)
    return parser


def main():
    args = set_manta().parse_args(sys.argv[1:])
    args = vars(args)
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
            sys.stdout.write('Format not accepted.' + '\n')
            sys.stdout.flush()
            exit()
    except Exception:
        sys.stdout.write('Could not import network file! ' + '\n')
        sys.stdout.flush()
        exit()
    # first need to convert network to undirected
    if args['direction']:
        if extension == 'txt':
            sys.stdout.write('Directed networks from edge lists not supported, use graphml or cyjs!. ' + '\n')
            sys.stdout.flush()
            exit()
    else:
        network = nx.to_undirected(network)
    results = cluster_graph(network, limit=args['limit'], max_clusters=args['max'],
                            min_clusters=args['min'], iterations=args['iter'],
                            ratio=args['ratio'], edgescale=args['edgescale'],
                            permutations=args['perm'], verbose=args['verbose'])
    graph = results[0]
    if args['central']:
        central_edge(graph, percentile=args['fp'], permutations=args['rel'],
                     error=args['error'], verbose=args['verbose'])
        central_node(graph)
    layout = None
    if args['layout']:
        layout = generate_layout(graph, args['tax'])
    if args['f'] == 'graphml':
        nx.write_graphml(graph, args['fp'])
    elif args['f'] == 'edgelist':
        nx.write_weighted_edgelist(graph, args['fp'])
    elif args['f'] == 'gml':
        nx.write_gml(graph, args['fp'])
    elif args['f'] == 'adj':
        nx.write_multiline_adjlist(graph, args['fp'])
    elif args['f'] == 'cyjs':
        write_cyjson(graph=graph, filename=args['fp'], layout=layout)
    sys.stdout.write('Wrote clustered network to ' + args['fp'] + '.' + '\n')
    sys.stdout.flush()
    exit(0)


if __name__ == '__main__':
    main()
