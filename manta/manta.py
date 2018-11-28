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
    parser.add_argument('-limit', '--convergence_limit',
                        dest='limit', type=float,
                        required=False,
                        help='The limit defines the minimum percentage decrease in error per iteration.'
                             ' If iterations do not decrease the error anymore, the matrix is considered converged. ',
                        default=2)
    parser.add_argument('-max', '--max_clusters',
                        dest='max', type=int,
                        required=False,
                        help='Maximum number of clusters to consider in K-means clustering. ',
                        default=4)
    parser.add_argument('-min', '--min_clusters',
                        dest='min', type=int,
                        required=False,
                        help='Minimum number of clusters to consider in K-means clustering. ',
                        default=2)
    parser.add_argument('-iter', '--iterations',
                        dest='iter', type=int,
                        required=False,
                        help='Number of iterations to repeat if convergence is not reached. ',
                        default=10)
    parser.add_argument('--central', dest='central', action='store_true',
                        help='With this flag, centrality values are calculated for the network. ', required=False)
    parser.set_defaults(central=False)
    parser.add_argument('--layout', dest='layout', action='store_true',
                        help='With this flag, layout coordinates are calculated for the network. ', required=False),
    parser.add_argument('-tax', '--taxonomy_table',
                        dest='tax',
                        help='Filepath to tab-delimited taxonomy table. '
                             'This table is used to calculate edge weights during layout calculation. '
                             'If the taxonomy table is already included as node properties in the input network,'
                             'these node properties are used instead. ',
                        default=None)
    parser.set_defaults(layout=False)
    parser.add_argument('-p', '--percentile',
                        dest='p', type=int,
                        required=False,
                        help='Percentile of central edges to return. For example, '
                             ' a percentile of 10 returns edges below the 10th and '
                             'edges above the 90th percentile. ',
                        default=10)
    parser.add_argument('-perm', '--permutation',
                        dest='perm', type=int,
                        required=False,
                        help='Number of permutation iterations for centrality estimates. ',
                        default=100)
    parser.add_argument('-cluster', '--clustering_algorithm',
                        dest='cluster', type=str,
                        choices=['KMeans', 'DBSCAN'],
                        required=False,
                        help='Choice for clustering algorithm. ',
                        default='KMeans')
    parser.add_argument('-e', '--error',
                        dest='error', type=int,
                        required=False,
                        help='Fraction of edges to rewire for reliability tests. ',
                        default=0.1)
    parser.add_argument('-scale', '--edgescale',
                        dest='edgescale', type=int,
                        required=False,
                        help='Edge scale used to separate out fuzzy clusters. '
                             'The larger the edge scale, the larger the fuzzy cluster.',
                        default=0.5)
    parser.add_argument('-fuzzy', '--fuzzy_nodes',
                        dest='fuzzy',
                        required=False,
                        help='If set to True, fuzzy nodes are separated from the clusters. ',
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
    network = nx.to_undirected(network)
    clustered = clus_central(network, limit=args['limit'],
                             max_clusters=args['max'], min_clusters=args['min'], iterations=args['iter'],
                             edgescale=args['edgescale'], central=args['central'], percentile=args['p'], permutations=args['perm'],
                             cluster=args['cluster'], error=args['error'], fuzzy=args['fuzzy'])
    layout = None
    if args['layout']:
        layout = generate_layout(clustered, args['tax'])
    if args['f'] == 'graphml':
        nx.write_graphml(clustered, args['fp'])
    elif args['f'] == 'edgelist':
        nx.write_weighted_edgelist(clustered, args['fp'])
    elif args['f'] == 'gml':
        nx.write_gml(clustered, args['fp'])
    elif args['f'] == 'adj':
        nx.write_multiline_adjlist(clustered, args['fp'])
    elif args['f'] == 'cyjs':
        write_cyjson(graph=clustered, filename=args['fp'], layout=layout)
    sys.stdout.write('Wrote clustered network to ' + args['fp'] + '.' + '\n')
    sys.stdout.flush()
    exit(0)


def clus_central(graph, limit=2, max_clusters=5, min_clusters=2, iterations=20, edgescale=0.5,
                 central=True, percentile=10, permutations=100, cluster='KMeans', error=0.1, fuzzy=True):
    """
    Main function that carries out graph clustering and calculates centralities.
    :param graph: NetworkX graph to cluster. Needs to have edge weights.
    :param limit: Percentage in error decrease until matrix is considered converged.
    :param max_clusters: Maximum number of clusters to evaluate in K-means clustering.
    :param min_clusters: Minimum number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :param edgescale: Scaling factor used to separate out fuzzy cluster.
    :param central: If True, centrality values are calculated.
    :param percentile: Determines percentile thresholds.
    :param permutations: Number of permutations.
    :param cluster: Algorithm for clustering of diffusion matrix.
    :param error: Fraction of edges to rewire for reliability tests.
    :param fuzzy: If true, fuzzy nodes are identified
    :return:
    """
    results = cluster_graph(graph, limit=limit, max_clusters=max_clusters,
                            min_clusters=min_clusters, iterations=iterations,
                            edgescale=edgescale, cluster=cluster, fuzzy=fuzzy)
    graph = results[0]
    if central:
        central_edge(graph, percentile=percentile,
                     permutations=permutations, error=error)
        central_node(graph)
    return graph


if __name__ == '__main__':
    main()
