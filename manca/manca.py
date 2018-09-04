#!/usr/bin/env python

"""
manca: microbial association network clustering algorithm
manca takes a networkx (weighted) microbial network as input and
uses a diffusion-based process to iterate over the network.
After the sparsity values converge, the resulting cluster set
should have a minimum sparsity value.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
from random import choice
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
from copy import deepcopy
from math import sqrt
from scipy import stats

def set_manca():
    """This parser gets input settings for running the manca clustering algorithm.
    Apart from the parameters specified by cluster_graph,
    it requires an input format that can be read by networkx."""
    parser = argparse.ArgumentParser(
        description='Run the microbial association network clustering algorithm.')
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
                        help='Format of network file.',
                        choices=['gml', 'edgelist',
                                 'graphml', 'adj'],
                        default='graphml')
    parser.add_argument('-limit', '--convergence_limit',
                        dest='limit',
                        required=False,
                        help='The convergence limit '
                             'specifies how long the algorithm will repeat after '
                             'reaching equal sparsity values. ',
                        default=100)
    parser.add_argument('-df', '--diffusion_range',
                        dest='df',
                        required=False,
                        help='Diffusion is considered over a range of k neighbours. ',
                        default=3)
    parser.add_argument('-mc', '--max_clusters',
                        dest='mc',
                        required=False,
                        help='Number of clusters to consider in K-means clustering. ',
                        default=4)
    parser.add_argument('-iter', '--iterations',
                        dest='iter',
                        required=False,
                        help='Number of iterations to repeat if convergence is not reached. ',
                        default=1000)
    parser.add_argument('--central', dest='central', action='store_true',
                        help='If True, centrality values are calculated for the network. ', required=False)
    parser.set_defaults(central=False)
    parser.add_argument('-p', '--percentage',
                        dest='p',
                        required=False,
                        help='Percentage of central edges to return. ',
                        default=10)
    parser.add_argument('-boot', '--bootstrap',
                        dest='boot',
                        required=False,
                        help='Number of bootstrap iterations for centrality estimates. ',
                        default=100)
    return parser


def manca():
    args = set_manca().parse_args(sys.argv[1:])
    try:
        if args['f'] == 'graphml':
            network = nx.read_graphml(args['graph'])
        elif args['f'] == 'edgelist':
            network = nx.read_weighted_edgelist(args['graph'])
        elif args['f'] == 'gml':
            network = nx.read_gml(args['graph'])
        elif args['f'] == 'adj':
            network = nx.read_multiline_adjlist(args['graph'])
        else:
            sys.stdout.write('Format not accepted.' + '\n')
            sys.stdout.flush()
            exit()
    except Exception:
        sys.stdout.write('Could not import network file! ' + '\n')
        sys.stdout.flush()
        exit()
    clustered = main(network, limit=args['limit'], diff_range=args['df'],
                     max_clusters=args['mc'], iterations=args['iter'],
             central=args['central'], percentage=args['p'], bootstraps=args['boot'])
    if args['f'] == 'graphml':
        nx.write_graphml(clustered, args['fp'])
    elif args['f'] == 'edgelist':
        nx.write_weighted_edgelist(clustered, args['fp'])
    elif args['f'] == 'gml':
        nx.write_gml(clustered, args['fp'])
    elif args['f'] == 'adj':
        nx.write_multiline_adjlist(clustered, args['fp'])
    sys.stdout.write('Wrote clustered network to ' + args['fp'] + '.' + '\n')
    sys.stdout.flush()


def cluster_graph(graph, limit, diff_range, max_clusters, iterations):
    """
    Takes a networkx graph
    and carries out network clustering until
    sparsity results converge. Directionality is ignored;
    if weight is available, this is considered during the diffusion process.
    Setting diff_range to 1 means that the algorithm
    will basically cluster the adjacency matrix.

    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub species.

    Parameters
    ----------
    :param graph: Weighted, undirected networkx graph.
    :param limit: Number of iterations to run until alg considers sparsity value converged.
    :param diff_range: Diffusion range of network perturbation.
    :param max_clusters: Number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :return: NetworkX graph, number of iterations and diffusion matrix.
    """
    delay = 0  # after delay reaches the limit value, algorithm is considered converged
    adj = np.zeros((len(graph.nodes), len(graph.nodes)))  # this considers diffusion, I could also just use nx adj
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    rev_index = {v: k for k, v in adj_index.items()}
    prev_sparsity = 0
    iters = 0
    while delay < limit and iters < iterations:
        # next part is to define clusters of the adj matrix
        # cluster number is defined through gap statistic
        # max cluster number to test is by default 5
        # define topscore and bestcluster for no cluster
        adj = diffuse_graph(graph, adj, diff_range, adj_index)
        topscore = 2
        bestcluster = None
        randomclust = np.random.randint(2, size=len(adj))
        try:
            sh_score = [silhouette_score(adj, randomclust)]
        except ValueError:
            sh_score = [0]  # the randomclust can result in all 1s or 0s which crashes
        # scaler = MinMaxScaler()
        # select optimal cluster by silhouette score
        # cluster may be arbitrarily bad before convergence
        # may scale adj mat values from 0 to 1 but scaling is probably not necessary
        for i in range(1, max_clusters+1):
            # scaler.fit(adj)
            # proc_adj = scaler.transform(adj)
            clusters = KMeans(i).fit_predict(adj)
            try:
                silhouette_avg = silhouette_score(adj, clusters)
            except ValueError:
                # if only 1 cluster label is defined this can crash
                silhouette_avg = 0
            sh_score.append(silhouette_avg)
        topscore = int(np.argmax(sh_score))
        if topscore != 0:
            bestcluster = KMeans(topscore).fit_predict(adj)
            # with bestcluster defined,
            # sparsity of cut can be calculated
            sparsity = 0
            for cluster_id in set(bestcluster):
                node_ids = list(np.where(bestcluster == cluster_id)[0])
                node_ids = [rev_index.get(item, item) for item in node_ids]
                cluster = graph.subgraph(node_ids)
                # per cluster node:
                # edges that are not inside cluster are part of cut-set
                # total cut-set should be as small as possible
                for node in cluster.nodes:
                    nbs = graph.neighbors(node)
                    for nb in nbs:
                        if nb not in node_ids:
                            # only add 1 to sparsity if it is a positive edge
                            # otherwise subtract 1
                            cut = graph[node][nb]['weight']
                            if cut > 0:
                                sparsity += 1
                            else:
                                sparsity -= 1
            # print("Complete cut-set sparsity: " + sparsity)
            if prev_sparsity > sparsity:
                delay = 0
            if prev_sparsity == sparsity:
                delay += 1
            prev_sparsity = sparsity
            iters += 1
            sys.stdout.write('Current sparsity level: ' + str(prev_sparsity) + '\n')
            sys.stdout.flush()
            sys.stdout.write('Number of iterations: ' + str(iters) + '\n')
            sys.stdout.flush()
    if iters == iterations:
        sys.stdout.write('Warning: algorithm did not converge.' + '\n')
        sys.stdout.flush()
    clusdict = dict()
    for i in range(len(graph.nodes)):
        clusdict[list(graph.nodes)[i]] = bestcluster[i]
    nx.set_node_attributes(graph, values=clusdict, name='cluster')
    return graph, iters, adj


def central_graph(matrix, graph, iterations, diff_range, percentage, bootstraps):
    """
    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub associations.

    If the bootstrap number is set to a value above 0, the centrality values are bootstrapped.

    The fraction of positive edges and negative edges is based on the ratio between
    positive and negative weights in the network.

    Hence, a network with 90 positive weights and 10 negative weights will have 90% positive hubs returned.

    Parameters
    ----------
    :param matrix: Outcome of diffusion process from cluster_graph.
    :param graph: NetworkX graph of a microbial association network.
    :param iterations: The number of iterations carried out by the clustering algorithm.
    :param diff_range: Diffusion range of network perturbation.
    :param percentage: Determines percentile of hub species to return.
    :param bootstraps: Number of bootstraps to carry out. If 0, no bootstrapping is done.
    :return: Networkx graph with hub ID / p-value as node property.
    """
    weights = nx.get_edge_attributes(graph, 'weight')
    # calculates the ratio of positive / negative weights
    # note that ratios need to be adapted, because the matrix is symmetric
    posnodes = sum(weights[x] > 0 for x in weights)
    ratio = posnodes / len(weights)
    negthresh = np.percentile(matrix, percentage*(1-ratio)/2)
    posthresh = np.percentile(matrix, 100-percentage*ratio/2)
    neghubs = np.argwhere(matrix < negthresh)
    poshubs = np.argwhere(matrix > posthresh)
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    if bootstraps > 0:
        boots = list()
        for i in range(bootstraps):
            bootstrap = null_graph(graph)
            adj = np.zeros((len(graph.nodes), len(graph.nodes)))
            boot_index = dict()
            for k in range(len(graph.nodes)):
                boot_index[list(graph.nodes)[k]] = k
            for j in range(iterations):
                adj = diffuse_graph(bootstrap, adj, diff_range, boot_index)
            boots.append(adj)
            sys.stdout.write('Bootstrap iteration ' + str(i) + '\n')
            sys.stdout.flush()
        pvals = bootstrap_test(matrix, boots)

    # need to make sure graph is undirected
    graph = nx.to_undirected(graph)
    # initialize empty dictionary to store edge ID
    edge_vals = dict()
    edge_pvals = dict()
    # need to convert matrix index to node ID
    for edge in neghubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'negative hub'
        edge_pvals[(node1, node2)] = pvals[adj_index[node1], adj_index[node2]]
    for edge in poshubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'positive hub'
        edge_pvals[(node1, node2)] = pvals[adj_index[node1], adj_index[node2]]
    nx.set_edge_attributes(graph, values=edge_vals, name='hub')
    nx.set_edge_attributes(graph, values=edge_pvals, name='hub p-value')


def diffuse_graph(graph, difmat, diff_range, adj_index):
    """
    Diffusion process for matrix generation.
    The diffusion process iterates over the matrix;
    this function represents one iteration step.
    In that step, a random node N is selected from the graph.
    Then a perturbation is propagated across the network.
    The perturbation is multiplied by the weights of the associations between k neighbours and
    then added to the matrix at position (N, Kth neighbour).
    :param graph: NetworkX graph of a microbial assocation network.
    :param difmat: Diffusion matrix.
    :param diff_range: Diffusion range.
    :param adj_index: Indices for diffusion matrix.
    :return:
    """
    node = choice(list(graph.nodes))
    # iterate over node neighbours across range
    nbs = dict()
    nbs[node] = 1.0
    upper_diff = list()
    upper_diff.append(nbs)
    for i in range(diff_range):
        # this loop specifies diffusion of weight value over the random node
        new_upper = list()
        for nbs in upper_diff:
            for nb in nbs:
                new_nbs = graph.neighbors(nb)
                for new_nb in new_nbs:
                    next_diff = dict()
                    try:
                        weight = graph[nb][new_nb]['weight']
                    except KeyError:
                        sys.stdout.write('Edge did not have a weight attribute! Setting to 1.0' + '\n')
                        sys.stdout.flush()
                        weight = 1.0
                    next_diff[new_nb] = weight * nbs[nb]
                    difmat[adj_index[node], adj_index[new_nb]] += weight * nbs[nb]
                    difmat[adj_index[new_nb], adj_index[node]] += weight * nbs[nb]
                    # undirected so both sides have weight added
                    new_upper.append(next_diff)
        upper_diff = new_upper
    return difmat


def null_graph(graph):
    """
    Returns a rewired copy of the original graph.
    The rewiring procedure preserves degree
    as well as connectedness.
    The number of rewirings is the square of the node amount.
    This ensures the network is completely randomized.
    The weight of the edges also needs to be normalized.
    Therefore, after the network is randomized, weight is sampled
    from the original graph and added to the randomized graph.
    Because this does not take negative / positive hubs into account,
    the fraction of positive / negative weights per node
    is not preserved.
    :param graph: Original graph.
    :return: NetworkX graph
    """
    model = deepcopy(graph)
    swaps = len(model.nodes) ** 2
    nx.algorithms.connected_double_edge_swap(model, nswap=swaps)
    model = nx.to_undirected(model)
    edge_weights = list()
    for edge in graph.edges:
        edge_weights.append(graph[edge[0]][edge[1]]['weight'])
    random_weights = dict()
    for edge in model.edges:
        random_weights[edge] = choice(edge_weights)
    nx.set_edge_attributes(model, random_weights, 'weight')
    return model

def bootstrap_test(matrix, bootstraps):
    """
    Returns the p-values of the bootstrap procedure.
    These p-values are generated from a 1-sided t-test.
    The standard error calculation has been described
    previously by Snijders & Borgatti, 1999.
    Each score is considered an individual statistic in this case.
    :param matrix: Matrix generated with diffuse_graph
    :param bootstraps: Bootstrapped diffuse_graph matrices
    :param index: Index of node of interest
    :return: Matrix of p-values
    """
    mean_straps = np.mean(np.array(bootstraps), axis=0)
    sums = list()
    for strap in bootstraps:
        boot = (strap - mean_straps) ** 2
        sums.append(boot)
    total_errors = np.sum(np.array(sums), axis = 0)
    se = np.sqrt((1 / (len(bootstraps)-1))*total_errors)
    t = (matrix - mean_straps) / se  # gives an error for edges that are 0
    pvals = deepcopy(t)
    for x in range(np.shape(matrix)[0]):
        for y in range(np.shape(matrix)[1]):
            tval = t[x,y]
            if not np.isnan(tval):
                pvals[x,y] = stats.t.sf(np.abs(tval), len(bootstraps) - 1)  # one-sided t-test
            else:
                pvals[x,y] = 0
    return pvals


def main(graph, limit=100, diff_range=3, max_clusters=5, iterations=1000,
          central=True, percentage=10, bootstraps=100):
    """
    Main function that carries out graph clusterning and calculates centralities.
    :param graph: NetworkX graph to cluster. Needs to have edge weights.
    :param limit: Number of iterations to run until alg considers sparsity value converged.
    :param diff_range: Diffusion range of network perturbation.
    :param max_clusters: Number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :param central: If True, centrality values are calculated.
    :param percentage: Determines percentile thresholds.
    :param bootstraps: Number of bootstrap iterations.
    :return:
    """
    results = cluster_graph(graph, limit, diff_range, max_clusters, iterations)
    graph = results[0]
    numbers = results[1]
    matrix = results[2]
    if central:
        central_graph(matrix, graph, numbers, diff_range,
                      percentage, bootstraps)
    return graph

if __name__ == '__main__':
    manca()