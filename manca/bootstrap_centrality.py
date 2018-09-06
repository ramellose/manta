#!/usr/bin/env python

"""
The centrality scores generated from cluster_sparse are tested for their robustness.
The idea behind this test is that random rewiring should, to some extent,
preserve the most central structures of the original graph.
We cannot know which edges are true positives and which ones are false positives,
but we do expect that global network properties are retained despite changes in identified associations.

First, null models are generated from the original graph.
These models are rewired copies: edge degree and connectedness are preserved.
Weight is assigned randomly by sampling with replacement from the original weight scores.
For each of these networks, the diffusion iterations specified in cluster_sparse are repeated
as many times as for the original network. The outcome is then a matrix of diffusion scores.

With these matrices, the bootstrap error can be estimated. This error specifies how variable
the centrality statistic is expected to be. It is used in 1-sided t-tests to determine
whether centrality scores are larger or smaller than the percentile thresholds.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import sys
from random import choice
import networkx as nx
from copy import deepcopy
import numpy as np
from scipy import stats


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


def bootstrap_test(matrix, bootstraps, posthresh, negthresh):
    """
    Returns the p-values of the bootstrap procedure.
    These p-values are generated from a 1-sided t-test;
    this test determines whether an edge centrality score
    is larger or smaller than the thresholds.
    The standard error calculation has been described
    previously by Snijders & Borgatti, 1999.
    Each score is considered an individual statistic in this case.
    :param matrix: Matrix generated with diffuse_graph
    :param bootstraps: Bootstrapped diffuse_graph matrices
    :param posthresh: Positive threshold for edges
    :param negthresh: Negative threshold for edges
    :return: Matrix of p-values
    """
    mean_straps = np.mean(np.array(bootstraps), axis=0)
    sums = list()
    for strap in bootstraps:
        boot = (strap - mean_straps) ** 2
        sums.append(boot)
    total_errors = np.sum(np.array(sums), axis=0)
    se = np.sqrt((1 / (len(bootstraps)-1))*total_errors)
    tpos = (matrix - posthresh) / se  # thresholds are different for positive / negative hubs
    tneg = (matrix - negthresh) / se
    t = deepcopy(tpos)
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            if matrix[i,j] > 0:
                t[i,j] = tpos[i,j]
            elif matrix[i, j] < 0:
                t[i,j] = tneg[i,j]
            else:
                t[i,j] = 1
    # the t statistic determines which nodes are likely to be above the percentile treshold
    pvals = deepcopy(t)
    for x in range(np.shape(matrix)[0]):
        for y in range(np.shape(matrix)[1]):
            tval = t[x, y]
            if not np.isnan(tval):
                pvals[x, y] = stats.t.sf(np.abs(tval), len(bootstraps) - 1)  # one-sided t-test
            else:
                pvals[x, y] = 0
    return pvals


def bootstrap_graph(graph, matrix, iterations, diff_range, bootstraps, posthresh, negthresh):
    """
    Calls the null_graph function and returns the p-values of the bootstrap procedure.
    Each score is considered an individual statistic in this case.
    :param graph: NetworkX graph of a microbial association network.
    :param matrix: Outcome of diffusion process from cluster_graph.
    :param iterations: The number of iterations carried out by the clustering algorithm.
    :param diff_range: Diffusion range of network perturbation.
    :param bootstraps: Number of bootstraps to carry out. If 0, no bootstrapping is done.
    :param posthresh: Positive threshold for edges
    :param negthresh: Negative threshold for edges
    :return: Matrix of p-values
    """
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
    pvals = bootstrap_test(matrix, boots, posthresh, negthresh)
    return pvals


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
