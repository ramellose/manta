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
from math import log


def null_graph(graph):
    """
    Returns a rewired copy of the original graph.
    The rewiring procedure preserves degree of each node.
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
    model = graph.copy()
    swaps = len(model.nodes) ** 2
    nx.algorithms.double_edge_swap(model, nswap=swaps, max_tries=(swaps*100))
    model = nx.to_undirected(model)
    edge_weights = list()
    for edge in graph.edges:
        edge_weights.append(graph[edge[0]][edge[1]]['weight'])
    random_weights = dict()
    for edge in model.edges:
        random_weights[edge] = choice(edge_weights)
    nx.set_edge_attributes(model, random_weights, 'weight')
    return model


def perm_test(matrix, permutations, posthresh, negthresh):
    """
    Returns the p-values of the permutation procedure to test edge centrality scores.
    These p-values are generated from a 1-sided t-test;
    this test determines whether an edge centrality score
    is larger or smaller than the thresholds based on permutation testing.
    :param matrix: Matrix generated with diffuse_graph
    :param permutations: Permuted diffuse_graph matrices
    :param posthresh: Positive threshold for edges
    :param negthresh: Negative threshold for edges
    :return: Matrix of p-values
    """
    possums = list()
    negsums = list()
    for perm in permutations:
        tpos = perm - posthresh  # thresholds are different for positive / negative hubs
        tneg = perm - negthresh
        possums.append(tpos)
        negsums.append(tneg)
    # p value equals number of permutations that exceeds / is smaller than matrix values
    t = deepcopy(tpos)
    for i in range(np.shape(matrix)[0]):
        for j in range(np.shape(matrix)[1]):
            if matrix[i,j] > 0:
                t[i,j] = sum(perm[i,j] > matrix[i,j] for perm in possums)/len(possums)
            elif matrix[i, j] < 0:
                t[i,j] = sum(perm[i,j] < matrix[i,j] for perm in negsums)/len(possums)
            else:
                t[i,j] = 1
    return t

def perm_graph(graph, matrix, limit, iterations, permutations, posthresh, negthresh):
    """
    Calls the null_graph function and returns the p-values of the permutation procedure.
    Each score is considered an individual statistic in this case.
    :param graph: NetworkX graph of a microbial association network.
    :param matrix: Outcome of diffusion process from cluster_graph.
    :param limit: Error limit for matrix convergence.
    :param permutations: Number of permutations to carry out. If 0, no bootstrapping is done.
    :param posthresh: Positive threshold for edges
    :param negthresh: Negative threshold for edges
    :return: Matrix of p-values
    """
    perms = list()
    for i in range(permutations):
        permutation = null_graph(graph)
        adj = diffuse_graph(permutation, limit, iterations)[1]
        perms.append(adj)
        sys.stdout.write('Permutation iteration ' + str(i) + '\n')
        sys.stdout.flush()
    central_pvals = perm_test(matrix, perms, posthresh, negthresh)
    return central_pvals


def diffuse_graph(graph, limit=0.00001, iterations=50):
    """
    Diffusion process for generation of scoring matrix.
    The implementation of this process is similar
    to the MCL algorithm. However, the rescaling step
    in the MCL algorithm has been adjusted to accomodate
    for negative signs.
    In the first step, the matrix is scaled;
    afterwards, every value in the matrix has 1 divided by the value added.
    Subsequently, the matrix is scaled. The
    cumulative error, relative to the previous iteration,
    is calculated by taking the mean of the difference.
    :param graph: NetworkX graph of a microbial assocation network.
    :param limit: Error limit for matrix convergence.
    :param iterations: Maximum number of iterations to carry out.
    :return:
    """
    scoremat = nx.to_numpy_matrix(graph)
    flowmat = scoremat.copy()
    # if the 'weight' property of the graph is set correctly
    # weight in the adj graph should equal this
    error = 1
    prev_error = 0
    iters = 0
    while error > limit and iters < iterations:
        updated_mat = np.linalg.matrix_power(scoremat, 2)
        # expansion step
        # squaring the matrix without normalisation
        # is equal to a a Galton-Watson branching process
        updated_mat = updated_mat / abs(np.max(updated_mat))
        # the flow matrix describes flow and is output to compute centralities
        # in the MCL implementation, the rows are normalized to sum to 1
        # this creates a column stochastic matrix
        # here, we normalize by dividing with absolute largest value
        # normally, there is an inflation step; values are raised to a power
        # with this normalisation, the inflation step causes
        # the algorithm to converge to 0
        # we need above-0 values to converge to -1, and the rest to 1
        for value in np.nditer(updated_mat, op_flags=['readwrite']):
            if value != 0:
                value[...] = value + (1 / value)
        updated_mat = updated_mat / abs(np.max(updated_mat))
        error = abs(np.mean(updated_mat - scoremat))
        sys.stdout.write('Current error: ' + str(error) + '\n')
        sys.stdout.flush()
        scoremat = updated_mat
        iters += 1
    if iters == iterations:
        sys.stdout.write('Warning: algorithm did not converge.' + '\n')
        sys.stdout.flush()
    for i in range(iters):
        flowmat = np.linalg.matrix_power(flowmat, 2)
        # expansion step
        # squaring the matrix without normalisation
        # is equal to a a Galton-Watson branching process
        flowmat = flowmat / abs(np.max(flowmat))
    return scoremat, flowmat

