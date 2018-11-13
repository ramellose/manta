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
import numpy as np


def rewire_graph(graph, error):
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
    :param graph: Original graph to rewire.
    :param error: Fraction of edges to rewire.
    :return: Rewired NetworkX graph
    """
    model = graph.copy()
    swaps = len(model.nodes) * error
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


def perm_graph(graph, limit, permutations, percentile, pos, neg, error):
    """
    Calls the rewire_graph function;
    returns reliability scores of edge centrality scores.
    Scores close to 100 imply that the scores are robust to perturbation.
    Reliability scores as proposed by:
    Frantz, T. L., & Carley, K. M. (2017).
    Reporting a network’s most-central actor with a confidence level.
    Computational and Mathematical Organization Theory, 23(2), 301-312.
    :param graph: NetworkX graph of a microbial association network.
    :param matrix: Outcome of diffusion process from cluster_graph.
    :param limit: Error limit for matrix convergence.
    :param permutations: Number of permutations to carry out. If 0, no bootstrapping is done.
    :param percentile: Determines percentile of hub species to return.
    :param pos: List of edges in the upper percentile. (e.g. positive hubs)
    :param neg: List of edges in the lower percentile. (e.g. negative hubs)
    :param error: Fraction of edges to rewire for reliability metric.
    :return: List of reliability scores.
    """
    perms = list()
    for i in range(permutations):
        permutation = rewire_graph(graph, error)
        adj = diffusion(graph=permutation, iterations=3, norm=False)[0]
        perms.append(adj)
        sys.stdout.write('Permutation iteration ' + str(i) + '\n')
        sys.stdout.flush()
    posmatches = dict()
    negmatches = dict()
    for hub in pos:
        posmatches[hub] = 0
    for hub in neg:
        negmatches[hub] = 0
    for perm in perms:
        negthresh = np.percentile(perm, percentile)
        posthresh = np.percentile(perm, 100 - percentile)
        permneg = list(map(tuple, np.argwhere(perm <= negthresh)))
        permpos = list(map(tuple, np.argwhere(perm >= posthresh)))
        matches = set(pos).intersection(permpos)
        for match in matches:
            posmatches[match] += 1
        matches = set(neg).intersection(permneg)
        for match in matches:
            negmatches[match] += 1
    reliability = posmatches.copy()
    reliability.update(negmatches)
    reliability = {k: (v/permutations) for k,v in reliability.items()}
    # p value equals number of permutations that exceeds / is smaller than matrix values
    return reliability


def diffuse_graph(graph, limit=0.00001, iterations=50):
    """
    Wrapper for the diffusion process.
    If the diffusion algorithm fails to converge,
    it retries with a lower error setting.
    :param graph: NetworkX graph of a microbial assocation network.
    :param limit: Error limit for matrix convergence.
    :param iterations: Maximum number of iterations to carry out.
    :return: score matrix
    """
    limits = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    convergence = False
    i = next(x[0] for x in enumerate(limits) if x[1] > limit)
    start = True
    # finds nearest value in limits that is larger than specified limit
    while not convergence and i < len(limits):
        if start:
            result = diffusion(graph=graph, limit=limit, iterations=iterations)
            start = False
        else:
            result = diffusion(graph, limits[i], iterations)
        convergence = result[1]
        scoremat = result[0]
        i += 1
    if i == len(limits):
        sys.stdout.write('Warning: no convergence possible.' + '\n' +
                         'Results are unlikely to be reliable. ' + '\n')
        sys.stdout.flush()
    return scoremat


def diffusion(graph, iterations, limit=0.00001, norm=True):
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
    :param iterations: Maximum number of iterations to carry out.
    :param limit: Error limit for matrix convergence. Arbitrary if norm set to false.
    :param norm: Normalize values so they converge to -1 or 1.
    :return: score matrix
    """
    scoremat = nx.to_numpy_array(graph)  # numpy matrix is deprecated
    # extract the maximum and minimum value
    # if either value has decimals, matrix is rescaled from 1 to 10
    if not (np.max(scoremat)).is_integer() or not (np.min(scoremat)).is_integer():
        negmin = np.max(scoremat[scoremat < 0])
        negmax = np.min(scoremat[scoremat < 0])
        posmax = np.max(scoremat[scoremat > 0])
        posmin = np.min(scoremat[scoremat > 0])
        for row in scoremat:
            for j, entry in enumerate(row):
                if entry > 0:
                    row[j] = 9 * (entry - posmin)/(posmax-posmin) + 1
                elif entry < 0:
                    row[j] = -9 * (entry - negmin)/(negmax-negmin) - 1
    # if the 'weight' property of the graph is set correctly
    # weight in the adj graph should equal this
    error = 1
    prev_error = 0
    iters = 0
    convergence = True
    while error > limit and iters < iterations:
        updated_mat = np.linalg.matrix_power(scoremat, 2)
        #updated_mat = deepcopy(scoremat)
        #for entry in np.nditer(updated_mat, op_flags=['readwrite']):
           # entry[...] = entry ** 2
        # expansion step
        # squaring the matrix without normalisation
        if norm:
            # negative values should be scaled separately
            # if there are far higher positive values,
            # negative values can disappear
            # old: updated_mat = updated_mat / abs(np.max(updated_mat))
            updated_mat[updated_mat > 0] = \
                updated_mat[updated_mat > 0] / \
                abs(np.max(updated_mat[updated_mat > 0]))
            updated_mat[updated_mat < 0] = \
                updated_mat[updated_mat < 0] / \
                abs(np.min(updated_mat[updated_mat < 0]))
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
        if norm:
            updated_mat[updated_mat > 0] = \
                updated_mat[updated_mat > 0] / \
                abs(np.max(updated_mat[updated_mat > 0]))
            updated_mat[updated_mat < 0] = \
                updated_mat[updated_mat < 0] / \
                abs(np.min(updated_mat[updated_mat < 0]))
        error = abs(np.mean(updated_mat - scoremat))
        if norm:
            sys.stdout.write('Current error: ' + str(error) + '\n')
            sys.stdout.flush()
        scoremat = updated_mat
        iters += 1
    if iters == iterations and norm:
        sys.stdout.write('Warning: algorithm did not converge.' + '\n' +
                         'Retrying with higher error limit. ' + '\n')
        sys.stdout.flush()
        convergence = False
    return scoremat, convergence

