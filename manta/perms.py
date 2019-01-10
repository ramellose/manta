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
from random import choice, sample
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


def perm_graph(graph, permutations, percentile, pos, neg, error):
    """
    Calls the rewire_graph function;
    returns reliability scores of edge centrality scores.
    Scores close to 100 imply that the scores are robust to perturbation.
    Reliability scores as proposed by:
    Frantz, T. L., & Carley, K. M. (2017).
    Reporting a network’s most-central actor with a confidence level.
    Computational and Mathematical Organization Theory, 23(2), 301-312.
    :param graph: NetworkX graph of a microbial association network.
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
        adj = diffusion(graph=permutation, iterations=3, norm=False, msg=False)[0]
        perms.append(adj)
        sys.stdout.write('Permutation ' + str(i) + '\n')
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
    reliability = {k: (v/permutations) for k, v in reliability.items()}
    # p value equals number of permutations that exceeds / is smaller than matrix values
    return reliability


def diffusion(graph, iterations, limit=2, norm=True, inflation=True, msg=False):
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

    If a memory effect is detected, not the outcome matrix but the first iteration is returned.
    Additionally, 5 iterations after the flip-flop state has been reached are calculated.

    :param graph: NetworkX graph of a microbial assocation network
    :param iterations: Maximum number of iterations to carry out
    :param limit: Percentage in error decrease until matrix is considered converged
    :param norm: Normalize values so they converge to -1 or 1
    :param inflation: Carry out network diffusion with/without inflation
    :param msg: If true, print error size per iteration
    :return: score matrix, memory effect, initial diffusion matrix
    """
    scoremat = nx.to_numpy_array(graph)  # numpy matrix is deprecated
    error = 100
    diffs = list()
    iters = 0
    memory = False
    convergence = True
    error_1 = 1  # error steps 1 and 2 iterations back
    error_2 = 1  # detects flip-flop effect; normal clusters can also increase in error first
    while error > limit and iters < iterations:
        # if there is no flip-flop state, the error will decrease after convergence
        updated_mat = np.linalg.matrix_power(scoremat, 2)
        # updated_mat = deepcopy(scoremat)
        # for entry in np.nditer(updated_mat, op_flags=['readwrite']):
        # entry[...] = entry ** 2
        # expansion step
        # squaring the matrix without normalisation
        if norm:
            # the flow matrix describes flow and is output to compute centralities
            # in the MCL implementation, the rows are normalized to sum to 1
            # this creates a column stochastic matrix
            # here, we normalize by dividing with absolute largest value
            updated_mat = updated_mat / abs(np.max(updated_mat))
            # updated_mat[updated_mat > 0] = \
            #    updated_mat[updated_mat > 0] / \
            #    abs(np.max(updated_mat[updated_mat > 0]))
            # updated_mat[updated_mat < 0] = \
            #    updated_mat[updated_mat < 0] / \
            #    abs(np.min(updated_mat[updated_mat < 0]))
            # the above code scales negative and positive values separately
            # interestingly, the matrix does not separate correctly if used
        # we need to check the percentile;
        # if over 99% of values are close to 0,
        # this indicates the matrix is busy converging to 0
        # in that case, we do the same as with the memory effect
        if np.percentile(updated_mat, 99) < 0.00000001:
            sys.stdout.write('Matrix converging to zero.' + '\n' +
                             'Clustering with partial network. ' + '\n')
            sys.stdout.flush()
            convergence = True
            break
        if inflation:
            for value in np.nditer(updated_mat, op_flags=['readwrite']):
                if value != 0:
                    # normally, there is an inflation step; values are raised to a power
                    # with this normalisation, the inflation step causes
                    # the algorithm to converge to 0
                    # we need above-0 values to converge to -1, and the rest to 1
                    # previous: value * abs(value)
                    # this inflation does not result in desired sparsity
                    try:
                        value[...] = value + (1/value)
                    except RuntimeWarning:
                        sys.stdout.write('Warning: matrix overflow detected.' + '\n' +
                                         'Please retry with a higher error limit. ' + '\n')
                        break
        if norm:
                updated_mat = updated_mat / abs(np.max(updated_mat))
        error = abs(updated_mat - scoremat)[np.where(updated_mat != 0)] / abs(updated_mat[np.where(updated_mat != 0)])
        error = np.mean(error) * 100
        if norm and msg:
            sys.stdout.write('Current error: ' + str(error) + '\n')
            sys.stdout.flush()
        try:
            if (error_2 / error > 0.99) and (error_2 / error < 1.01) and not memory:
                # if there is a flip-flop state, the error will alternate between two values
                sys.stdout.write('Detected memory effect at iteration: ' + str(iters) + '\n')
                sys.stdout.flush()
                memory = True
                iterations = iters + 5
        except RuntimeWarning:
            convergence = True
            sys.stdout.write('Matrix converged to zero.' + '\n' +
                             'Clustering with partial network. ' + '\n')
            sys.stdout.flush()
        error_2 = error_1
        error_1 = error
        scoremat = updated_mat
        if iters == 0:
            firstmat = updated_mat
        diffs.append(scoremat)
        iters += 1
    if memory:
        diffs = diffs[-5:]
        scoremat = firstmat
    return scoremat, memory, convergence, diffs


def partial_diffusion(graph, iterations, limit=2):
    """
    Partial diffusion process for generation of scoring matrix.
    Some matrices may be unable to reach convergence
    or enter a flip-flopping state.
    A partial diffusion process can still discover relationships
    between unlinked taxa when this is not possible.
    :param graph: NetworkX graph of a microbial assocation network
    :param iterations: Maximum number of iterations to carry out
    :param limit: Percentage in error decrease until matrix is considered converged
    :return: score matrix, memory effect, initial diffusion matrix
    """
    scoremat = nx.to_numpy_array(graph)  # numpy matrix is deprecated
    nums = int(len(graph)/5)*4  # fraction of edges in subnetwork set to 0
    result = list()
    subnum = 100  # number of subnetworks generated
    for b in range(subnum):  # 100 is arbitrarily chosen
        indices = sample(range(len(graph)), nums)
        # we randomly sample from the indices and create a subgraph from this
        submat = np.copy(scoremat)
        submat[indices, :] = 0
        submat[:, indices] = 0
        error = 100
        diffs = list()
        iters = 0
        max_iters = iterations
        memory = False
        convergence = True
        error_1 = 1  # error steps 1 and 2 iterations back
        error_2 = 1  # detects flip-flop effect; normal clusters can also increase in error first
        while error > limit and iters < max_iters:
            # if there is no flip-flop state, the error will decrease after convergence
            updated_mat = np.linalg.matrix_power(submat, 2)
            # updated_mat = deepcopy(scoremat)
            # for entry in np.nditer(updated_mat, op_flags=['readwrite']):
            # entry[...] = entry ** 2
            # expansion step
            # squaring the matrix without normalisation
            # the flow matrix describes flow and is output to compute centralities
            # in the MCL implementation, the rows are normalized to sum to 1
            # this creates a column stochastic matrix
            # here, we normalize by dividing with absolute largest value
            updated_mat = updated_mat / abs(np.max(updated_mat))
            # updated_mat[updated_mat > 0] = \
            #    updated_mat[updated_mat > 0] / \
            #    abs(np.max(updated_mat[updated_mat > 0]))
            # updated_mat[updated_mat < 0] = \
            #    updated_mat[updated_mat < 0] / \
            #    abs(np.min(updated_mat[updated_mat < 0]))
            # the above code scales negative and positive values separately
            # interestingly, the matrix does not separate correctly if used
            # we need to check the percentile;
            # if over 99% of values are close to 0,
            # this indicates the matrix is busy converging to 0
            # in that case, we do the same as with the memory effect
            if np.percentile(updated_mat, 99) < 0.00000001:
                convergence = True
                break
            for value in np.nditer(updated_mat, op_flags=['readwrite']):
                if value != 0:
                    # normally, there is an inflation step; values are raised to a power
                    # with this normalisation, the inflation step causes
                    # the algorithm to converge to 0
                    # we need above-0 values to converge to -1, and the rest to 1
                    # previous: value * abs(value)
                    # this inflation does not result in desired sparsity
                    try:
                        value[...] = value + (1/value)
                    except RuntimeWarning:
                        sys.stdout.write('Warning: matrix overflow detected.' + '\n')
                        break
            updated_mat = updated_mat / abs(np.max(updated_mat))
            error = abs(updated_mat - submat)[np.where(updated_mat != 0)] / abs(updated_mat[np.where(updated_mat != 0)])
            error = np.mean(error) * 100
            try:
                if (error_2 / error > 0.99) and (error_2 / error < 1.01) and not memory:
                    # if there is a flip-flop state, the error will alternate between two values
                    memory = True
                    max_iters = iters + 5
            except RuntimeWarning:
                convergence = True
                sys.stdout.write('Matrix converged to zero.' + '\n' +
                                 'Skipping this iteration. ' + '\n')
                sys.stdout.flush()
            error_2 = error_1
            error_1 = error
            submat = updated_mat
            if iters == 0:
                firstmat = updated_mat
            diffs.append(submat)
            iters += 1
        if memory:
            submat = firstmat
        result.append(submat)
    posfreq = np.zeros((len(graph), len(graph)))
    negfreq = np.zeros((len(graph), len(graph)))
    for b in range(subnum):
        posfreq[result[b] > 0] += 1
        negfreq[result[b] < 0] += 1
    # we count how many times specific values in matrix have
    # been assigned positive or negative values
    outcome = np.zeros((len(graph), len(graph)))
    pos_results = np.where(posfreq > 3*negfreq)
    neg_results = np.where(negfreq > 3*posfreq)
    # if the number of positive/negative values is large enough,
    # this edge can be considered stable
    # the section below adds only positive values
    # for edges that are stable (negatively)
    outcome = np.zeros((len(graph), len(graph)))
    for b in range(subnum):
        pos_sums = result[b][pos_results]
        pos_sums[pos_sums < 0] = 0
        outcome[pos_results] += pos_sums
        neg_sums = result[b][neg_results]
        neg_sums[neg_sums > 0] = 0
        outcome[neg_results] += neg_sums
    outcome = outcome / abs(np.max(outcome))
    return outcome, result