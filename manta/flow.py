#!/usr/bin/env python

"""
Cluster assignment by manta makes use of a scoring matrix.
Functions in this module can generate such a scoring matrix.

First, an empty adjacency matrix is instantiated.
Then, the following steps are carried out until an error threshold is reached:

1. Raise the matrix to power 2
2. Normalize matrix by absolute maximum value
3. Add 1 / value for each value in matrix
4. Normalize matrix by absolute maximum value
5. Calculate error by subtracting previous matrix from current matrix

If a network consists of clusters connected by positive edges,
it will frequently fail to reach the error threshold.
This can happen because of two reasons: firstly, the scoring matrix enters a flip-flop state.
Secondly, the matrix converges to zero.
Flip-flop states are detected by comparing the error to the error 2 iterations ago.
If this error is 99% identical, manta continues clustering with a partial iteration strategy.
Zero convergence is detected when the threshold of the 99th percentile
is below 0.00000001.
In either of these cases, manta proceeds with a partial iteration strategy.
This strategy repeats the same steps as the normal network flow strategy,
but does so on a subset of the network.
The scoring matrix is then recovered from stable edges;
stable edges are defined as edges that have only positive or negative values during a minimum fraction of permutations.
Those permutations are summed together to give positions in the scoring matrix.

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import sys
from random import sample
import networkx as nx
import numpy as np
import os
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


def diffusion(graph, iterations, limit, verbose, norm=True, inflation=True):
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

    Parameters
    ----------
    :param graph: NetworkX graph of a microbial assocation network
    :param iterations: Maximum number of iterations to carry out
    :param limit: Percentage in error decrease until matrix is considered converged
    :param verbose: Verbosity level of function
    :param norm: Normalize values so they converge to -1 or 1
    :param inflation: Carry out network diffusion with/without inflation
    :return: score matrix, memory effect, initial diffusion matrix
    """
    scoremat = nx.to_numpy_array(graph)  # numpy matrix is deprecated
    error = 100
    diffs = list()
    iters = 0
    memory = False
    convergence = False
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
            if np.max(updated_mat) == 0:
                    # this indicates the matrix is busy converging to 0
                    # in that case, we do the same as with the memory effect
                    if verbose:
                        logger.info('Matrix converging to zero.' + '\n' +
                                    'Clustering with partial network. ')
                    convergence = True
                    break
            else:
                updated_mat = updated_mat / np.max(abs(updated_mat))
            # updated_mat[updated_mat > 0] = \
            #    updated_mat[updated_mat > 0] / \
            #    abs(np.max(updated_mat[updated_mat > 0]))
            # updated_mat[updated_mat < 0] = \
            #    updated_mat[updated_mat < 0] / \
            #    abs(np.min(updated_mat[updated_mat < 0]))
            # the above code scales negative and positive values separately
            # interestingly, the matrix does not separate correctly if used
        # this indicates the matrix is busy converging to 0
        # in that case, we do the same as with the memory effect
        if inflation:
            for value in np.nditer(updated_mat, op_flags=['readwrite']):
                if value != 0:
                    # normally, there is an inflation step; values are raised to a power
                    # with this normalisation, the inflation step causes
                    # the algorithm to converge to 0
                    # we need above-0 values to converge to -1, and the rest to 1
                    value[...] = value + (1/value)

        if norm:
            updated_mat = updated_mat / np.max(abs(updated_mat))
        error = abs(updated_mat - scoremat)[np.where(updated_mat != 0)] / abs(updated_mat[np.where(updated_mat != 0)])
        error = np.mean(error) * 100
        if norm and verbose:
            logger.info('Current error: ' + str(error))
        try:
            if (error_2 / error > 0.99) and (error_2 / error < 1.01) and not memory:
                # if there is a flip-flop state, the error will alternate between two values
                if verbose:
                    logger.info('Detected memory effect at iteration: ' + str(iters))
                memory = True
                iterations = iters + 5
            if np.isnan(error) and not memory:
                if verbose:
                    logger.info('Error calculation failed at iteraiton: ' + str(iters))
                memory = True
                iterations = iters + 5
        except RuntimeWarning:
            convergence = True
            if verbose:
                logger.info('Matrix converged to zero.' + '\n' +
                            'Clustering with partial network. ')
        error_2 = error_1
        error_1 = error
        scoremat = updated_mat
        if iters == 0:
            firstmat = updated_mat
        diffs.append(scoremat)
        iters += 1
    if memory or np.any(np.isnan(scoremat)):
        diffs = diffs[-5:]
        scoremat = firstmat
    if iters == iterations:
        convergence = True
        if verbose:
            logger.info('Matrix failed to converge.' + '\n' +
                        'Clustering with partial network. ')
    return scoremat, memory, convergence, diffs


def partial_diffusion(graph, iterations, limit, ratio, permutations, verbose):
    """
    Partial diffusion process for generation of scoring matrix.
    Some matrices may be unable to reach convergence
    or enter a flip-flopping state.
    A partial diffusion process can still discover relationships
    between unlinked taxa when this is not possible.
    Parameters
    ----------
    :param graph: NetworkX graph of a microbial assocation network
    :param iterations: Maximum number of iterations to carry out
    :param limit: Percentage in error decrease until matrix is considered converged
    :param ratio: Ratio of positive / negative edges required for edge stability
    :param permutations: Number of permutations for network subsetting
    :param verbose: Verbosity level of function
    :return: score matrix, memory effect, initial diffusion matrix
    """
    scoremat = nx.to_numpy_array(graph)  # numpy matrix is deprecated
    # this internal parameter is currently set to 80%; maybe test adjustments?
    # nonlinear behaviour of ratio implies that this var may not be so important
    nums = int(len(graph)/5)*4  # fraction of edges in subnetwork set to 0
    # this fraction can be set to 0.8 or higher, gives good results
    # results in file manta_ratio_perm.csv
    result = list()
    subnum = len(graph)
    if permutations:
        subnum = permutations  # number of subnetworks generated
    b = 0
    while b < subnum:
        # only add 1 to b if below snippet completes
        # otherwise, keep iterating
        # runtime warnings from the diffusion function are a problem
        # runtime warnings here are likely a result of the permutation
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
        error_1 = 1  # error steps 1 and 2 iterations back
        error_2 = 1  # detects flip-flop effect; normal clusters can also increase in error first
        while error > limit and iters < max_iters:
            # if there is no flip-flop state, the error will decrease after convergence
            updated_mat = np.linalg.matrix_power(submat, 2)
            if np.max(updated_mat) == 0:
                    # this indicates the matrix is busy converging to 0
                    # in that case, we do the same as with the memory effect
                    break
            else:
                updated_mat = updated_mat / np.max(abs(updated_mat))
            for value in np.nditer(updated_mat, op_flags=['readwrite']):
                if value != 0:
                    value[...] = value + (1/value)
            if not np.isnan(updated_mat).any():
                updated_mat = updated_mat / np.max(abs(updated_mat))
            else:
                break
            error = abs(updated_mat - submat)[np.where(updated_mat != 0)] / abs(updated_mat[np.where(updated_mat != 0)])
            error = np.mean(error) * 100
            if error != 0:
                if (error_2 / error > 0.99) and (error_2 / error < 1.01) and not memory:
                    # if there is a flip-flop state, the error will alternate between two values
                    max_iters = iters + 5
                    memory = True
            else:
                break
            error_2 = error_1
            error_1 = error
            submat = updated_mat
            if iters == 0:
                firstmat = updated_mat
            diffs.append(submat)
            iters += 1
        if memory:
            submat = firstmat
        if not np.isnan(submat).any():
            result.append(submat)
            b += 1
            if verbose:
                logger.info("Partial diffusion " + str(b))
    posfreq = np.zeros((len(graph), len(graph)))
    negfreq = np.zeros((len(graph), len(graph)))
    for b in range(subnum):
        posfreq[result[b] > 0] += 1
        negfreq[result[b] < 0] += 1
    # we count how many times specific values in matrix have
    # been assigned positive or negative values
    outcome = np.zeros((len(graph), len(graph)))
    # add pseudo count of 1 to prevent errors with zero divison
    posfreq += 1
    pos_results = np.where((posfreq - negfreq) / posfreq > ratio)
    negfreq += 1
    neg_results = np.where((negfreq - posfreq) / negfreq > ratio)
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


def harary_balance(graph):
    """
    Checks whether a graph is balanced according to Harary's theorem.
    Python implementation of the algorithm as described in the article below.
    Harary's algorithm quickly finds whether a signed graph is balanced.
    A signed graph is balanced if the product of edge signs around
    every circle is positive.

    Harary, F., & Kabell, J. A. (1980).
    A simple algorithm to detect balance in signed graphs.
    Mathematical Social Sciences, 1(1), 131-136.
    :param graph: NetworkX graph
    :return: True if the tree is balanced, False otherwise.
    """
    # Step 1: Select a spanning tree T
    tree = nx.algorithms.minimum_spanning_tree(graph)
    marks = dict.fromkeys(tree.nodes)
    lines = dict.fromkeys(graph.edges)
    # Step 2: Root T at an arbitrary point v0
    root = sample(tree.nodes, 1)
    # Step 3: Mark v0 positive
    marks[root[0]] = 1.0
    balance = True
    while not all(lines.values()):
        # Step 7: Is there a line that has not been tested?
        while not all(marks.values()):
            # Step 6: Is there a value that has not been tested?
            step4 = False
            while not step4:
                # Step 4: Select an unsigned point adjacent in T to a signed point
                # get all unsigned nodes
                unsigned = [i for i in marks if not marks[i]]
                # find an unsigned node that is a neighbour of a signed node
                signed = [i for i in marks if marks[i]]
                for unsign in unsigned:
                    match = set(nx.neighbors(tree, unsign)).intersection(signed)
                    if len(match) > 0:
                        step4 = True
                        match = sample(match, 1)[0]
                        # Step 5: Label the selected point with the product
                        # of the sign of the previously point to which it is
                        # adjacent in T and the sign of the line joining them
                        marks[unsign] = marks[match] * tree[match][unsign]['weight']
                        try:
                            lines[(unsign, match)] = True
                        except KeyError:
                            lines[(match, unsign)] = True
        # Step 8: Select an untested line of S - E(T)
        untested = sample([i for i in lines if not lines[i]], 1)[0]
        # Step 9: Is the sign of the selected line equal to the product
        # of the signs of its two points?
        untested_sign = marks[untested[0]] * marks[untested[1]]
        if untested_sign == graph.edges[untested]['weight']:
            lines[untested] = True
        else:
            # Step 10: Stop, S is not balanced
            balance = False
            break
    if balance:
        logger.info("Graph is balanced according to Harary's theorem!")
    else:
        logger.info("Graph is unbalanced according to Harary's theorem! ")
    return balance




