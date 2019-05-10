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
            with np.errstate(divide='raise', invalid='raise'):
                try:
                    updated_mat = updated_mat / abs(np.max(updated_mat))
                except FloatingPointError:
                    # this indicates the matrix is busy converging to 0
                    # in that case, we do the same as with the memory effect
                    if verbose:
                        sys.stdout.write('Matrix converging to zero.' + '\n' +
                                         'Clustering with partial network. ' + '\n')
                        sys.stdout.flush()
                        convergence = True
                        break
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
        if norm and verbose:
            sys.stdout.write('Current error: ' + str(error) + '\n')
            sys.stdout.flush()
        try:
            if (error_2 / error > 0.99) and (error_2 / error < 1.01) and not memory:
                # if there is a flip-flop state, the error will alternate between two values
                if verbose:
                    sys.stdout.write('Detected memory effect at iteration: ' + str(iters) + '\n')
                    sys.stdout.flush()
                memory = True
                iterations = iters + 5
        except RuntimeWarning:
            convergence = True
            if verbose:
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
    subnum = permutations  # number of subnetworks generated
    b = 0
    while b < subnum:
        # only add 1 to b if below snippet completes
        # otherwise, keep iterating
        # supposed to catch runtime warnings
        # runtime warnings from the diffusion function are a problem
        # runtime warnings here are likely a result of the permutation
        try:
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
                # expansion step
                # in the MCL implementation, the rows are normalized to sum to 1
                # this creates a column stochastic matrix
                # here, we normalize by dividing with absolute largest value
                with np.errstate(divide='raise', invalid='raise'):
                    try:
                        updated_mat = updated_mat / abs(np.max(updated_mat))
                    except FloatingPointError:
                        # this indicates the matrix is busy converging to 0
                        # in that case, we do the same as with the memory effect
                        if verbose:
                            sys.stdout.write('Matrix converging to zero.' + '\n' +
                                             'Clustering with partial network. ' + '\n')
                            sys.stdout.flush()
                            break                # updated_mat[updated_mat > 0] = \
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
                            if verbose:
                                sys.stdout.write('Warning: matrix overflow detected.' + '\n')
                                sys.stdout.flush()
                            break
                updated_mat = updated_mat / abs(np.max(updated_mat))
                error = abs(updated_mat - submat)[np.where(updated_mat != 0)] / abs(updated_mat[np.where(updated_mat != 0)])
                error = np.mean(error) * 100
                if (error_2 / error > 0.99) and (error_2 / error < 1.01) and not memory:
                    # if there is a flip-flop state, the error will alternate between two values
                    max_iters = iters + 5
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
            b += 1
            sys.stdout.write("Permutation " + str(b))
            sys.stdout.flush()
        except RuntimeWarning:
            pass
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

