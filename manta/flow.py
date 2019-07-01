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

    Normally, memory effects should not be detected because these only happen
    if the graph is unbalanced.
    The Harary algorithm is used to check whether graphs are unbalanced.
    If they are, this diffusion process should not be run.

    Parameters
    ----------
    :param graph: NetworkX graph of a microbial assocation network
    :param iterations: Maximum number of iterations to carry out
    :param limit: Percentage in error decrease until matrix is considered converged
    :param verbose: Verbosity level of function
    :param norm: Normalize values so they converge to -1 or 1
    :param inflation: Carry out network diffusion with/without inflation
    :return: score matrix, memory effect
    """
    scoremat = nx.to_numpy_array(graph)  # numpy matrix is deprecated
    error = 100
    diffs = list()
    iters = 0
    memory = False
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
            memory = True
            if verbose:
                logger.info('Matrix converged to zero.' + '\n' +
                            'Clustering with partial network. ')
        error_2 = error_1
        error_1 = error
        scoremat = updated_mat
        if iters == 0:
            firstmat = updated_mat
        iters += 1
        diffs.append(scoremat)
    if memory or np.any(np.isnan(scoremat)):
        diffs = diffs[-5:]
        scoremat = firstmat
    if iters == iterations:
        convergence = True
        if verbose:
            logger.info('Matrix failed to converge.' + '\n' +
                        'Clustering with partial network. ')
    return scoremat, memory, diffs


def partial_diffusion(graph, iterations, limit, subset, ratio, permutations, verbose):
    """
    Partial diffusion process for generation of scoring matrix.
    Some matrices may be unable to reach convergence
    or enter a flip-flopping state.
    A partial diffusion process can still discover relationships
    between unlinked taxa when this is not possible.

    The partial diffusion process takes a random subgraph and
    checks whether this subgraph is balanced. If it is, the full diffusion process
    is carried out. Otherwise, one iteration is carried out.

    Parameters
    ----------
    :param graph: NetworkX graph of a microbial assocation network
    :param iterations: Maximum number of iterations to carry out
    :param limit: Percentage in error decrease until matrix is considered converged
    :param subset: Fraction of edges used in subsetting procedure
    :param ratio: Ratio of positive / negative edges required for edge stability
    :param permutations: Number of permutations for network subsetting
    :param verbose: Verbosity level of function
    :return: score matrix, memory effect, initial diffusion matrix
    """
    # scoremat indices are ordered by graph.nodes()
    scoremat = nx.to_numpy_array(graph)
    mat_index = {list(graph.nodes)[i]: i for i in range(len(graph.nodes))}
    nums = int(len(graph)*subset)  # fraction of edges in subnetwork set to 0
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
        indices = sample(graph.nodes, nums)
        num_indices = [mat_index[i] for i in indices]
        subgraph = nx.subgraph(graph, indices)
        # we randomly sample from the nodes and create a subgraph from this
        # this can give multiple connected components
        balanced = harary_components(subgraph, verbose=False)
        # if there is a balanced component, carry out network flow separately
        balanced_matrix = np.copy(scoremat)
        if any(balanced.values()):
            balanced_components = [x for x in balanced if balanced[x]]
            for component in balanced_components:
                if len(component) > 0.1 ( len(graph)):
                    # get score matrix for balanced component
                    partial_score = diffusion(graph=component, limit=limit,
                                              iterations=iterations, verbose=False)[0]
                    # map score matrix to balanced_matrix
                    for i in range(partial_score.shape[0]):
                        for j in range(partial_score.shape[0]):
                            node_ids = [list(component.nodes)[i],
                                        list(component.nodes)[j]]
                            mat_ids = [mat_index[node] for node in node_ids]
                            balanced_matrix[mat_ids[0], mat_ids[1]] = partial_score[i,j]
            # carry out 1 step propagation on entire matrix
        submat = np.copy(scoremat)
        submat[num_indices, :] = 0
        submat[:, num_indices] = 0
        # if there is no flip-flop state, the error will decrease after convergence
        updated_mat = np.linalg.matrix_power(submat, 2)
        updated_mat = updated_mat / np.max(abs(updated_mat))
        for value in np.nditer(updated_mat, op_flags=['readwrite']):
            if value != 0:
                value[...] = value + (1 / value)
        updated_mat = updated_mat / np.max(abs(updated_mat))
        # the permutation matrix is a combination of balanced components
        # and a propagation step (1-step expansion + inflation)
        updated_mat = updated_mat + balanced_matrix
        # normalize again
        updated_mat = updated_mat / np.max(abs(updated_mat))
        result.append(updated_mat)
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


def harary_components(graph, verbose):
    """
    This wrapper for the balance test can accept graphs
    that consist of multiple  connected components.

    :param graph: NetworkX graph
    :param verbose: Prints result of test to logger if True
    :return: Returns a dictionary with connected components as keys;
             components that are balanced have a True value.
    """
    all_components = [graph]
    component_balance = dict()
    if not nx.is_connected(graph):
        all_components = []
        component_generator = nx.connected_components(graph)
        for component in component_generator:
            if len(component) > 1:
                all_components.append(nx.subgraph(graph, component))
    for component in all_components:
        component_balance[component] = harary_balance(component)
    if verbose:
        if all(component_balance.values()) and len(component_balance.values()) == 1:
            logger.info("Graph is balanced.")
        elif all(component_balance.values()):
            logger.info("All connected components of the graph are balanced.")
        elif any(component_balance.values()):
            logger.info("At least one connected component of the graph"
                        " is balanced.")
        else:
            logger.info("Graph is unbalanced.")
    return component_balance


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
    :return: Returns a dictionary with connected components as keys;
             components that are balanced have a True value.
    """
    # Step 1: Select a spanning tree T
    balance = True
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
        # it is possible that all lines have been tested;
        # in this case, the graph is balanced for sure
        if not all(lines.values()):
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
    return balance


