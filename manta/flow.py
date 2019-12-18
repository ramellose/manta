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
import multiprocessing as mp
from manta.utils import _partial_diffusion
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


def partial_diffusion(graph, iterations, limit, subset, ratio, permutations, verbose, core=4):
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
    :param core: Number of cores to use, 4 by default
    :return: score matrix, memory effect, initial diffusion matrix
    """
    # scoremat indices are ordered by graph.nodes()
    scoremat = nx.to_numpy_array(graph)
    mat_index = {list(graph.nodes)[i]: i for i in range(len(graph.nodes))}
    nums = int(len(graph)*subset)  # fraction of edges in subnetwork set to 0
    # this fraction can be set to 0.8 or higher, gives good results
    # results in file manta_ratio_perm.csv
    subnum = len(graph)
    if permutations:
        subnum = permutations  # number of subnetworks generated
    values = {'graph': graph,
              'scoremat': scoremat,
              'mat_index': mat_index,
              'nums': nums,
              'limit': limit,
              'iterations': iterations}
    values = [values for x in range(subnum)]
    pool = mp.Pool(core)
    result = pool.map(_partial_diffusion, values)
    pool.close()
    for x in range(len(result)):
        # ocassionally, the iteration converges to zero, so this function
        # then retries the iteration
        if not result[x]:
            scoremat = result[x]
            while not scoremat:
                scoremat = _partial_diffusion(values[0])
            result[x] = scoremat
    posfreq = np.zeros((len(graph), len(graph)))
    negfreq = np.zeros((len(graph), len(graph)))
    for b in range(subnum):
        posfreq[result[b] > 0] += 1
        negfreq[result[b] < 0] += 1
    # we count how many times specific values in matrix have
    # been assigned positive or negative values
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


