#!/usr/bin/env python

"""
Centrality scores and cluster assignments can be tested for their reliability.
The idea behind this test is that random rewiring should, to some extent,
preserve the most central structures of the original graph.
We cannot know which edges are true positives and which ones are false positives,
but we do expect that global network properties are retained despite changes in identified associations.

First, null models are generated from the original graph.
These models are rewired copies: edge degree and connectedness are preserved.
Weight is assigned randomly by sampling with replacement from the original weight scores.
For each of these networks, the diffusion iterations specified in cluster_sparse are repeated
as many times as for the original network. The outcome is then a matrix of diffusion scores.

With these matrices, the reliability can be estimated. This error specifies how variable
the assignments are expected to be.

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import networkx as nx
import numpy as np
from manta.flow import diffusion
from manta.cluster import cluster_graph
from scipy.stats import binom_test, norm
from random import choice
import sys


def central_edge(graph, percentile, permutations, error, verbose):
    """
    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub associations.

    If the permutation number is set to a value above 0, the centrality values are tested against permuted graphs.

    The fraction of positive edges and negative edges is based on the ratio between
    positive and negative weights in the network.

    Hence, a network with 90 positive weights and 10 negative weights will have 90% positive hubs returned.

    Parameters
    ----------
    :param graph: NetworkX graph of a microbial association network.
    :param percentile: Determines percentile of hub species to return.
    :param permutations: Number of permutations to carry out. If 0, no permutation test is done.
    :param error: Fraction of edges to rewire for reliability metric.
    :param verbose: Verbosity level of function
    :return: Networkx graph with hub ID / p-value as node property.
    """
    scoremat = diffusion(graph, limit=2, iterations=3, norm=False, verbose=verbose)[0]
    negthresh = np.percentile(scoremat, percentile)
    posthresh = np.percentile(scoremat, 100-percentile)
    neghubs = list(map(tuple, np.argwhere(scoremat <= negthresh)))
    poshubs = list(map(tuple, np.argwhere(scoremat >= posthresh)))
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    if permutations > 0:
        score = perm_edges(graph, percentile=percentile, permutations=permutations,
                           pos=poshubs, neg=neghubs, error=error)
    # need to make sure graph is undirected
    graph = nx.to_undirected(graph)
    # initialize empty dictionary to store edge ID
    edge_vals = dict()
    edge_scores = dict()
    # need to convert matrix index to node ID
    for edge in neghubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'negative hub'
        if permutations > 0 and score is not None:
            edge_scores[(node1, node2)] = score[(adj_index[node1], adj_index[node2])]
    for edge in poshubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'positive hub'
        if permutations > 0 and score is not None:
            edge_scores[(node1, node2)] = score[(adj_index[node1], adj_index[node2])]
    nx.set_edge_attributes(graph, values=edge_vals, name='hub')
    if permutations > 0 and score is not None:
        nx.set_edge_attributes(graph, values=edge_scores, name='reliability score')


def central_node(graph):
    """
    Given a graph with hub edges assigned (see central_edge),
    this function checks whether a node is significantly more connected
    to edges with high scores than expected by chance.
    The p-value is calculated with a binomial test.
    Edge sign is ignored; hubs can have both positive and negative
    edges.

    Parameters
    ----------
    :param graph: NetworkX graph with edge centrality scores assigned
    :return: NetworkX graph with hub centrality for nodes
    """
    edges = nx.get_edge_attributes(graph, "hub")
    hubs = list()
    for edge in edges:
        hubs.append(edge[0])
        hubs.append(edge[1])
    hubs = list(set(hubs))
    sighubs = dict()
    pvals = dict()
    for node in hubs:
        hub_edges = 0
        for edge in graph[node]:
            if 'hub' in graph[node][edge]:
                hub_edges += 1
        # given that some of the edges
        # this is compared to the total edge number of the node
        # probability is calculated by dividing total number of hub edges in graph
        # by total number of edges in graph
        pval = binom_test(hub_edges, len(graph[node]),
                          (len(edges)/len(graph.edges)), alternative='greater')
        if pval < 0.05:
            sighubs[node] = 'hub'
            pvals[node] = float(pval)
    nx.set_node_attributes(graph, values=sighubs, name='hub')
    nx.set_node_attributes(graph, values=pvals, name='hub p-value')


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

    Part of the rewire_graph function has been adapted from the original
    NetworkX double_edge_swap function. The adapted version also swaps edge weights.

    License
    =======
    NetworkX is distributed with the 3-clause BSD license.
    ::
       Copyright (C) 2004-2018, NetworkX Developers
       Aric Hagberg <hagberg@lanl.gov>
       Dan Schult <dschult@colgate.edu>
       Pieter Swart <swart@lanl.gov>
       All rights reserved.
       Redistribution and use in source and binary forms, with or without
       modification, are permitted provided that the following conditions are
       met:
         * Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.
         * Redistributions in binary form must reproduce the above
           copyright notice, this list of conditions and the following
           disclaimer in the documentation and/or other materials provided
           with the distribution.
         * Neither the name of the NetworkX Developers nor the names of its
           contributors may be used to endorse or promote products derived
           from this software without specific prior written permission.
       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
       "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
       A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
       OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
       SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
       LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
       THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
       (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
       OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Parameters
    ----------
    :param graph: Original graph to rewire.
    :param error: Fraction of edges to rewire.
    :return: Rewired NetworkX graph
    """
    model = graph.copy(as_view=False).to_undirected(as_view=False)
    swaps = round(len(model.nodes) * error)
    swapfail = False
    max_tries = len(model.nodes) * 10
    try:
        n = 0
        swapcount = 0
        keys, degrees = zip(*model.degree())  # keys, degree
        cdf = nx.utils.cumulative_distribution(degrees)  # cdf of degree
        discrete_sequence = nx.utils.discrete_sequence
        while swapcount < swaps:
            #        if random.random() < 0.5: continue # trick to avoid periodicities?
            # pick two random edges without creating edge list
            # choose source node indices from discrete distribution
            (ui, xi) = discrete_sequence(2, cdistribution=cdf)
            if ui == xi:
                continue  # same source, skip
            u = keys[ui]  # convert index to label
            x = keys[xi]
            # choose target uniformly from neighbors
            v = choice(list(model[u]))
            y = choice(list(model[x]))
            if v == y:
                continue  # same target, skip
            if (x not in model[u]) and (y not in model[v]):  # don't create parallel edges
                weight_uv = model.edges[u, v]['weight']
                weight_xy = model.edges[x, y]['weight']
                model.add_edge(u, x)
                model.edges[u, x]['weight'] = weight_uv
                model.add_edge(v, y)
                model.edges[v, y]['weight'] = weight_xy
                model.remove_edge(u, v)
                model.remove_edge(x, y)
                swapcount += 1
            if n >= max_tries:
                e = ('Maximum number of swap attempts (%s) exceeded ' % n +
                     'before desired swaps achieved (%s).' % swaps)
                raise nx.NetworkXAlgorithmError(e)
            n += 1
    except nx.exception.NetworkXAlgorithmError:
        sys.stdout.write('Cannot permute this network fraction. ' + '\n' +
                         'Please choose a lower error parameter, or avoid calculating a centrality score. ' + '\n')
        sys.stdout.flush()
        swapfail = True
    # edge_weights = list()
    # for edge in graph.edges:
    #     edge_weights.append(graph[edge[0]][edge[1]]['weight'])
    # random_weights = dict()
    # for edge in model.edges:
    #     random_weights[edge] = choice(edge_weights)
    # nx.set_edge_attributes(model, random_weights, 'weight')
    return model, swapfail


def perm_edges(graph, permutations, percentile, pos, neg, error):
    """
    Calls the rewire_graph function;
    returns reliability scores of edge centrality scores.
    Scores close to 100 imply that the scores are robust to perturbation.
    Reliability scores as proposed by:
    Frantz, T. L., & Carley, K. M. (2017).
    Reporting a network’s most-central actor with a confidence level.
    Computational and Mathematical Organization Theory, 23(2), 301-312.

    Parameters
    ----------
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
        permutation, swapfail = rewire_graph(graph, error)
        if swapfail:
            return
        adj = diffusion(graph=permutation, limit=2, iterations=3, norm=False, verbose=False)[0]
        perms.append(adj)
        # sys.stdout.write('Permutation ' + str(i) + '\n')
        # sys.stdout.flush()
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


def perm_clusters(graph, limit, max_clusters, min_clusters,
                  iterations, ratio, partialperms, relperms, error, verbose):
    """
    Calls the rewire_graph function and robustness function
    to compute robustness of cluster assignments.
    Scores close to 1 imply that the scores are robust to perturbation.


    Parameters
    ----------
    :param graph: NetworkX graph of a microbial association network. Cluster assignment should be a network property.
    :param limit: Percentage in error decrease until matrix is considered converged.
    :param max_clusters: Maximum number of clusters to evaluate in K-means clustering.
    :param min_clusters: Minimum number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :param ratio: Ratio of scores that need to be positive or negative for a stable edge
    :param partialperms: Number of permutations for partial diffusion.
    :param relperms: Number of permutations for reliability testing.
    :param error: Fraction of edges to rewire for reliability metric.
    :param verbose: Verbosity level of function
    :return:
    """
    assignments = list()
    rev_assignments = list()
    for i in range(relperms):
        permutation, swapfail = rewire_graph(graph, error)
        if swapfail:
            return
        permutation, mat = cluster_graph(graph=permutation, limit=limit, max_clusters=max_clusters,
                      min_clusters=min_clusters, iterations=iterations,
                      ratio=ratio, edgescale=0, permutations=partialperms,
                      verbose=False)
        cluster = nx.get_node_attributes(permutation, 'cluster')
        # cluster.values() has same order as permutation.nodes
        assignments.append(cluster)
        subassignments = dict()
        for k, v in cluster.items():
            subassignments.setdefault(v, set()).add(k)
        rev_assignments.append(subassignments)
        if verbose:
            sys.stdout.write('Permutation ' + str(i) + '\n')
            sys.stdout.flush()
    graphclusters = nx.get_node_attributes(graph, 'cluster')
    clusjaccards, nodejaccards, ci_width = robustness(graphclusters, assignments)
    lowerCI = dict()
    upperCI = dict()
    for node in nodejaccards:
        lowerCI[node] = str(nodejaccards[node][0])
        upperCI[node] = str(nodejaccards[node][1])
        ci_width[node] = str(ci_width[node])
    nx.set_node_attributes(graph, lowerCI, "lowerCI")
    nx.set_node_attributes(graph, upperCI, "upperCI")
    nx.set_node_attributes(graph, ci_width, "widthCI")
    if verbose:
        sys.stdout.write("Completed estimation of node Jaccard similarities across bootstraps. \n")
        sys.stdout.flush()


def robustness(graphclusters, permutations):
    """
    Compares vectors of cluster assignments to estimate cluster-wise robustness
    and node-wise robustness. These are returned as dictionaries.

    Inspired by reliablity scores as proposed by:
    Frantz, T. L., & Carley, K. M. (2017).
    Reporting a network’s most-central actor with a confidence level.
    Computational and Mathematical Organization Theory, 23(2), 301-312.

    Because calculating the accuracy of a cluster assignment is not trivial,
    the function does not compare cluster labels directly.
    Instead, this function calculates the Jaccard similarity between cluster assignments.

    Parameters
    ----------
    :param graphclusters: Dictionary of original cluster assignments
    :return: Two dictionaries of reliability scores (cluster-wise and node-wise).
    """
    rev_assignments = list()
    for assignment in permutations:
        subassignments = dict()
        for k, v in assignment.items():
            subassignments.setdefault(v, set()).add(k)
        rev_assignments.append(subassignments)
    revclusters = dict()
    for k, v in graphclusters.items():
        revclusters.setdefault(v, set()).add(k)
    # clusterwise jaccard
    clusjaccards = dict()
    for cluster in set(graphclusters.values()):
        true_composition = revclusters[cluster]
        jaccards = list()
        # keys don't have to match so both cluster assignments should be evaluated
        for rev_assignment in rev_assignments:
            scores = list()
            for key in rev_assignment:
                scores.append(jaccard_similarity_score(true_composition, rev_assignment[key]))
            bestmatch = np.max(scores)
            jaccards.append(bestmatch)
        clusjaccards[cluster] = np.round(norm.interval(0.95, np.mean(jaccards), np.std(jaccards)), 4)
    sys.stdout.write("Confidence intervals for Jaccard similarity of cluster assignments: \n")
    sys.stdout.write(str(clusjaccards) + "\n")
    sys.stdout.flush()
    nodejaccards = dict.fromkeys(graphclusters.keys())
    ci_width = dict.fromkeys(graphclusters.keys())
    for node in nodejaccards:
        true_composition = revclusters[graphclusters[node]]
        jaccards = list()
        for i in range(len(permutations)):
            clusid = permutations[i][node]
            rev_assignment = rev_assignments[i][clusid]
            jaccards.append(jaccard_similarity_score(true_composition, rev_assignment))
        nodejaccards[node] = np.round(norm.interval(0.95, np.mean(jaccards), np.std(jaccards)), 4)
        ci_width[node] = np.round(nodejaccards[node][1] - nodejaccards[node][0], 4)
    return clusjaccards, nodejaccards, ci_width


def jaccard_similarity_score(vector1, vector2):
    """
    The sklearn implementation of the Jaccard similarity requires vectors to be equal length.
    This implementation does not.
    :param vector1: List of strings or numbers.
    :param vector2: List of strings or numbers.
    :return:
    """
    jaccard = len(vector1.intersection(vector2)) / \
              (len(vector1) + len(vector2) - len(vector1.intersection(vector2)))
    return jaccard

