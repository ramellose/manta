#!/usr/bin/env python

"""
The generated score matrices and flow matrices from the diffusion function are supplied to the centrality function.
The centrality of an edge is calculated by the value of an edge in the matrix.
Hence, this does not imply that an edge had a large effect on other edges;
rather, it implies that it was affected by many of the diffusion processes during the iterations.

These centrality scores are then tested for their robustness using the functions in perm_graph.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import networkx as nx
import numpy as np
from manta.perms import diffusion, perm_graph
from scipy.stats import binom_test


def central_edge(graph, percentile, permutations, error):
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
    :return: Networkx graph with hub ID / p-value as node property.
    """
    scoremat = diffusion(graph, iterations=3, norm=False)[0]
    negthresh = np.percentile(scoremat, percentile)
    posthresh = np.percentile(scoremat, 100-percentile)
    neghubs = list(map(tuple, np.argwhere(scoremat <= negthresh)))
    poshubs = list(map(tuple, np.argwhere(scoremat >= posthresh)))
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    if permutations > 0:
        score = perm_graph(graph, scoremat, percentile=percentile, permutations=permutations,
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
        if permutations > 0:
            edge_scores[(node1, node2)] = score[(adj_index[node1], adj_index[node2])]
    for edge in poshubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'positive hub'
        if permutations > 0:
            edge_scores[(node1, node2)] = score[(adj_index[node1], adj_index[node2])]
    nx.set_edge_attributes(graph, values=edge_vals, name='hub')
    if permutations > 0:
        nx.set_edge_attributes(graph, values=edge_scores, name='reliability score')


def central_node(graph):
    """
    Given a graph with hub edges assigned (see central_edge),
    this function checks whether a node is significantly more connected
    to edges with high scores than expected by chance.
    The p-value is calculated with a binomial test.
    Edge sign is ignored; hubs can have both positive and negative
    edges.
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
