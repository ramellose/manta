#!/usr/bin/env python

"""
The clustering algorithm works in several steps. First, an empty adjacency matrix is instantiated.
Then, the following steps are carried out until sparsity has stabilized:

1. Raise the matrix to power 2
2. Normalize matrix by absolute maximum value
3. Add 1 / value for each value in matrix
4. Normalize matrix by absolute maximum value
5. Calculate error by subtracting previous matrix from current matrix
6. If error threshold is reached, the algorithm stops

The generated score matrices and flow matrices are then supplied to the centrality function.
The centrality of an edge is calculated by the value of an edge in the matrix.
Hence, this does not imply that an edge had a large effect on other edges;
rather, it implies that it was affected by many of the diffusion processes during the iterations.

These centrality scores are then tested for their robustness using the functions in bootstrap_centrality.
As the absolute value of the scores is irrelevant,
only the ID of the edge (negative or positive hub) and the p-value is returned.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import sys
from manca.perms import perm_graph, diffuse_graph
from scipy.stats import binom_test


def cluster_graph(graph, limit, max_clusters, min_clusters, iterations,
                  mode='sparsity', cluster='DBSCAN'):
    """
    Takes a networkx graph
    and carries out network clustering until
    sparsity results converge. Directionality is ignored;
    if weight is available, this is considered during the diffusion process.


    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub species.

    Parameters
    ----------
    :param graph: Weighted, undirected networkx graph.
    :param limit: Number of iterations to run until alg considers sparsity value converged.
    :param max_clusters: Maximum number of clusters to evaluate in K-means clustering.
    :param min_clusters: Minimum number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :param mode: Criterion for evaluating clusters.
    :param cluster: Algorithm for clustering of score matrix.
    :return: NetworkX graph, score matrix and diffusion matrix.
    """
    adj = np.zeros((len(graph.nodes), len(graph.nodes)))  # this considers diffusion, I could also just use nx adj
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    rev_index = {v: k for k, v in adj_index.items()}
    # next part is to define clusters of the adj matrix
    # cluster number is defined through gap statistic
    # max cluster number to test is by default 5
    # define topscore and bestcluster for no cluster
    diffusion = diffuse_graph(graph, limit, iterations)
    scoremat = diffusion[0]
    bestcluster = None
    randomclust = np.random.randint(2, size=len(adj))
    scores = list()
    scores.append(sparsity_score(graph, randomclust, rev_index))
    sys.stdout.write('Sparsity level for 2 clusters, randomly assigned labels: ' + str(scores[0]) + '\n')
    sys.stdout.flush()
    # the randomclust is a random separation into two clusters
    # if K-means can't beat this, the user is given a warning
    # select optimal cluster by sparsity score
    if cluster == 'KMeans':
        for i in range(min_clusters, max_clusters+1):
            clusters = KMeans(i).fit_predict(scoremat)
            score = sparsity_score(graph, clusters, rev_index)
            sys.stdout.write('Sparsity level of k=' + str(i) + ' clusters: ' + str(score) + '\n')
            sys.stdout.flush()
            scores.append(score)
        topscore = int(np.argmin(scores)) + min_clusters - 1
        if topscore >= min_clusters:
            sys.stdout.write('Lowest score for k=' + str(topscore) + ' clusters: ' + str(np.min(scores)) + '\n')
            sys.stdout.flush()
        else:
            sys.stdout.write('Warning: random clustering performed best. \n Setting cluster amount to minimum value. \n')
            sys.stdout.flush()
            topscore = min_clusters
        bestcluster = KMeans(topscore).fit_predict(scoremat)
    elif cluster == 'DBSCAN':
        bestcluster = DBSCAN(min_samples=len(graph.nodes) / max_clusters).fit_predict(scoremat)
        score = sparsity_score(graph, bestcluster, rev_index)
        sys.stdout.write('Sparsity level of ' + str(len(set(bestcluster))) + ' clusters: ' + str(score) + '\n')
        sys.stdout.flush()
    clusdict = dict()
    for i in range(len(graph.nodes)):
        clusdict[list(graph.nodes)[i]] = int(bestcluster[i])
    nx.set_node_attributes(graph, values=clusdict, name='cluster')
    return graph, diffusion


def central_edge(matrix, graph, percentage=10, permutations=10000, iterations=20, limit=0.00001):
    """
    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub associations.

    If the permutation number is set to a value above 0, the centrality values are tested against permuted graphs.

    The fraction of positive edges and negative edges is based on the ratio between
    positive and negative weights in the network.

    Hence, a network with 90 positive weights and 10 negative weights will have 90% positive hubs returned.

    Parameters
    ----------
    :param matrix: Outcome of flow process from cluster_graph.
    :param graph: NetworkX graph of a microbial association network.
    :param iterations: The number of iterations carried out by the clustering algorithm.
    :param percentage: Determines percentile of hub species to return.
    :param permutations: Number of permutations to carry out. If 0, no permutation test is done.
    :return: Networkx graph with hub ID / p-value as node property.
    """
    weights = nx.get_edge_attributes(graph, 'weight')
    # calculates the ratio of positive / negative weights
    # note that ratios need to be adapted, because the matrix is symmetric
    posnodes = sum(weights[x] > 0 for x in weights)
    ratio = posnodes / len(weights)
    negthresh = np.percentile(matrix, percentage*(1-ratio)/2)
    posthresh = np.percentile(matrix, 100-percentage*ratio/2)
    neghubs = np.argwhere(matrix < negthresh)
    poshubs = np.argwhere(matrix > posthresh)
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    if permutations > 0:
        pvals = perm_graph(graph, matrix, limit=limit, iterations=iterations,
                           permutations=permutations, posthresh=posthresh, negthresh=negthresh)
    # need to make sure graph is undirected
    graph = nx.to_undirected(graph)
    # initialize empty dictionary to store edge ID
    edge_vals = dict()
    edge_pvals = dict()
    # need to convert matrix index to node ID
    for edge in neghubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'negative hub'
        if permutations > 0:
            edge_pvals[(node1, node2)] = pvals[adj_index[node1], adj_index[node2]]
    for edge in poshubs:
        node1 = list(graph.nodes)[edge[0]]
        node2 = list(graph.nodes)[edge[1]]
        edge_vals[(node1, node2)] = 'positive hub'
        if permutations > 0:
            edge_pvals[(node1, node2)] = pvals[adj_index[node1], adj_index[node2]]
    nx.set_edge_attributes(graph, values=edge_vals, name='hub')
    if permutations > 0:
        nx.set_edge_attributes(graph, values=edge_pvals, name='hub p-value')


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
            pvals[node] = pval
    nx.set_node_attributes(graph, values=sighubs, name='hub')
    nx.set_node_attributes(graph, values=pvals, name='hub p-value')


def sparsity_score(graph, clusters, rev_index):
    """
    Given a graph, and a list of cluster identities,
    this function calculates how many edges need to be cut
    to assign these cluster identities.
    Cuts through positively-weighted edges are penalized,
    whereas cuts through negatively-weighted edges are rewarded.
    The lowest sparsity score represents the best clustering.
    :param graph: NetworkX weighted, undirected graph
    :param clusters: List of cluster identities
    :param rev_index: Index matching node ID to matrix index
    :return: Sparsity score
    """
    sparsity = 0
    edges = list()
    for cluster_id in set(clusters):
        # get the set of edges that is NOT in either cluster
        node_ids = list(np.where(clusters == cluster_id)[0])
        node_ids = [rev_index.get(item, item) for item in node_ids]
        cluster = graph.subgraph(node_ids)
        edges.extend(list(cluster.edges))
        # penalize for having negative edges inside cluster
        weights = nx.get_edge_attributes(cluster, 'weight')
        for x in weights:
            if weights[x] < 0:
                sparsity += 1
    all_edges = list(graph.edges)
    cuts = list()
    for edge in all_edges:
        if edge not in edges and (edge[1], edge[0]) not in edges:
            # problem with cluster edges having swapped orders
            cuts.append(edge)
    for edge in cuts:
        # only add 1 to sparsity if it is a positive edge
        # otherwise subtract 1
        cut = graph[edge[0]][edge[1]]['weight']
        if cut > 0:
            sparsity += 1
        else:
            sparsity -= 1
    return sparsity