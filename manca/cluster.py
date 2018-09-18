#!/usr/bin/env python

"""
The clustering algorithm works in several steps. First, an empty adjacency matrix is instantiated.
Then, the following steps are carried out until sparsity has stabilized:

1. Choose a random node i and select k-neighbours for k in diffusion range
2. For each node b in k-neighbours, multiply 1 by the edge weight of edge (i, b) and the value generated from (i, k-1)
3. Perform K-Means clustering on the resulting diffusion score matrix
4. Calculate sparsity of clusters;
            positively weighted edges between clusters add 1 to the sparsity value
            negatively weighted edges between clusters subtract 1 of the sparsity value
5. Sparsity lower than previous sparsity? --> continue iterating
   Sparsity equal to previous sparsity? --> iterate until convergence limit is reached

The number of iterations and the generated matrix is then supplied to the centrality function.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
from manca.perms import perm_graph, diffuse_graph


def cluster_graph(graph, limit, max_clusters, iterations):
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
    :param max_clusters: Number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :return: NetworkX graph, number of iterations and diffusion matrix.
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
    scoremat = diffuse_graph(graph, limit, iterations)
    bestcluster = None
    randomclust = np.random.randint(2, size=len(adj))
    try:
        sh_score = [silhouette_score(scoremat, randomclust)]
    except ValueError:
        sh_score = [0]  # the randomclust can result in all 1s or 0s which crashes
    # select optimal cluster by silhouette score
    for i in range(1, max_clusters+1):
        clusters = KMeans(i).fit_predict(scoremat)
        try:
            silhouette_avg = silhouette_score(scoremat, clusters)
        except ValueError:
            # if only 1 cluster label is defined this can crash
            silhouette_avg = 0
        sh_score.append(silhouette_avg)
    topscore = int(np.argmax(sh_score))
    if topscore != 0:
        bestcluster = KMeans(topscore).fit_predict(scoremat)
        # with bestcluster defined,
        # sparsity of cut can be calculated
        sparsity = 0
        for cluster_id in set(bestcluster):
            node_ids = list(np.where(bestcluster == cluster_id)[0])
            node_ids = [rev_index.get(item, item) for item in node_ids]
            cluster = graph.subgraph(node_ids)
            # per cluster node:
            # edges that are not inside cluster are part of cut-set
            # total cut-set should be as small as possible
            for node in cluster.nodes:
                nbs = graph.neighbors(node)
                for nb in nbs:
                    if nb not in node_ids:
                        # only add 1 to sparsity if it is a positive edge
                        # otherwise subtract 1
                        cut = graph[node][nb]['weight']
                        if cut > 0:
                            sparsity += 1
                        else:
                            sparsity -= 1
    sys.stdout.write('Sparsity level of clusters: ' + str(sparsity) + '\n')
    sys.stdout.flush()
    clusdict = dict()
    for i in range(len(graph.nodes)):
        clusdict[list(graph.nodes)[i]] = bestcluster[i]
    nx.set_node_attributes(graph, values=clusdict, name='cluster')
    return graph, scoremat


def central_graph(matrix, graph, percentage=10, permutations=10000, iterations=1000, limit=0.00001):
    """
    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub associations.

    If the permutation number is set to a value above 0, the centrality values are tested against permuted graphs.

    The fraction of positive edges and negative edges is based on the ratio between
    positive and negative weights in the network.

    Hence, a network with 90 positive weights and 10 negative weights will have 90% positive hubs returned.

    Parameters
    ----------
    :param matrix: Outcome of diffusion process from cluster_graph.
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
        pvals = perm_graph(graph, matrix, limit, iterations, permutations, posthresh, negthresh)
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

