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

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import sys
from manta.perms import diffusion
from itertools import combinations, chain


def cluster_graph(graph, limit, max_clusters, min_clusters, iterations, edgescale,
                  cluster='KMeans'):
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
    :param limit: Percentage in error decrease until matrix is considered converged.
    :param max_clusters: Maximum number of clusters to evaluate in K-means clustering.
    :param min_clusters: Minimum number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :param edgescale: Scaling factor used to separate out fuzzy cluster.
    :param cluster: Algorithm for clustering of score matrix.
    :return: NetworkX graph, score matrix and diffusion matrix.
    """
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    rev_index = {v: k for k, v in adj_index.items()}
    # next part is to define clusters of the adj matrix
    # cluster number is defined through gap statistic
    # max cluster number to test is by default 5
    # define topscore and bestcluster for no cluster
    scoremat, memory, diffs = diffusion(graph=graph, limit=limit, iterations=iterations)
    bestcluster = None
    # the randomclust is a random separation into two clusters
    # if K-means can't beat this, the user is given a warning
    # select optimal cluster by sparsity score
    if not memory:
        bestcluster = cluster_hard(graph=graph, rev_index=rev_index, scoremat=scoremat,
                                   max_clusters=max_clusters, min_clusters=min_clusters, cluster=cluster)
    if memory:
        bestcluster = cluster_fuzzy(graph=graph, rev_index=rev_index, adj_index=adj_index,
                                    diffs=diffs, scoremat=scoremat, edgescale=edgescale, max_clusters=max_clusters,
                                    min_clusters=min_clusters, cluster=cluster)
    clusdict = dict()
    for i in range(len(graph.nodes)):
        clusdict[list(graph.nodes)[i]] = float(bestcluster[i])
    nx.set_node_attributes(graph, values=clusdict, name='cluster')
    return graph, scoremat


def sparsity_score(graph, clusters, rev_index):
    """
    Given a graph, and a list of cluster identities,
    this function calculates how many edges need to be cut
    to assign these cluster identities.
    Cuts through positively-weighted edges are penalized,
    whereas cuts through negatively-weighted edges are rewarded.
    The lowest sparsity score represents the best clustering.
    Because this sparsity score rewards cluster assignments
    that separate out negative hubs.
    This penalty ensures that nodes with some intra-cluster
    negative edges are still assigned to clusters where
    they predominantly co-occur with other cluster members.
    :param graph: NetworkX weighted, undirected graph
    :param clusters: List of cluster identities
    :param rev_index: Index matching node ID to matrix index
    :return: Sparsity score
    """
    # set up scale for positive + negative edges
    # the sparsity score scales from -1 to 1
    # 1 is the worst possible assignment,
    # all negative edges inside clusters and positive outside clusters
    # -1 is the best,
    # with clusters containing only positive edges
    cut_score = 1/len(graph.edges)
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
                sparsity += cut_score
            else:
                sparsity -= cut_score
    all_edges = list(graph.edges)
    cuts = list()
    for edge in all_edges:
        if edge not in edges and (edge[1], edge[0]) not in edges:
            # problem with cluster edges having swapped orders
            cuts.append(edge)
    for edge in cuts:
        cut = graph[edge[0]][edge[1]]['weight']
        if cut > 0:
            sparsity += cut_score
        else:
            sparsity -= cut_score
    return sparsity


def cluster_fuzzy(graph, diffs, scoremat, adj_index, rev_index, edgescale,
                  max_clusters, min_clusters, cluster='KMeans'):
    """
    If a memory effect is demonstrated to exist during
    matrix diffusion, the fuzzy clustering algorithm assigns
    cluster identity based on multiple diffusion steps.
    :param graph: NetworkX weighted, undirected graph
    :param adj_index: Index matching matrix index to node ID
    :param rev_index: Index matching node ID to matrix index
    :param diffs: List of diffusion matrices extracted from flip-flops
    :param scoremat: Diffusion matrix
    :param edgescale: Scaling factor used to separate out fuzzy cluster.
    :param max_clusters: Maximum cluster number
    :param min_clusters: Minimum cluster number
    :param cluster: Clustering method (only KMeans supported for now)
    :return: Vector with cluster assignments
    """
    # clustering on the 1st iteration already yields reasonable results
    # however, we can't separate nodes that are 'intermediates' between clusters
    # solution: find nodes with the lowest cumulative edge amplitudes
    # the nodes that oscillate the most, have the largest self-loop amplitude
    # these are on the cluster periphery
    # nodes that are in-between clusters do not have large self-loop amplitudes
    bestcluster = cluster_hard(graph=graph, rev_index=rev_index, scoremat=scoremat,
                               max_clusters=max_clusters, min_clusters=min_clusters, cluster=cluster)
    # cluster assignment 0 is reserved for fuzzy clusters
    sys.stdout.write('Determining fuzzy nodes. \n')
    sys.stdout.flush()
    # diffs is a 3-dimensional array; need to extract 2D dataframe with timeseries for each edge
    # each timeseries is 5 flip-flops long
    diffs = np.array(diffs)
    # only upper triangle of matrix is indexed this way
    oscillators = list()
    oscillators_series = list()
    for index in range(len(bestcluster)):
        # node amplitude is NOT correlated to position in network
        seq = diffs[:, index, index]
        ampli = np.max(seq) - np.min(seq)
        if ampli > 0.5:
            # if the amplitude is this large,
            # the node may be an oscillator
            # in that case, mean amplitude may be low
            oscillators.append(index)
            oscillators_series.append(seq)
    oscillators = [rev_index[x] for x in oscillators]
    sys.stdout.write('Found the following strong oscillators: ' + str(oscillators) + '\n')
    sys.stdout.flush()
    amplis = dict()
    clusdict = dict.fromkeys(oscillators)
    for x in clusdict:
        clusdict[x] = bestcluster[adj_index[x]]
    # we find anti-correlated oscillator nodes
    # there should be at least one node represented for each cluster
    for pair in combinations(range(len(oscillators)), 2):
        total = oscillators_series[pair[0]] - oscillators_series[pair[1]]
        # need to be careful with this number,
        # the core oscillators should converge to 1 and -1
        # but may stick a little below that value
        amplis[(oscillators[pair[0]], oscillators[pair[1]])] = (np.max(total) - np.min(total))
    # need to find the largest anti-correlation per cluster
    clus_corrs = dict.fromkeys(set(bestcluster), 0)
    clus_nodes = dict.fromkeys(set(bestcluster))
    for corr in amplis:
        cluster1 = clusdict[corr[0]]
        cluster2 = clusdict[corr[1]]
        if amplis[corr] > clus_corrs[cluster1]:
            clus_nodes[cluster1] = corr
            clus_corrs[cluster1] = amplis[corr]
        if amplis[corr] > clus_corrs[cluster2]:
            clus_nodes[cluster2] = corr
            clus_corrs[cluster2] = amplis[corr]
    clus_nodes = {k: v for k, v in clus_nodes.items() if v is not None}
    # it is possible for clusters to not have a strong oscillator
    core_oscillators = set(list(chain.from_iterable(list(clus_nodes.values()))))
    anti_corrs = dict.fromkeys(core_oscillators, 0)
    anti_sizes = dict.fromkeys(core_oscillators, 0)
    for nodes in combinations(core_oscillators, 2):
        try:
            size = amplis[nodes]
        except KeyError:
            size = amplis[(nodes[1], nodes[0])]
        if size > anti_sizes[nodes[0]]:
            anti_corrs[nodes[0]] = nodes[1]
            anti_sizes[nodes[0]] = size
        elif size > anti_sizes[nodes[1]]:
            anti_corrs[nodes[1]] = nodes[0]
            anti_sizes[nodes[1]] = size
    clusdict = {v: k for k, v in clusdict.items()}
    for core in core_oscillators:
        anti_corrs[clusdict[core]] = anti_corrs.pop(core)
        anti_corrs[clusdict[core]] = clusdict[anti_corrs[clusdict[core]]]
    # oscillator is defined as strongest anti-correlation
    # get all shortest paths to/from oscillators
    corrdict = dict.fromkeys(core_oscillators)
    weights = nx.get_edge_attributes(graph, 'weight')
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}
    rev_weights = dict()
    for key in weights:
        newkey = (key[1], key[0])
        rev_weights[newkey] = weights[key]
    weights = {**weights, **rev_weights}
    # first scale edge weights
    for node in core_oscillators:
        targets = list(graph.nodes)
        corrdict[node] = dict()
        for target in targets:
            shortest_paths = list(nx.all_shortest_paths(graph, source=node, target=target))
            total_weight = 0
            for path in shortest_paths:
                edge_weight = 1
                for i in range(len(path) - 1):
                    edge_weight *= weights[(path[i], path[i + 1])]
                total_weight += edge_weight
            total_weight = total_weight / len(shortest_paths)
            corrdict[node][target] = total_weight
    varweights = list()  # stores nodes that have low weights of mean shortest paths
    clus_matches = list()  # stores nodes that have matching signs for oscillators
    clus_assign = list()  # stores nodes that have negative shortest paths to cluster oscillator
    # first need to scale weight variables for this
    clusdict = dict.fromkeys(core_oscillators)
    for x in clusdict:
        clusdict[x] = bestcluster[adj_index[x]]
    clusdict = {v: k for k, v in clusdict.items()}
    for target in graph.nodes:
        assignment = bestcluster[adj_index[target]]
        try:
            opposite = anti_corrs[assignment]
            # not all clusters have an oscillator,
            # therefore, they don't have an anti-correlating oscillator either
            if np.sign(corrdict[clusdict[assignment]][target]) == np.sign(corrdict[clusdict[opposite]][target]):
                clus_matches.append(target)
        except KeyError:
            pass
            # if the signs of the shortest paths to the oscillators are the same,
            # this implies the node is in between cluster
        # if nodes are assigned to the same cluster as the oscillators
        # cumulative edge weights of shortest paths should be + 1
        weight = corrdict[clusdict[assignment]][target]
        if np.sign(weight) == -1:
            clus_assign.append(target)
        if -edgescale < weight < edgescale:  # the 0.5 and -0.5 values are arbitrary
            varweights.append(target)
    sys.stdout.write('Sign of cumulative edge weights does not match cluster assignment for: \n' +
                     str(clus_assign) + '\n' +
                     'Low mean edge weights of shortest paths to oscillator for: \n' +
                     str(varweights) + '\n' +
                     'Cumulative edge weights of shortest paths are the same sign for opposing oscillators: \n' +
                     str(clus_matches) + '\n')
    sys.stdout.flush()
    remove_cluster = [adj_index[x] for x in clus_assign + varweights + clus_matches]
    bestcluster[remove_cluster] = 0
    return bestcluster


def cluster_hard(graph, rev_index, scoremat, max_clusters, min_clusters, cluster='KMeans'):
    """
    If no memory effects are demonstrated, clusters can be identified
    without fuzzy clustering.
    :param graph: NetworkX weighted, undirected graph
    :param rev_index: Index matching node ID to matrix index
    :param scoremat: Converged diffusion matrix
    :param max_clusters: Maximum cluster number
    :param min_clusters: Minimum cluster number
    :param cluster: Clustering method (only KMeans supported for now)
    :return: Vector with cluster assignments
    """
    randomclust = np.random.randint(2, size=len(scoremat))
    scores = list()
    scores.append(sparsity_score(graph, randomclust, rev_index))
    sys.stdout.write('Sparsity level for 2 clusters, randomly assigned labels: ' + str(scores[0]) + '\n')
    sys.stdout.flush()
    bestcluster = None
    if cluster == 'KMeans':
        for i in range(min_clusters, max_clusters + 1):
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
            sys.stdout.write(
                'Warning: random clustering performed best. \n Setting cluster amount to minimum value. \n')
            sys.stdout.flush()
        bestcluster = KMeans(topscore).fit_predict(scoremat)
    else:
        sys.stdout.write(
            'Warning: only K-means clustering is supported at the moment. \n '
            'Setting cluster amount to minimum value. \n')
        sys.stdout.flush()
    bestcluster = bestcluster + 1
    # cluster assignment 0 is reserved for fuzzy clusters
    return bestcluster
