#!/usr/bin/env python

"""
The clustering algorithm works in several steps to generate cluster assignments.

1. Generate a scoring matrix using a network flow strategy
2. Cluster on the scoring matrix.
3. In case the network displays memory effects, define weak nodes.

The scoring matrix is first clustered with the AgglomerativeClustering algorithm.
Because the network flow strategy can result in central values being separated from the clusters,
agglomerative clustering is repeated on score matrices with removed high-scoring nodes
until larger clusters are identified.
After this step, clustering assignments are  set.
However, if the network caused the scoring matrix to enter a flip-flop state,
nodes can still be defined as 'weak'.
In this case, manta uses a shortest path strategy to assess whether nodes belong to a cluster
or are in conflict with the cluster oscillator.

"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import networkx as nx
import numpy as np
import random
# from sklearn.mixture import GaussianMixture  #  This works quite well, slightly better Sn
from sklearn.cluster import AgglomerativeClustering
import sys
from manta.flow import partial_diffusion, diffusion, harary_components
from itertools import combinations, chain
from copy import deepcopy
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


def cluster_graph(graph, limit, max_clusters, min_clusters, min_cluster_size,
                  iterations, subset, ratio, edgescale, permutations, seed=11111, verbose=True):
    """
    Takes a networkx graph and carries out network clustering.
    The returned graph contains cluster assignments and weak assignments.
    If weight is available, this is considered during the diffusion process.

    Parameters
    ----------
    :param graph: Weighted, undirected networkx graph.
    :param limit: Percentage in error decrease until matrix is considered converged.
    :param max_clusters: Maximum number of clusters to evaluate in K-means clustering.
    :param min_clusters: Minimum number of clusters to evaluate in K-means clustering.
    :param min_cluster_size: Minimum cluster size as fraction of network size
    :param iterations: If algorithm does not converge, it stops here.
    :param subset: Fraction of edges used in subsetting procedure
    :param ratio: Ratio of scores that need to be positive or negative for a stable edge
    :param edgescale: Mean edge weight for node removal
    :param permutations: Number of permutations for partial iterations
    :param seed: Integer of seed, 11111 means no seed is used
    :param verbose: Verbosity level of function
    :return: NetworkX graph, score matrix and diffusion matrix.
    """
    
    # suggested additions by Theresa
    # the seed 123 should be replaced with the input seed
    if seed == 11111:
        np.random.seed(seed)
        random.seed(seed)

    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    rev_index = {v: k for k, v in adj_index.items()}
    # next part is to define scoring matrix
    balanced = [False]
    scoremat, memory, diffs = diffusion(graph=graph, limit=limit, iterations=iterations, verbose=verbose)
    if not nx.is_directed(graph):
        balanced = harary_components(graph, verbose=verbose).values()
        # partial diffusion results in unclosed graphs for directed graphs,
        # and can therefore not be used here.
        if balanced:
            logger.info("This is a balanced network, "
                        "so you may be able to get good results with the Kernighan-Lin algorithm.")
        if verbose:
            logger.info("Carrying out diffusion on partial graphs. ")
        # ratio from 0.7 to 0.9 appears to give good results on 3 clusters
        scoremat, partials = partial_diffusion(graph=graph, iterations=iterations, limit=limit, subset=subset,
                                               ratio=ratio, permutations=permutations, seed=seed, verbose=verbose)
    bestcluster = None
    # the randomclust is a random separation into two clusters
    # if clustering can't beat this, the user is given a warning
    # select optimal cluster by sparsity score
    bestcluster = cluster_hard(graph=graph, adj_index=adj_index, rev_index=rev_index, scoremat=scoremat,
                               max_clusters=max_clusters,
                               min_clusters=min_clusters, min_cluster_size=min_cluster_size, seed=seed, verbose=verbose)
    flatcluster = _cluster_vector(bestcluster, adj_index)
    if not all(balanced):
        weak_nodes = cluster_weak(graph, diffs=diffs, cluster=flatcluster,
                                  edgescale=edgescale,
                                  adj_index=adj_index, rev_index=rev_index, verbose=verbose)
        weak_dict = dict()
        for node in graph.nodes:
            if adj_index[node] in weak_nodes:
                weak_dict[node] = 'weak'
            else:
                weak_dict[node] = 'strong'
            nx.set_node_attributes(graph, values=weak_dict, name='assignment')
    nx.set_node_attributes(graph, values=bestcluster, name='cluster')
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

    Parameters
    ----------
    :param graph: NetworkX weighted, undirected graph
    :param clusters: List of cluster identities
    :param rev_index: Index matching node ID to matrix index
    :return: Sparsity score
    """
    # set up scale for positive + negative edges
    # the sparsity score scales from -1 to 1
    # -1 is the worst possible assignment,
    # all negative edges inside clusters and positive outside clusters
    # 1 is the best,
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
                sparsity -= cut_score
            else:
                sparsity += cut_score
    all_edges = list(graph.edges)
    cuts = list()
    for edge in all_edges:
        if edge not in edges and (edge[1], edge[0]) not in edges:
            # problem with cluster edges having swapped orders
            cuts.append(edge)
    for edge in cuts:
        cut = graph[edge[0]][edge[1]]['weight']
        if cut > 0:
            sparsity -= cut_score
        else:
            sparsity += cut_score
    return sparsity


def cluster_hard(graph, adj_index, rev_index, scoremat,
                 max_clusters, min_clusters, min_cluster_size, seed=11111, verbose=False):
    """
    Agglomerative clustering is used to separate nodes based on the scoring matrix.
    Because the scoring matrix generally results in separation of 'central' nodes,
    these nodes are removed and clustering is repeated until larger clusters are identified.
    Afterwards, the nodes are assigned to clusters based on a shortest path strategy.

    Parameters
    ----------
    :param graph: NetworkX weighted, undirected graph
    :param adj_index: Index matching matrix index to node ID
    :param rev_index: Index matching node ID to matrix index
    :param scoremat: Converged diffusion matrix
    :param max_clusters: Maximum cluster number
    :param min_clusters: Minimum cluster number
    :param min_cluster_size: Minimum cluster size as fraction of network size
    :param seed: Random seed, if 11111 no random seed is used
    :param verbose: Verbosity level of function
    :return: Dictionary of nodes with cluster assignments
    """
    # get the mean of 100 assignments
    randomscores = list()
    for i in range(5):
        if seed != 11111:
            rng = np.random.default_rng(seed + i)
            randomclust = rng.integers(2, size=len(scoremat))
        else:
            randomclust = np.random.randint(2, size=len(scoremat))
        randomscores.append(sparsity_score(graph, randomclust, rev_index))
    scores = dict()
    scores['random'] = np.median(randomscores)
    if verbose:
        logger.info('Sparsity level for 2 clusters, randomly assigned labels: ' + str(scores['random']))
    bestclusters = dict()
    clusnum = min_clusters
    scoremat_index = rev_index.copy()
    outliers = dict()
    outliers[clusnum] = list()
    clustermat = scoremat.copy()
    # minimum cluster size is 10% of nodes
    minclus = len(graph) * min_cluster_size
    while clusnum < max_clusters + 1:
        clusters = AgglomerativeClustering(n_clusters=clusnum).fit_predict(clustermat)
        counts = np.bincount(clusters)
        # then add to cluster based on shortest paths
        if len(np.where(counts > (minclus / clusnum))[0]) < 2:
            logger.warning('All nodes are binned into a single cluster for k = ' + str(clusnum))
            scores[clusnum] = -1
            break
        elif len(np.where(counts < (minclus / clusnum))[0]) > 0:
            # if there are at least 5 cluster with fewer than 3 nodes,
            # remove nodes that separate into tiny cluster
            # get cluster ID and location for this cluster
            locs = np.where(counts < (minclus / clusnum))[0]
            # we only remove one cluster pos at a time
            # repeated clustering may assign node differently
            loc = locs[0]
            clusid = list(set(clusters))[loc]
            clusloc = np.where(clusters == clusid)[0][0]
            # we need to update the rev_index so that adjacency indices point to taxon IDs
            # outlier nodes are added to a list, to be dealt with later
            outliers[clusnum].append(scoremat_index[clusloc])
            clustermat, scoremat_index = _remove_node(clusloc, clustermat, scoremat_index)
            # now the smaller clusters are deleted, we can cluster on the updated scoring matrix
            if clustermat.shape[0] <= max_clusters:
                # indicates that there is no good clustering possible for this cluster number
                clusnum += 1
                outliers[clusnum] = list()
                # reset scoring matrix in case different cluster assignment does assign outliers
                clustermat = scoremat.copy()
                scoremat_index = rev_index.copy()
        else:
            scores[clusnum] = sparsity_score(graph, clusters, rev_index)
            bestclusters[clusnum] = clusters
            if verbose:
                logger.info('Sparsity level of k=' + str(clusnum) + ' clusters: '
                            + str(scores[clusnum]) + '.')
            clusnum += 1
            outliers[clusnum] = list()
            # reset scoring matrix in case different cluster assignment does assign outliers
            clustermat = scoremat.copy()
            scoremat_index = rev_index.copy()
    topscore = max(scores, key=scores.get)
    if topscore != 'random':
        if verbose:
            logger.info('Highest score for k=' + str(topscore) + ' clusters: ' + str(scores[topscore]))
    else:
        logger.warning('Warning: random clustering performed best.'
                       ' \n Setting cluster amount to minimum value.')
        topscore = min_clusters
        # it is possible that all evaluated cluster assignments did not work out
        # in that case, the assignment below is without the binning strategy
        if min_clusters not in bestclusters:
            bestclusters[min_clusters] = AgglomerativeClustering(n_clusters=min_clusters).fit_predict(clustermat)
    # given a topscore, clustering is carried out on scoremat without outliers
    outlier_locs = [adj_index[x] for x in outliers[topscore]]
    scoremat_index = rev_index.copy()
    clustermat = scoremat.copy()
    clustermat, scoremat_index = _remove_node(outlier_locs, clustermat, scoremat_index)
    bestcluster = bestclusters[topscore]
    # we need to assign outlier nodes to clusters after clustering on main network
    corrdict = _path_weights(outliers[topscore], graph, verbose)
    scoremat_index = {v: k for k, v in scoremat_index.items()}
    # generate dictionary of cluster assignments
    # use this to reconstruct bestcluster vector
    cluster_index = adj_index.copy()
    for key in cluster_index:
        try:
            cluster_index[key] = float(bestcluster[scoremat_index[key]])
        except KeyError:
            # key error happens for outlier that has not been assigned cluster ID yet
            pass
    # replace node values in corrdict with cluster ids
    for node in corrdict:
        # initialize list to store path values per cluster
        clusdict = dict.fromkeys(set(bestcluster))
        for key in clusdict:
            clusdict[key] = list()
        # lookup cluster ID of node
        for value in corrdict[node]:
            try:
                clusid = cluster_index[value]
                clusdict[clusid].append(corrdict[node][value])
            except KeyError:
                pass
        closest_cluster = list(set(bestcluster))[np.argmax([np.mean(clusdict[x]) for x in clusdict])]
        cluster_index[node] = float(closest_cluster)
    return cluster_index


def cluster_weak(graph, diffs, cluster, edgescale, adj_index, rev_index, verbose):
    """
    Although clusters can be assigned with cluster_hard, cluster_weak tests
    whether cluster assignments are in conflict with oscillator nodes present in clusters.

    Oscillators can only be defined from flip-flopping states;
    hence, this function should not be called for scoring matrices that converge to -1 and 1.

    Parameters
    ----------
    :param graph: NetworkX weighted, undirected graph
    :param diffs: Diffusion matrices from flip-flop states generated by diffusion
    :param cluster: Cluster assignment
    :param edgescale: Mean edge weight for node removal
    :param adj_index: Node index
    :param rev_index: Reverse node index
    :param verbose: Verbosity level of function
    :return: Vector with cluster assignments
    """
    # clustering on the 1st iteration already yields reasonable results
    # however, we can't separate nodes that are 'intermediates' between clusters
    # solution: find nodes with the lowest cumulative edge amplitudes
    # the nodes that oscillate the most, have the largest self-loop amplitude
    # these are on the cluster periphery
    # nodes that are in-between clusters do not have large self-loop amplitudes
    #bestcluster = cluster_hard(graph=graph, rev_index=rev_index, scoremat=scoremat,
    #                           max_clusters=max_clusters, min_clusters=min_clusters, cluster=cluster)
    # cluster assignment 0 is reserved for weak clusters
    if verbose:
        logger.info('Determining weak nodes.')
    # diffs is a 3-dimensional array; need to extract 2D dataframe with timeseries for each edge
    # each timeseries is 5 flip-flops long
    diffs = np.array(diffs)
    # only upper triangle of matrix is indexed this way
    core, anti = _core_oscillators(difmats=diffs, assignment=cluster,
                                   adj_index=adj_index, rev_index=rev_index, verbose=verbose)
    # for each node with contrasting edge products,
    # we can check whether the sparsity score is improved by
    # removing the node or adding it to another cluster.
    remove_cluster = _oscillator_paths(graph=graph, core_oscillators=core, assignment=cluster,
                                       adj_index=adj_index, edgescale=edgescale, verbose=verbose)

    # we only remove nodes with conflicting shortest paths if
    # they have a large impact on sparsity score
    #posfreq = np.zeros((len(graph), len(graph)))
    #negfreq = np.zeros((len(graph), len(graph)))
    # count how many times specific values in matrix have
    # been assigned positive or negative values
    #for b in range(len(partials)):
    #    posfreq[partials[b] > 0] += 1
    #    negfreq[partials[b] < 0] += 1
    # count relative frequeny - values close to 1 or 0 are ok
    #posratio = posfreq[np.where(posfreq != 0)] / \
    #           (posfreq[np.where(posfreq != 0)] + negfreq[np.where(posfreq != 0)])
    #weak_edges = np.where(np.logical_and(posratio < 0.8, posratio > 0.2))
    #weak_ids = np.where(posfreq != 0)
    #indices = weak_ids[0][weak_edges[0]]
    #weak_freq = np.unique(indices, return_counts=True)
    # determine how to go from weak edges to weak nodes
    # [0] or [1] does not matter, weak_edges is symmetrical
    #weak_nodes = list()
    #for i in range(len(weak_freq[0])):
    #   node = weak_freq[0][i]
    #    freq = weak_freq[1][i]
    #    deg = nx.degree(graph, list(graph.nodes)[0])
    #    if freq / deg > edgescale:
    #        weak_nodes.append(i)
    return remove_cluster


def _core_oscillators(difmats, assignment, adj_index, rev_index, verbose):
    """
    Given a list of diffusion matrices calculated during a flip-flop state,
    this function identifies core oscillators as well as their anti-correlated partners.

    Parameters
    ----------
    :param difmats: Diffusion matrices during flip-flop state
    :param assignment: Cluster assignment
    :param adj_index: Dictionary for indexing
    :param rev_index: Dictionary for indexing
    :param verbose: Verbosity level of function
    :return: Tuple with list of oscillators and dictionary of anti-correlated oscillators
    """
    oscillators = list()
    oscillators_series = list()
    for index in range(len(assignment)):
        # node amplitude is NOT correlated to position in network
        seq = difmats[:, index, index]
        ampli = np.max(seq) - np.min(seq)
        if ampli > 0.5:
            # if the amplitude is this large,
            # the node may be an oscillator
            # in that case, mean amplitude may be low
            oscillators.append(index)
            oscillators_series.append(seq)
    if len(oscillators) == 0:
        logger.warning("No oscillating nodes found.\n"
                       "No weak and strong clustering assignments can be made.\n"
                       "Try reducing the subset parameter. ")
    oscillators = [rev_index[x] for x in oscillators]
    if verbose:
        logger.info('Found the following strong oscillators: ' + str(oscillators))
    amplis = dict()
    clusdict = dict.fromkeys(oscillators)
    for x in clusdict:
        clusdict[x] = assignment[adj_index[x]]
    # we find anti-correlated oscillator nodes
    # there should be at least one node represented for each cluster
    for pair in combinations(range(len(oscillators)), 2):
        total = oscillators_series[pair[0]] - oscillators_series[pair[1]]
        # need to be careful with this number,
        # the core oscillators should converge to 1 and -1
        # but may stick a little below that value
        amplis[(oscillators[pair[0]], oscillators[pair[1]])] = (np.max(total) - np.min(total))
    # need to find the largest anti-correlation per cluster
    clus_corrs = dict.fromkeys(set(assignment), 0)
    clus_nodes = dict.fromkeys(set(assignment))
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
    id_corrs = dict.fromkeys(core_oscillators, 0)
    anti_sizes = dict.fromkeys(core_oscillators, 0)
    for nodes in combinations(core_oscillators, 2):
        try:
            size = amplis[nodes]
        except KeyError:
            size = amplis[(nodes[1], nodes[0])]
        if size > anti_sizes[nodes[0]]:
            id_corrs[nodes[0]] = nodes[1]
            anti_sizes[nodes[0]] = size
        if size > anti_sizes[nodes[1]]:
            id_corrs[nodes[1]] = nodes[0]
            anti_sizes[nodes[1]] = size
    [clusdict.pop(x) for x in list(clusdict.keys()) if x not in core_oscillators]
    anti_corrs = dict()
    for core in core_oscillators:
        anti_corrs[clusdict[core]] = clusdict[id_corrs[core]]
    # oscillator is defined as strongest anti-correlation
    return core_oscillators, anti_corrs


def _oscillator_paths(graph, core_oscillators,
                      assignment, adj_index, edgescale, verbose):
    """
    Given optimal cluster assignments and a set of core oscillators,
    this function computes paths to / from oscillators and identifies nodes
    which do not match their cluster + oscillator assignment.
    A list of these nodes is then returned.

    Parameters
    ----------
    :param graph: NetworkX weighted, undirected graph
    :param core_oscillators: List of core oscillators
    :param anti_corrs: Dictionary  of anti-correlated oscillators
    :param assignment: Cluster assignment
    :param adj_index: Dictionary for indexing
    :param edgescale: Threshold for node removal
    :param verbose: Verbosity level of function
    :return: List of node indices that are in conflict with oscillators
    """
    # get all shortest paths to/from oscillators
    corrdict = _path_weights(core_oscillators, graph, verbose)
    varweights = list()  # stores nodes that have low weights of mean shortest paths
    clus_matches = list()  # stores nodes that have matching signs for oscillators
    clus_assign = list()  # stores nodes that have negative shortest paths to cluster oscillator
    # first need to scale weight variables for this
    clusdict = dict.fromkeys(core_oscillators)
    for x in clusdict:
        clusdict[x] = assignment[adj_index[x]]
    clusdict = {v: k for k, v in clusdict.items()}
    for target in graph.nodes:
        clus_id = assignment[adj_index[target]]
        try:
            weight = corrdict[clusdict[clus_id]][target]
            if np.sign(weight) == -1:
                # this criterion filters out nodes that have negative path weight to their oscillator
                clus_assign.append(target)
            if -edgescale < weight < edgescale:
                # this criterion filters out nodes that have low cumulative path weights
                varweights.append(target)
        except KeyError:
            pass
            # cannot check oscillator sign for clusters w/o oscillators
    if verbose:
        logger.info('Sign of mean edge products does not match cluster assignment for: \n' +
                    str(clus_assign))
        logger.info('Mean edge products are small for: \n' +
                    str(varweights))
    remove_cluster = set([adj_index[x] for x in clus_assign + varweights])
    return list(remove_cluster)


def _node_sparsity(graph, removals, assignment, rev_index):
    default_sparsity = sparsity_score(graph, assignment, rev_index)
    clusters = set(assignment)
    updated_removals = deepcopy(removals)
    for node in removals:
        clus_id = list()
        clus_id.append(assignment[node])
        other_ids = list(clusters.difference(clus_id))
        other_sparsities = list()
        for id in other_ids:
            updated_assignment = deepcopy(assignment)
            updated_assignment[node] = id
            other_sparsities.append(sparsity_score(graph, updated_assignment, rev_index))
        if np.max(other_sparsities) < (default_sparsity - 0.3):
            updated_removals.remove(node)
    return updated_removals


def _path_weights(source, graph, verbose):
    """
    Given a list of nodes, this function returns a dictionary of all shortest path weights
    from the sources to all other nodes in the network.

    Parameters
    ----------
    :param source: List of source nodes
    :param graph: NetworkX graph object
    :param verbose: Verbosity level of function
    :return: Dictionary of shortest path weights
    """
    corrdict = dict.fromkeys(source)
    weights = nx.get_edge_attributes(graph, 'weight')
    max_weight = max(weights.values())
    weights = {k: v / max_weight for k, v in weights.items()}
    rev_weights = dict()
    for key in weights:
        newkey = (key[1], key[0])
        rev_weights[newkey] = weights[key]
    weights = {**weights, **rev_weights}
    # first scale edge weights
    for node in source:
        targets = list(graph.nodes)
        corrdict[node] = dict()
        for target in targets:
            try:
                shortest_paths = list(nx.all_shortest_paths(graph, source=node, target=target))
                total_weight = 0
                for path in shortest_paths:
                    edge_weight = 1
                    for i in range(len(path) - 1):
                        edge_weight *= weights[(path[i], path[i + 1])]
                    total_weight += edge_weight
                total_weight = total_weight / len(shortest_paths)
            except nx.exception.NetworkXNoPath:
                if verbose:
                    logger.warning("Could not find shortest path for: " + target)
                total_weight = -1
            corrdict[node][target] = total_weight
    return corrdict


def _remove_node(loc, mat, mat_index):
    """
    Given an outlier node to remove,
    this function updates the matrix and matrix index.

    Parameters
    ----------
    :param loc: Node(s) to remove
    :param mat: Scoring matrix
    :param mat_index: Matrix index
    :return: Tuple of  matrix and matrix_index
    """
    mat = np.delete(mat, loc, axis=0)
    mat = np.delete(mat, loc, axis=1)
    if type(loc) == list:
        loc.sort()
    else:
        loc = [loc]
    for i in range(len(loc)):
        item = loc[i]
        mat_index.pop(item)
        # need mat_index remove last key if this is not the clusloc
        if item != (len(mat_index)):
            for remainder in range(item, len(mat_index)):
                mat_index[remainder] = mat_index[remainder + 1]
            mat_index.pop(len(mat_index) - 1)
            loc = [x - 1 for x in loc]
    # update scoring matrix with removed nodes
    return mat, mat_index


def _cluster_vector(assignment, adj_index):
    """
    Given a dictionary of cluster assignments,
    this helper function returns a vector with cluster IDs.

    Parameters
    ----------
    :param assignment: Dictionary with nodes as keys, cluster IDs as values
    :param adj_index: Dictionary with nodes as keys, matrix index as values
    :return: Numpy array with cluster assignments
    """
    result = np.zeros(shape=(1, len(assignment)))[0]
    for key in adj_index:
        result[adj_index[key]] = assignment[key]
    return result
