import networkx as nx
from manca.manca import clus_central
from manca.cluster import central_graph, cluster_graph, diffuse_graph
from manca.perms import null_graph, perm_graph, diffuse_graph
from manca.layout import generate_layout, generate_tax_weights
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import sys


g = nx.Graph()
nodes = ["OTU_1", "OTU_2", "OTU_3", "OTU_4", "OTU_5",
         "OTU_6", "OTU_7", "OTU_8", "OTU_9", "OTU_10"]
g.add_nodes_from(nodes)
g.add_edges_from([("OTU_1", "OTU_2"), ("OTU_1", "OTU_3"),
                  ("OTU_2", "OTU_5"), ("OTU_3", "OTU_4"),
                  ("OTU_4", "OTU_8"), ("OTU_5", "OTU_7"),
                  ("OTU_10", "OTU_1"), ("OTU_7", "OTU_8"),
                  ("OTU_10", "OTU_9"), ("OTU_9", "OTU_6"),
                  ("OTU_3", "OTU_8")])

weights = dict()
weights[("OTU_1", "OTU_2")] = float(1.0)
weights[("OTU_1", "OTU_3")] = float(1.0)
weights[("OTU_2", "OTU_5")] = float(1.0)
weights[("OTU_3", "OTU_4")] = float(-1.0)
weights[("OTU_4", "OTU_8")] = float(-1.0)
weights[("OTU_5", "OTU_7")] = float(1.0)
weights[("OTU_10", "OTU_1")] = float(-1.0)
weights[("OTU_7", "OTU_8")] = float(1.0)
weights[("OTU_10", "OTU_9")] = float(1.0)
weights[("OTU_9", "OTU_6")] = float(1.0)
weights[("OTU_3", "OTU_8")] = float(1.0)


nx.set_edge_attributes(g, values=weights, name='weight')

g = g.to_undirected()

limit = 100
diff_range = 3
max_clusters = 5
iterations = 1000
central = True
percentage = 10
bootstraps = 100

def cluster_graph_animation(graph, limit, diff_range, max_clusters, iterations):
    """
    Takes a networkx graph
    and carries out network clustering until
    sparsity results converge. Directionality is ignored;
    if weight is available, this is considered during the diffusion process.
    Setting diff_range to 1 means that the algorithm
    will basically cluster the adjacency matrix.

    This alternative version returns the matrix at each time point.

    The min / max values that are the result of the diffusion process
    are used as a centrality measure and define positive as well as negative hub species.

    Parameters
    ----------
    :param graph: Weighted, undirected networkx graph.
    :param limit: Number of iterations to run until alg considers sparsity value converged.
    :param diff_range: Diffusion range of network perturbation.
    :param max_clusters: Number of clusters to evaluate in K-means clustering.
    :param iterations: If algorithm does not converge, it stops here.
    :return: NetworkX graph, number of iterations and diffusion matrix.
    """
    images = list()
    delay = 0  # after delay reaches the limit value, algorithm is considered converged
    adj = np.zeros((len(graph.nodes), len(graph.nodes)))  # this considers diffusion, I could also just use nx adj
    adj_index = dict()
    for i in range(len(graph.nodes)):
        adj_index[list(graph.nodes)[i]] = i
    rev_index = {v: k for k, v in adj_index.items()}
    prev_sparsity = 0
    iters = 0
    while delay < limit and iters < iterations:
        # next part is to define clusters of the adj matrix
        # cluster number is defined through gap statistic
        # max cluster number to test is by default 5
        # define topscore and bestcluster for no cluster
        adj = diffuse_graph(graph, adj, diff_range, adj_index)
        bestcluster = None
        randomclust = np.random.randint(2, size=len(adj))
        try:
            sh_score = [silhouette_score(adj, randomclust)]
        except ValueError:
            sh_score = [0]  # the randomclust can result in all 1s or 0s which crashes
        # scaler = MinMaxScaler()
        # select optimal cluster by silhouette score
        # cluster may be arbitrarily bad before convergence
        # may scale adj mat values from 0 to 1 but scaling is probably not necessary
        for i in range(1, max_clusters+1):
            # scaler.fit(adj)
            # proc_adj = scaler.transform(adj)
            clusters = KMeans(i).fit_predict(adj)
            try:
                silhouette_avg = silhouette_score(adj, clusters)
            except ValueError:
                # if only 1 cluster label is defined this can crash
                silhouette_avg = 0
            sh_score.append(silhouette_avg)
        topscore = int(np.argmax(sh_score))
        if topscore != 0:
            bestcluster = KMeans(topscore).fit_predict(adj)
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
            # print("Complete cut-set sparsity: " + sparsity)
            if prev_sparsity > sparsity:
                delay = 0
            if prev_sparsity == sparsity:
                delay += 1
            prev_sparsity = sparsity
            iters += 1
            sys.stdout.write('Current sparsity level: ' + str(prev_sparsity) + '\n')
            sys.stdout.flush()
            sys.stdout.write('Number of iterations: ' + str(iters) + '\n')
            sys.stdout.flush()
            images.append(deepcopy(adj))
    if iters == iterations:
        sys.stdout.write('Warning: algorithm did not converge.' + '\n')
        sys.stdout.flush()
    clusdict = dict()
    for i in range(len(graph.nodes)):
        clusdict[list(graph.nodes)[i]] = bestcluster[i]
    nx.set_node_attributes(graph, values=clusdict, name='cluster')
    return graph, iters, adj, images

results = cluster_graph_animation(g, limit, max_clusters, iterations)

images = results[3]

import matplotlib.pyplot as plt


names = list()
for i in range(50):
    plt.imshow(images[i], cmap='hot', interpolation='nearest')
    name = str(i) + ".png"
    plt.savefig(name)
    names.append(name)

from moviepy.editor import *

clips = [ImageClip(m).set_duration(0.5) for m in names]

concat_clip = concatenate_videoclips(clips, method="compose")
concat_clip.write_videofile("test.mp4", fps=4)

from subprocess import call
for name in names:
    call("rm " + name)