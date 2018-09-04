"""
This file contains a testing function + resources for testing the clustering algorithm.
"""

__author__ = 'Lisa Rottjers'
__maintainer__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import unittest
import networkx as nx
from manca.manca import cluster_graph, main, central_graph, null_graph, diffuse_graph, bootstrap_test, set_manca
from copy import deepcopy
import numpy as np

g = nx.Graph()
nodes = ["OTU_1", "OTU_2", "OTU_3", "OTU_4", "OTU_5",
         "OTU_6", "OTU_7", "OTU_8", "OTU_9", "OTU_10"]
g.add_nodes_from(nodes)
g.add_edges_from([("OTU_1", "OTU_2"), ("OTU_1", "OTU_3"),
                  ("OTU_2", "OTU_5"), ("OTU_3", "OTU_4"),
                  ("OTU_4", "OTU_8"), ("OTU_5", "OTU_7"),
                  ("OTU_10", "OTU_1"), ("OTU_7", "OTU_8"),
                  ("OTU_10", "OTU_9"), ("OTU_9", "OTU_6")])

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

nx.set_edge_attributes(g, values=weights, name='weight')

g = g.to_undirected()

limit = 100
diff_range = 3
max_clusters = 5
iterations = 1000
central = True
percentage = 10
bootstraps = 100


class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def test_default_manca(self):
        """
        Checks whether the main function carries out both clustering and centrality estimates.
        """
        clustered_graph = main(deepcopy(g))
        clusters = nx.get_node_attributes(clustered_graph, 'cluster')
        hubs = nx.get_edge_attributes(clustered_graph, 'hub')

    def test_center_manca(self):
        """
        Checks if the edge between 1 and 2 is identified as a positive hub.

        WARNING: at the moment the test indicates that centrality measures
        are not stable.
        """
        results = cluster_graph(deepcopy(g), limit, diff_range, max_clusters, iterations)
        graph = results[0]
        numbers = results[1]
        matrix = results[2]
        central_graph(matrix, graph, numbers, diff_range, percentage, bootstraps)
        hubs = nx.get_edge_attributes(graph, 'hub')
        self.assertEqual(hubs[list(hubs.keys())[0]], 'positive hub')

    def test_cluster_manca(self):
        """Checks whether the correct cluster IDs are assigned. """
        clustered_graph = cluster_graph(deepcopy(g), limit, diff_range, max_clusters, iterations)
        clusters = nx.get_node_attributes(clustered_graph[0], 'cluster')
        self.assertEqual(clusters['OTU_10'], clusters['OTU_6'])

    def test_diffuse_graph(self):
        """Checks if the diffusion process operates correctly. """
        adj = np.zeros((len(g.nodes), len(g.nodes)))  # this considers diffusion, I could also just use nx adj
        adj_index = dict()
        for i in range(len(g.nodes)):
            adj_index[list(g.nodes)[i]] = i
        new_adj = diffuse_graph(g, adj, diff_range, adj_index)
        self.assertGreater(np.mean(new_adj), 0)

    def test_null_graph_equal(self):
        """Checks if a permuted graph with identical number of edges is generated. """
        null = null_graph(g)
        self.assertEqual(len(null.edges), len(g.edges))

    def test_null_graph_difference(self):
        """Checks if the resulting graph is permuted. """
        null = null_graph(g)
        self.assertNotEqual(list(null.edges), list(g.edges))

    def test_bootstrap(self):
        """Checks if p-values for the graph are returned. """
        results = cluster_graph(deepcopy(g), limit, diff_range, max_clusters, iterations)
        boots = list()
        for i in range(100):
            bootstrap = null_graph(g)
            adj = np.zeros((len(g.nodes), len(g.nodes)))
            boot_index = dict()
            for k in range(len(g.nodes)):
                boot_index[list(g.nodes)[k]] = k
            for j in range(iterations):
                adj = diffuse_graph(bootstrap, adj, diff_range, boot_index)
            boots.append(adj)
        bootmats = bootstrap_test(results[2], boots)
        self.assertEqual(results[2].shape, bootmats.shape)


if __name__ == '__main__':
    unittest.main()
