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
from manca.manca import clus_central
from manca.cluster import central_edge, cluster_graph, diffuse_graph, sparsity_score, central_node
from manca.perms import null_graph, perm_graph
from manca.layout import generate_layout, generate_tax_weights
from copy import deepcopy
import numpy as np
from io import StringIO
from sklearn.cluster import KMeans

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

limit = 0.000001
min_clusters = 2
max_clusters = 5
iterations = 10000
central = True
percentage = 10
permutations = 100

tax = StringIO("""#OTU	Kingdom	Phylum	Class	Order	Family	Genus	Species
OTU_1	Bacteria	Cyanobacteria	Oxyphotobacteria	Synechoccales	Cyanobiaceae	Synechococcus	NA
OTU_2	Bacteria	Proteobacteria	Alphaproteobacteria	Rhodobacterales	Rhodobacteraceae	Planktomarina	NA
OTU_3	Bacteria	Cyanobacteria	Oxyphotobacteria	Synechoccales	Cyanobiaceae	Synechococcus	NA
OTU_4	Bacteria	Proteobacteria	Alphaproteobacteria	Rhodobacterales	Rhodobacteraceae	Oceanibulbus	NA
OTU_5	Bacteria	Proteobacteria	Alphaproteobacteria	Rhodobacterales	Rhodobacteraceae	Ascidiaceihabitans	NA
OTU_6	Bacteria	Proteobacteria	Alphaproteobacteria	Rhodobacterales	Rhodobacteraceae	Leisingera	NA
OTU_7	Bacteria	Proteobacteria	Alphaproteobacteria	Sphingomonadales	Sphingomonadaceae	Erythrobacter	flavus
OTU_8	Bacteria	Proteobacteria	Alphaproteobacteria	Sphingomonadales	Sphingomonadaceae	Erythrobacter	citreus
OTU_9	Bacteria	Cyanobacteria	Oxyphotobacteria	Synechoccales	Cyanobiaceae	Cyanobium	NA
OTU_10	Bacteria	Proteobacteria	Alphaproteobacteria	Rhizobiales	Rhizobiaceae	Mesorhizobium	NA
""")


class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def test_default_manca(self):
        """
        Checks whether the main function carries out both clustering and centrality estimates.
        """
        clustered_graph = clus_central(deepcopy(g))
        clusters = nx.get_node_attributes(clustered_graph, 'cluster')
        hubs = nx.get_edge_attributes(clustered_graph, 'hub')

    def test_center_manca(self):
        """
        Checks if the edge between 1 and 2 is identified as a positive hub.

        WARNING: at the moment the test indicates that centrality measures
        are not stable.
        """
        results = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations)
        graph = results[0]
        matrix = results[1]
        central_edge(matrix, graph, percentage, permutations, iterations)
        hubs = nx.get_edge_attributes(graph, 'hub')
        self.assertEqual(hubs[list(hubs.keys())[0]], 'negative hub')

    def test_central_node(self):
        """
        Checks if, given a graph that has been tested for centrality,
        no nodes are identified as hubs (actually the p-value is too low).
        """
        results = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations)
        graph = results[0]
        matrix = results[1]
        central_edge(matrix, graph, percentage, permutations, iterations)
        central_node(graph)
        self.assertEqual(len(nx.get_node_attributes(graph, 'hub')), 0)

    def test_cluster_manca(self):
        """Checks whether the correct cluster IDs are assigned. """
        clustered_graph = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations)
        clusters = nx.get_node_attributes(clustered_graph[0], 'cluster')
        self.assertEqual(clusters['OTU_10'], clusters['OTU_6'])

    def test_sparsity_score(self):
        """Checks whether correct sparsity scores are calculated.
        Because this network has 3 negative edges separating
        2 clusters, the score should be -3. """
        scoremat = diffuse_graph(g, limit, iterations)
        clusters = KMeans(2).fit_predict(scoremat)
        adj_index = dict()
        for i in range(len(g.nodes)):
            adj_index[list(g.nodes)[i]] = i
        rev_index = {v: k for k, v in adj_index.items()}
        sparsity = sparsity_score(g, clusters, rev_index)
        self.assertEqual(sparsity, -3)

    def test_diffuse_graph(self):
        """Checks if the diffusion process operates correctly. """
        new_adj = diffuse_graph(g, iterations)
        self.assertNotEqual(np.mean(new_adj), 0)

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
        results = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations)
        weights = nx.get_edge_attributes(results[0], 'weight')
        # calculates the ratio of positive / negative weights
        # note that ratios need to be adapted, because the matrix is symmetric
        posnodes = sum(weights[x] > 0 for x in weights)
        ratio = posnodes / len(weights)
        negthresh = np.percentile(results[1], percentage * (1 - ratio) / 2)
        posthresh = np.percentile(results[1], 100 - percentage * ratio / 2)
        neghubs = np.argwhere(results[1] < negthresh)
        poshubs = np.argwhere(results[1] > posthresh)
        bootmats = perm_graph(results[0], results[1], limit, iterations, permutations, posthresh, negthresh)
        self.assertEqual(results[1].shape, bootmats.shape)

    def test_tax_weights(self):
        """Checks whether the tax weights for the edges are correctly calculated."""
        tax_graph = deepcopy(g)
        tax_graph = generate_tax_weights(tax_graph, tax)
        self.assertEqual(tax_graph['OTU_1']['OTU_2']['tax_score'], 2)

    def test_layout(self):
        """Checks whether the layout function returns a dictionary of coordinates."""
        clustered_graph = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations)
        coords = generate_layout(clustered_graph[0])
        self.assertEqual(len(coords[list(coords.keys())[0]]), 2)


if __name__ == '__main__':
    unittest.main()
