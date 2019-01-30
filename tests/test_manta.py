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
from manta.manta import clus_central
from manta.cluster import cluster_graph, sparsity_score, cluster_fuzzy
from manta.perms import rewire_graph, perm_graph, diffusion
from manta.centrality import central_edge, central_node
from manta.layout import generate_layout, generate_tax_weights
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
iterations = 20
central = True
percentile = 10
permutations = 100
error = 0.1

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
        self.assertGreater(len(hubs), 0)

    def test_center_manca(self):
        """
        Checks if the edge between 1 and 2 is identified as a positive hub.

        WARNING: at the moment the test indicates that centrality measures
        are not stable.
        """
        results = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations, edgescale=0.5)
        graph = results[0]
        central_edge(graph, percentile, permutations, error)
        hubs = nx.get_edge_attributes(graph, 'hub')
        self.assertEqual(hubs[list(hubs.keys())[0]], 'negative hub')

    def test_central_node(self):
        """
        Checks if, given a graph that has been tested for centrality,
        no nodes are identified as hubs (actually the p-value is too low).
        """
        results = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations, edgescale=0.5)
        graph = results[0]
        central_edge(graph, percentile, permutations, error)
        central_node(graph)
        self.assertEqual(len(nx.get_node_attributes(graph, 'hub')), 0)

    def test_cluster_manca(self):
        """Checks whether the correct cluster IDs are assigned. """
        clustered_graph = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations, edgescale=0.5)
        clusters = nx.get_node_attributes(clustered_graph[0], 'cluster')
        self.assertEqual(clusters['OTU_10'], clusters['OTU_6'])

    def test_sparsity_score(self):
        """Checks whether correct sparsity scores are calculated.
        Because this network has 3 negative edges separating
        2 clusters, the score should be -3 + the penalty of 2000. """
        scoremat = diffusion(g, limit, iterations)[0]
        clusters = KMeans(2).fit_predict(scoremat)
        adj_index = dict()
        for i in range(len(g.nodes)):
            adj_index[list(g.nodes)[i]] = i
        rev_index = {v: k for k, v in adj_index.items()}
        sparsity = sparsity_score(g, clusters, rev_index)
        self.assertEqual(int(sparsity), 1)

    def test_diffuse_graph(self):
        """Checks if the diffusion process operates correctly. """
        new_adj = diffusion(g, iterations=iterations)[0]
        self.assertNotEqual(np.mean(new_adj), 0)

    def test_rewire_graph_equal(self):
        """Checks if a permuted graph with identical number of edges is generated. """
        null = rewire_graph(g, error)
        self.assertEqual(len(null.edges), len(g.edges))

    def test_rewire_graph_difference(self):
        """Checks if the resulting graph is permuted. """
        null = rewire_graph(g, error)
        self.assertNotEqual(list(null.edges), list(g.edges))

    def test_bootstrap(self):
        """Checks if reliability scores for the graph are returned. """
        results = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations, edgescale=0.5)
        # calculates the ratio of positive / negative weights
        # note that ratios need to be adapted, because the matrix is symmetric
        matrix = results[1]
        negthresh = np.percentile(matrix, percentile)
        posthresh = np.percentile(matrix, 100 - percentile)
        neghubs = list(map(tuple, np.argwhere(matrix <= negthresh)))
        poshubs = list(map(tuple, np.argwhere(matrix >= posthresh)))
        bootmats = perm_graph(g, permutations, percentile, poshubs, neghubs, error=0.1)
        self.assertEqual(100, len(bootmats))

    def test_tax_weights(self):
        """Checks whether the tax weights for the edges are correctly calculated."""
        tax_graph = deepcopy(g)
        tax_graph = generate_tax_weights(tax_graph, tax)
        self.assertEqual(tax_graph['OTU_1']['OTU_2']['tax_score'], 2)

    def test_layout(self):
        """Checks whether the layout function returns a dictionary of coordinates."""
        clustered_graph = cluster_graph(deepcopy(g), limit, max_clusters, min_clusters, iterations, edgescale=0.5)
        coords = generate_layout(clustered_graph[0])
        self.assertEqual(len(coords[list(coords.keys())[0]]), 2)

    def test_cluster_fuzzy(self):
        """Cluster_hard is already tested through the regular cluster_graph function.
        To test cluster_fuzzy, this function carries out clustering as
        if a memory effect has been detected."""
        graph = deepcopy(g)
        adj_index = dict()
        for i in range(len(graph.nodes)):
            adj_index[list(graph.nodes)[i]] = i
        rev_index = {v: k for k, v in adj_index.items()}
        scoremat, memory, diffs = diffusion(graph=graph, limit=limit, iterations=iterations)
        bestcluster = cluster_fuzzy(graph=graph, rev_index=rev_index, adj_index=adj_index,
                                    diffs=diffs, scoremat=scoremat, edgescale=0.5)
        # So some nodes with contrasting signs are actually identified..
        self.assertEqual(bestcluster[0], 0)


if __name__ == '__main__':
    unittest.main()
