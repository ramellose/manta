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
from manca.manca import cluster_graph

g = nx.Graph()
nodes = ["OTU_1", "OTU_2", "OTU_3", "OTU_4", "OTU_5",
         "OTU_6", "OTU_7", "OTU_8", "OTU_9", "OTU_10"]
g.add_nodes_from(nodes)
g.add_edges_from([("OTU_1", "OTU_2"), ("OTU_1", "OTU_3"),
                  ("OTU_2", "OTU_5"), ("OTU_3", "OTU_4"),
                  ("OTU_4", "OTU_8"), ("OTU_5", "OTU_7"),
                  ("OTU_10", "OTU_1"), ("OTU_7", "OTU_8"),
                  ("OTU_10", "OTU_9"), ("OTU_9", "OTU_6")])
g["OTU_1"]["OTU_2"]['weight'] = float(1.0)
g["OTU_1"]["OTU_3"]['weight'] = float(1.0)
g["OTU_2"]["OTU_5"]['weight'] = float(1.0)
g["OTU_3"]["OTU_4"]['weight'] = float(-1.0)
g["OTU_4"]["OTU_8"]['weight'] = float(-1.0)
g["OTU_5"]["OTU_7"]['weight'] = float(1.0)
g["OTU_10"]["OTU_1"]['weight'] = float(-1.0)
g["OTU_7"]["OTU_8"]['weight'] = float(1.0)
g["OTU_10"]["OTU_9"]['weight'] = float(1.0)
g["OTU_9"]["OTU_6"]['weight'] = float(1.0)

g = g.to_undirected()

class TestMain(unittest.TestCase):
    """"
    Tests whether the main clustering function properly assigns cluster IDs.
    """

    def test_default_manca(self):
        """
        Checks the cluster assignments are correct;
        """
        clustered_graph = cluster_graph(g)
        self.assertEqual(clustered_graph.node['OTU_10'],
                         clustered_graph.node['OTU_6'])


if __name__ == '__main__':
    unittest.main()
