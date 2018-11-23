#!/usr/bin/env python

"""
Although NetworkX already contains read/write utilities for Cyjson networks,
these do not accomodate networks without an index (e.g. those generated with CoNet).
Moreover, they do not support inclusion of a network layout.

The functions below have been adapted to tackle these issues.
They are derived from the original NetworkX read/write functions.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'


import networkx as nx
import json


def read_cyjson(filename):
    """Small utility function for reading Cytoscape json files
    generated with CoNet.

    :param filename: Filepath to location of cyjs file.
    :return: NetworkX graph.

    Adapted from the NetworkX cytoscape_graph function.

    License
    =======

    NetworkX is distributed with the 3-clause BSD license.

    ::

       Copyright (C) 2004-2018, NetworkX Developers
       Aric Hagberg <hagberg@lanl.gov>
       Dan Schult <dschult@colgate.edu>
       Pieter Swart <swart@lanl.gov>
       All rights reserved.

       Redistribution and use in source and binary forms, with or without
       modification, are permitted provided that the following conditions are
       met:

         * Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.

         * Redistributions in binary form must reproduce the above
           copyright notice, this list of conditions and the following
           disclaimer in the documentation and/or other materials provided
           with the distribution.

         * Neither the name of the NetworkX Developers nor the names of its
           contributors may be used to endorse or promote products derived
           from this software without specific prior written permission.

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
       "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
       A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
       OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
       SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
       LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
       THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
       (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    with open(filename) as f:
        data = json.load(f)
    name = 'name'
    ident = 'id'
    if len(set([ident, name])) < 2:
        raise nx.NetworkXError('Attribute names are not unique.')
    graph = nx.Graph()
    graph.graph = dict(data.get('data'))
    i = 0
    for d in data["elements"]["nodes"]:
        # only modification: 'value' key is not included in CoNet output
        # now graph only needs ID and name values
        node_data = d["data"].copy()
        try:
            node = d["data"].get(ident)
        except KeyError:
            # if no index is found, one is generated
            node = i
            i += 1
        if d["data"].get(name):
            node_data[name] = d["data"].get(name)
        graph.add_node(node)
        graph.nodes[node].update(node_data)
    for d in data["elements"]["edges"]:
        edge_data = d["data"].copy()
        sour = d["data"].pop("source")
        targ = d["data"].pop("target")
        graph.add_edge(sour, targ)
        graph.edges[sour, targ].update(edge_data)
    if 'interactionType' in graph.edges[list(graph.edges)[0]]:
        # this indicates this is a CoNet import
        for edge in graph.edges:
            if graph.edges[edge]['interactionType'] == 'copresence':
                graph.edges[edge]['weight'] = 1
            elif graph.edges[edge]['interactionType'] == 'mutualExclusion':
                graph.edges[edge]['weight'] = -1
            else:
                graph.edges[edge]['weight'] = 0
    return graph


def write_cyjson(filename, graph, layout=None):
    """Small utility function for writing Cytoscape json files.
    Also accepts a layout dictionary to add to the file.

    :param filename: Filepath to location of cyjs file.
    :param graph: NetworkX graph to write to disk.
    :param layout: Dictionary of layout coordinates that is written to cyjs file.
    :return:

    Adapted from the NetworkX cytoscape_graph function.

    License
    =======

    NetworkX is distributed with the 3-clause BSD license.

    ::

       Copyright (C) 2004-2018, NetworkX Developers
       Aric Hagberg <hagberg@lanl.gov>
       Dan Schult <dschult@colgate.edu>
       Pieter Swart <swart@lanl.gov>
       All rights reserved.

       Redistribution and use in source and binary forms, with or without
       modification, are permitted provided that the following conditions are
       met:

         * Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.

         * Redistributions in binary form must reproduce the above
           copyright notice, this list of conditions and the following
           disclaimer in the documentation and/or other materials provided
           with the distribution.

         * Neither the name of the NetworkX Developers nor the names of its
           contributors may be used to endorse or promote products derived
           from this software without specific prior written permission.

       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
       "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
       LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
       A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
       OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
       SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
       LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
       THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
       (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    name = 'name'
    ident = 'id'

    jsondata = {"data": list(graph.graph.items())}
    jsondata["elements"] = {"nodes": [], "edges": []}
    nodes = jsondata["elements"]["nodes"]
    edges = jsondata["elements"]["edges"]

    for i, j in graph.nodes.items():
        n = {"data": j.copy()}
        n["data"]["id"] = j.get(ident) or str(i)
        n["data"]["value"] = i
        n["data"]["name"] = j.get(name) or str(i)
        if layout:
            n["position"] = dict()
            n["position"]["x"] = layout[i][0]
            n["position"]["y"] = layout[i][1]
        nodes.append(n)

    for e in graph.edges():
        n = {"data": graph.adj[e[0]][e[1]].copy()}
        n["data"]["source"] = e[0]
        n["data"]["target"] = e[1]
        edges.append(n)

    with open(filename, 'w') as outfile:
        json.dump(jsondata, outfile)
