#!/usr/bin/env python

"""
These functions generate a layout for microbial association networks.
Instead of using default edge weight, a different edge weight is calculated
that can take taxonomic similarity as well as structural similarity into account.
Nodes that share taxonomic features will be located more closely together,
as well as nodes that share neighbours.

The function uses this alternative edge weight to run the Fruchterman-Reingold force-directed algorithm.
This algorithm is run once per cluster.
The generated layouts are then rotated depending on the number of clusters and combined.
"""

__author__ = 'Lisa Rottjers'
__email__ = 'lisa.rottjers@kuleuven.be'
__status__ = 'Development'
__license__ = 'Apache 2.0'

import networkx as nx
import sys
from math import sin, cos, radians
import csv
from copy import deepcopy


def generate_layout(graph, tax=None):
    """
    Generates a dictionary of layout coordinates.
    The layout is based on the Fruchterman-Reingold force-directed algorithm,
    where a layout is calculated for each of the clusters specified
    in the supplied NetworkX graph. These layouts are then shifted and combined
    into a full layout.

    Parameters
    ----------
    :param graph: NetworkX graph with cluster IDs
    :param tax: Filepath to tab-delimited taxonomy table.
    :return: dictionary of layout coordinates
    """
    try:
        clusters = nx.get_node_attributes(graph, 'cluster')
    except KeyError:
        sys.stdout.write('Graph does not appear to have a cluster attribute. ' + '\n')
        sys.stdout.flush()
    total = list()
    [total.append(clusters[x]) for x in clusters]
    num_clusters = list(set(total))
    coord_list = list()
    total_nodes = len(graph.nodes)
    for i in range(len(num_clusters)):
        cluster = num_clusters[i]
        node_list = list()
        for node in clusters:
            if clusters[node] == cluster:
                node_list.append(node)
        clustgraph = graph.subgraph(node_list)
        if tax:
            with open(tax, 'r') as taxdata:
                clustgraph = generate_tax_weights(clustgraph, taxdata)
            subcoords = nx.spring_layout(clustgraph, weight='tax_score')
        else:
            clustgraph = generate_tax_weights(clustgraph, tax=None)
            subcoords = nx.spring_layout(clustgraph, weight=None)
        # currently, weight attribute is set to None
        # phylogenetic similarity would be nice though
        for node in subcoords:
            # need to scale coordinate system and transpose
            # spring_layout coordinates are placed in box of size[0,1]
            # transpose them vertically to box of size[0, 1*number of clusters]
            # but only if more than 10 nodes; otherwise they are placed at the center
            subcoords[node][0] = subcoords[node][0] * (len(clustgraph.nodes) / total_nodes) * 10
            subcoords[node][1] = subcoords[node][1] * (len(clustgraph.nodes) / total_nodes) * 10
            if len(clustgraph.nodes) > 10:
                subcoords[node][1] += len(num_clusters)
            else:
                subcoords[node][1] += (len(clustgraph.nodes) * 0.3)
        coord_list.append(subcoords)
    angles = 360 / len(num_clusters)
    spins = 0
    # each cluster is rotated around coordinates [0,0]
    new_coords = dict()
    while spins < 361:
        for subcoords in coord_list:
            spins += angles
            for coord in subcoords:
                new_x = cos(radians(spins)) * subcoords[coord][0] - sin(radians(spins)) * subcoords[coord][1]
                new_y = sin(radians(spins)) * subcoords[coord][0] + cos(radians(spins)) * subcoords[coord][1]
                new_x = new_x * 100
                new_y = new_y * 100
                new_coords[coord] = [new_x, new_y]
    return new_coords


def generate_tax_weights(graph, tax):
    """
    Returns supplied graph with node similarity as node properties.
    This node similarity is determined by
    similarity at different taxonomic levels and by structural similarity.
    The more similar nodes are, the larger their similarity score is;
    hence, force-directed layouts will place such nodes closer together.
    Assumes that species assignments of NA or None are not relevant.

    Parameters
    ----------
    :param graph: NetworkX graph
    :param tax: Taxonomy table
    :return:
    """
    tax_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    tax_weights = dict()
    if tax:
        taxdict = dict()
        for item in tax_levels:
            taxdict[item] = dict()
        taxdata = csv.reader(tax, delimiter='\t')
        header = next(taxdata)
        for row in taxdata:
            for i in range(1, len(row)):
                if row[i] != 'NA' or 'None':
                    taxdict[tax_levels[i-1]][row[0]] = row[i]
        for edge in graph.edges:
            score = 7
            # maximum score 7 means all taxonomic levels are equal
            # attractive force in spring layout algorithm is then largest
            for i in range(7):
                try:
                    if taxdict[tax_levels[7-i]][edge[0]] != taxdict[tax_levels[7-i]][edge[1]]:
                        score -= 1
                except IndexError:
                    pass
            # can also correct for interaction similarity
            # structural equivalence: if nodes share many neighbours, they are similar
            # even if taxonomic similarity is low, similar interaction patterns -> high score
            # using Jaccard coefficient
            tax_weights[edge] = score
    else:
        # if no taxonomy table is supplied, taxonomy can be present as node property
        example_attrs = graph.nodes[list(graph.nodes)[0]]
        tax_in_network = False
        tax_network = deepcopy(tax_levels)
        for attribute in example_attrs:
            if attribute.lower() in tax_levels:
                # need to make sure attribute names are correct, can be capitalized
                tax_id = [tax_levels.index(i) for i in tax_levels if attribute.lower() in i]
                tax_network[tax_id[0]] = attribute
                tax_in_network = True
        if tax_in_network:
            for edge in graph.edges:
                score = 7
                # maximum score 7 means all taxonomic levels are equal
                # attractive force in spring layout algorithm is then largest
                for i in range(7):
                    try:
                        if graph.nodes[edge[0]][tax_network[7-i]] != graph.nodes[edge[1]][tax_network[7-i]]:
                            score -= 1
                    except IndexError:
                        pass
                # can also correct for interaction similarity
                # structural equivalence: if nodes share many neighbours, they are similar
                # even if taxonomic similarity is low, similar interaction patterns -> high score
                # using Jaccard coefficient
                tax_weights[edge] = score
        else:
            # if no attribute has taxonomic values
            for edge in graph.edges:
                tax_weights[edge] = 1
    jaccard = nx.jaccard_coefficient(graph, list(tax_weights.keys()))
    for u, v, p in jaccard:
        edge = (u, v)
        if p > 0:
            # average taxonomic similarity will easily be 2 or 3
            # high neighbour similarity should be more important
            # 14 is chosen as an arbitrary scaling factor
            p_scaled = p * 14
            tax_weights[edge] = tax_weights[edge] + p_scaled
    nx.set_edge_attributes(graph, tax_weights, 'tax_score')
    return graph
