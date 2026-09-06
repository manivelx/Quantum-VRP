import networkx as nx
import numpy as np


def calculate_distance_matrix(G, location_nodes):
    """
    Calculate shortest-road distance between
    every depot/customer pair.

    Returns distance in kilometers.
    """

    nodes = list(location_nodes.values())

    n = len(nodes)

    matrix = np.full(
        (n, n),
        np.inf
    )

    for i, source in enumerate(nodes):

        lengths = nx.single_source_dijkstra_path_length(
            G,
            source,
            weight="length"
        )

        for j, target in enumerate(nodes):

            if target in lengths:
                matrix[i][j] = lengths[target] / 1000.0

    return matrix


def calculate_time_matrix(G, location_nodes):
    """
    Calculate shortest-road travel time between
    every depot/customer pair.

    Returns travel time in minutes.
    """

    nodes = list(location_nodes.values())

    n = len(nodes)

    matrix = np.full(
        (n, n),
        np.inf
    )

    for i, source in enumerate(nodes):

        lengths = nx.single_source_dijkstra_path_length(
            G,
            source,
            weight="travel_time"
        )

        for j, target in enumerate(nodes):

            if target in lengths:
                matrix[i][j] = lengths[target] / 60.0

    return matrix


def get_shortest_route(G, source, target):
    """
    Return the actual road-network path
    between two nodes.
    """

    return nx.shortest_path(
        G,
        source,
        target,
        weight="travel_time"
    )