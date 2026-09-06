import networkx as nx


def build_distance_matrix(G, location_nodes):
    """
    Build a road-distance matrix between all snapped locations.

    location_nodes:
        List of OSM road node IDs.

    Returns:
        NumPy-compatible 2D distance matrix in km.
    """

    n = len(location_nodes)

    distance_matrix = []

    for i in range(n):

        row = []

        for j in range(n):

            if i == j:
                row.append(0.0)
                continue

            try:

                distance = nx.shortest_path_length(
                    G,
                    location_nodes[i],
                    location_nodes[j],
                    weight="length"
                )

                # OSMnx/networkx edge length is normally metres
                distance_km = distance / 1000.0

                row.append(distance_km)

            except (
                nx.NetworkXNoPath,
                nx.NodeNotFound
            ):

                # Very large value = unreachable
                row.append(1e9)

        distance_matrix.append(row)

    return distance_matrix