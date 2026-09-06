import networkx as nx


def build_vehicle_navigation_route(
    G,
    location_nodes,
    route
):
    """
    Convert a QPSO vehicle route into
    an actual road-network navigation path.

    route example:
        [3, 5, 2]

    location_nodes:
        index 0 = depot
        index 1...N = customers
    """

    complete_route = [0] + list(route) + [0]

    full_path = []

    for i in range(len(complete_route) - 1):

        start_index = complete_route[i]
        end_index = complete_route[i + 1]

        start_node = location_nodes[start_index]
        end_node = location_nodes[end_index]

        try:

            segment = nx.shortest_path(
                G,
                start_node,
                end_node,
                weight="length"
            )

        except nx.NetworkXNoPath:

            return []

        if i == 0:
            full_path.extend(segment)

        else:
            # Avoid duplicate connection node
            full_path.extend(segment[1:])

    return full_path


def navigation_coordinates(G, path):
    """
    Convert navigation path into
    [latitude, longitude] coordinates.
    """

    coordinates = []

    for node in path:

        coordinates.append([
            G.nodes[node]["y"],
            G.nodes[node]["x"]
        ])

    return coordinates


def navigation_distance(G, path):
    """
    Calculate actual road distance
    of a navigation path in km.
    """

    distance = 0.0

    for i in range(len(path) - 1):

        start = path[i]
        end = path[i + 1]

        edge_data = G.get_edge_data(
            start,
            end
        )

        if not edge_data:
            continue

        edge = min(
            edge_data.values(),
            key=lambda x: x.get(
                "length",
                float("inf")
            )
        )

        distance += edge.get(
            "length",
            0
        )

    return distance / 1000.0