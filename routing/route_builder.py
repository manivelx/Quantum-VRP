import networkx as nx


def convert_location_route_to_nodes(
    snapped_locations,
    location_route
):
    """
    Convert customer/depot IDs into OSM road node IDs.

    Example:

    ["DEPOT", "C1", "C3", "DEPOT"]

    becomes:

    [12345, 67890, 54321, 12345]
    """

    node_sequence = []

    for location_id in location_route:

        if location_id not in snapped_locations:
            raise ValueError(
                f"Location '{location_id}' "
                f"not found in snapped locations."
            )

        node_sequence.append(
            snapped_locations[location_id]["node"]
        )

    return node_sequence


def build_route_path(G, node_sequence):
    """
    Find the actual shortest road path between
    every consecutive location in the route.
    """

    complete_path = []

    for i in range(len(node_sequence) - 1):

        start_node = node_sequence[i]
        end_node = node_sequence[i + 1]

        try:

            path = nx.shortest_path(
                G,
                start_node,
                end_node,
                weight="travel_time"
            )

        except nx.NetworkXNoPath:

            path = nx.shortest_path(
                G,
                start_node,
                end_node,
                weight="length"
            )

        if i > 0:
            path = path[1:]

        complete_path.extend(path)

    return complete_path


def route_coordinates(G, node_sequence):
    """
    Convert the road-node path into
    [latitude, longitude] coordinates
    for Folium.
    """

    road_path = build_route_path(
        G,
        node_sequence
    )

    coordinates = []

    for node in road_path:

        latitude = G.nodes[node]["y"]
        longitude = G.nodes[node]["x"]

        coordinates.append(
            [latitude, longitude]
        )

    return coordinates