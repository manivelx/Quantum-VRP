# graph/location_snapper.py

import osmnx as ox
import networkx as nx


def _get_routing_graph(G):
    """
    Return a strongly connected directed component
    suitable for routing.

    For a directed OSM road graph, using the largest
    strongly connected component prevents locations from
    being snapped onto isolated one-way sections that
    cannot participate in complete routes.
    """

    if G is None or len(G.nodes) == 0:
        raise ValueError("Road network is empty.")

    # OSMnx road graphs are normally MultiDiGraph.
    if not G.is_directed():
        return G

    components = list(
        nx.strongly_connected_components(G)
    )

    if not components:
        raise ValueError(
            "No strongly connected component found "
            "in the road network."
        )

    largest_component = max(
        components,
        key=len
    )

    routing_graph = G.subgraph(
        largest_component
    ).copy()

    return routing_graph


def snap_locations(G, locations):
    """
    Snap customer/depot coordinates to the nearest
    road-network node.

    The snapping is performed on the largest strongly
    connected component of the directed road network.

    Parameters
    ----------
    G : networkx graph
        OSM road network.

    locations : dict
        Dictionary containing location information.

    Returns
    -------
    dict
        Snapped locations with OSM road-node IDs.
    """

    if not locations:
        return {}

    routing_graph = _get_routing_graph(G)

    # Extract coordinates of routing nodes.
    node_ids = list(
        routing_graph.nodes
    )

    if not node_ids:
        raise ValueError(
            "Routing graph contains no nodes."
        )

    X = [
        routing_graph.nodes[node]["x"]
        for node in node_ids
    ]

    Y = [
        routing_graph.nodes[node]["y"]
        for node in node_ids
    ]

    snapped_locations = {}

    for location_id, data in locations.items():

        latitude = float(
            data["latitude"]
        )

        longitude = float(
            data["longitude"]
        )

        # ----------------------------------------------------
        # Find nearest node
        # ----------------------------------------------------

        nearest_node = ox.distance.nearest_nodes(
            routing_graph,
            X=longitude,
            Y=latitude
        )

        node_latitude = routing_graph.nodes[
            nearest_node
        ]["y"]

        node_longitude = routing_graph.nodes[
            nearest_node
        ]["x"]

        # ----------------------------------------------------
        # Preserve original location information
        # ----------------------------------------------------

        snapped_locations[location_id] = {

            "type": data["type"],

            "latitude": latitude,

            "longitude": longitude,

            "demand": float(
                data.get(
                    "demand",
                    0
                )
            ),

            "node": nearest_node,

            "snapped_latitude": node_latitude,

            "snapped_longitude": node_longitude
        }

    return snapped_locations