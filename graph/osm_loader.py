import osmnx as ox


def load_city_graph(place_name: str):
    """
    Load a drivable road network from OpenStreetMap.
    """

    print(f"Loading road network for: {place_name}")

    # Load the road network.
    # OSMnx handles the graph creation.
    G = ox.graph_from_place(
        place_name,
        network_type="drive",
        simplify=True
    )

    # Add speed information.
    G = ox.add_edge_speeds(
        G,
        fallback=30
    )

    # Add travel time information.
    G = ox.add_edge_travel_times(G)

    print(
        f"Graph loaded: "
        f"{G.number_of_nodes():,} nodes, "
        f"{G.number_of_edges():,} edges"
    )

    return G