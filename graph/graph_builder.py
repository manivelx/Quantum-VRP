import networkx as nx


def describe_graph(G):
    """
    Return basic information about the road graph.
    """

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "directed": G.is_directed()
    }


def validate_graph(G):
    """
    Basic validation of the road network.
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Road graph contains no nodes.")

    if G.number_of_edges() == 0:
        raise ValueError("Road graph contains no edges.")

    if not G.is_directed():
        raise ValueError(
            "VRP road network should be directed."
        )

    return True