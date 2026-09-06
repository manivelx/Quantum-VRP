import os
import sys
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# ============================================================
# CONFIGURATION
# ============================================================

PLACE = "Koregaon Park, Pune, India"

BEST_ROUTE_FILE = os.path.join(
    PROJECT_ROOT,
    "best_route.pkl"
)


# ============================================================
# LOAD BEST ROUTE
# ============================================================

def load_best_route():

    if not os.path.exists(
        BEST_ROUTE_FILE
    ):

        raise FileNotFoundError(
            "best_route.pkl not found.\n"
            "Run compare_algorithms.py first."
        )

    with open(
        BEST_ROUTE_FILE,
        "rb"
    ) as file:

        data = pickle.load(file)

    return data


# ============================================================
# LOAD ROAD NETWORK
# ============================================================

def load_network():

    print()
    print(
        f"Loading road network: {PLACE}"
    )

    from graph.osm_loader import load_city_graph

    G = load_city_graph(
        PLACE
    )

    print(
        f"Road nodes: {len(G.nodes):,}"
    )

    print(
        f"Road edges: {len(G.edges):,}"
    )

    return G


# ============================================================
# GET ROUTE NODES
# ============================================================

def get_route_nodes(
    G,
    location_names,
    routes,
    locations
):

    route_nodes = []

    for route in routes:

        nodes = []

        # ----------------------------------------------------
        # DEPOT
        # ----------------------------------------------------

        depot_name = location_names[0]

        depot_node = locations[
            depot_name
        ]["node"]

        nodes.append(
            depot_node
        )

        # ----------------------------------------------------
        # CUSTOMERS
        # ----------------------------------------------------

        for customer_index in route:

            customer_name = (
                location_names[
                    customer_index
                ]
            )

            customer_node = locations[
                customer_name
            ]["node"]

            nodes.append(
                customer_node
            )

        # ----------------------------------------------------
        # RETURN TO DEPOT
        # ----------------------------------------------------

        nodes.append(
            depot_node
        )

        route_nodes.append(
            nodes
        )

    return route_nodes


# ============================================================
# BUILD ROUTE PATH
# ============================================================

def build_route_path(
    G,
    route_nodes
):

    planning_graph = G

    if G.is_directed():

        planning_graph = G.to_undirected(
            reciprocal=False
        )

    complete_path = []

    for i in range(
        len(route_nodes) - 1
    ):

        start = route_nodes[i]

        end = route_nodes[i + 1]

        try:

            path = nx.shortest_path(
                planning_graph,
                start,
                end,
                weight="length"
            )

        except nx.NetworkXNoPath:

            continue

        if complete_path:

            complete_path.extend(
                path[1:]
            )

        else:

            complete_path.extend(
                path
            )

    return complete_path


# ============================================================
# VISUALIZE ROUTES
# ============================================================

def visualize_routes(
    G,
    route_nodes,
    location_names,
    routes,
    locations,
    algorithm_name
):

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.8,
        bgcolor="white",
        show=False,
        close=False
    )

    # --------------------------------------------------------
    # DRAW EACH VEHICLE ROUTE
    # --------------------------------------------------------

    for vehicle_number, nodes in enumerate(
        route_nodes
    ):

        route_path = build_route_path(
            G,
            nodes
        )

        if not route_path:
            continue

        xs = []
        ys = []

        for node in route_path:

            xs.append(
                G.nodes[node]["x"]
            )

            ys.append(
                G.nodes[node]["y"]
            )

        ax.plot(
            xs,
            ys,
            linewidth=3,
            label=f"Vehicle {vehicle_number + 1}"
        )

    # --------------------------------------------------------
    # DRAW LOCATION POINTS
    # --------------------------------------------------------

    for index, name in enumerate(
        location_names
    ):

        node = locations[
            name
        ]["node"]

        x = G.nodes[node]["x"]

        y = G.nodes[node]["y"]

        if index == 0:

            ax.scatter(
                x,
                y,
                s=120,
                marker="*",
                zorder=10
            )

            ax.text(
                x,
                y,
                "  DEPOT",
                fontsize=10,
                fontweight="bold"
            )

        else:

            ax.scatter(
                x,
                y,
                s=60,
                zorder=10
            )

            ax.text(
                x,
                y,
                f"  {name}",
                fontsize=8
            )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        f"BLACK ROCK VRP - {algorithm_name}",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        PROJECT_ROOT,
        "best_route_map.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print()
    print(
        f"Route map saved to: "
        f"{output_file}"
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)

    print(
        "BLACK ROCK VRP - ROUTE VISUALIZATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # LOAD BEST RESULT
    # --------------------------------------------------------

    best_data = load_best_route()

    algorithm_name = (
        best_data["algorithm"]
    )

    routes = (
        best_data["routes"]
    )

    location_names = (
        best_data["location_names"]
    )

    print()

    print(
        f"Best Algorithm : "
        f"{algorithm_name}"
    )

    print(
        f"Best Distance  : "
        f"{best_data['distance']:.3f} km"
    )

    print(
        f"Vehicles Used  : "
        f"{best_data['vehicles']}"
    )

    # --------------------------------------------------------
    # LOAD ROAD NETWORK
    # --------------------------------------------------------

    G = load_network()

    # --------------------------------------------------------
    # LOAD CUSTOMER DATA
    # --------------------------------------------------------

    from benchmarks.run_qpso import load_customers

    customers_df = load_customers()

    locations = {}

    for _, row in customers_df.iterrows():

        locations[row["id"]] = {

            "type": row["type"],

            "latitude": float(
                row["latitude"]
            ),

            "longitude": float(
                row["longitude"]
            ),

            "demand": float(
                row["demand"]
            )
        }

    # --------------------------------------------------------
    # SNAP LOCATIONS
    # --------------------------------------------------------

    from graph.location_snapper import snap_locations

    print()

    print(
        "Snapping locations..."
    )

    snapped = snap_locations(
        G,
        locations
    )

    # --------------------------------------------------------
    # ADD NODE INFORMATION
    # --------------------------------------------------------

    for name in locations:

        if name in snapped:

            locations[name]["node"] = (
                snapped[name]["node"]
            )

    # --------------------------------------------------------
    # BUILD ROUTE NODES
    # --------------------------------------------------------

    route_nodes = get_route_nodes(
        G,
        location_names,
        routes,
        locations
    )

    # --------------------------------------------------------
    # VISUALIZE
    # --------------------------------------------------------

    visualize_routes(
        G,
        route_nodes,
        location_names,
        routes,
        locations,
        algorithm_name
    )

    print()

    print("=" * 60)

    print(
        "ROUTE VISUALIZATION COMPLETED"
    )

    print("=" * 60)