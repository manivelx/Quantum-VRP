import sys
import os

# Add project root to Python path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np

from algorithms.pso import PSO_VRP
from benchmarks.benchmark import BenchmarkEngine

from graph.osm_loader import load_city_graph
from graph.location_snapper import snap_locations

import pandas as pd
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

PLACE = "Koregaon Park, Pune, India"

CUSTOMER_FILE = "data/customers.csv"

NUM_VEHICLES = 2
VEHICLE_CAPACITY = 200

NUM_PARTICLES = 30
MAX_ITERATIONS = 100

INERTIA = 0.7
COGNITIVE = 1.5
SOCIAL = 1.5

SEED = 42


# ============================================================
# LOAD CUSTOMER DATA
# ============================================================

def load_customers():

    df = pd.read_csv(
        CUSTOMER_FILE
    )

    required_columns = [
        "id",
        "type",
        "latitude",
        "longitude",
        "demand"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["demand"] = pd.to_numeric(
        df["demand"],
        errors="coerce"
    )

    if df[
        [
            "latitude",
            "longitude",
            "demand"
        ]
    ].isnull().any().any():

        raise ValueError(
            "Invalid latitude, longitude, "
            "or demand values."
        )

    return df


# ============================================================
# CREATE PLANNING GRAPH
# ============================================================

def create_planning_graph(G):

    if G.is_directed():

        return G.to_undirected(
            reciprocal=False
        )

    return G.copy()


# ============================================================
# BUILD DISTANCE MATRIX
# ============================================================

def build_distance_matrix(
    G,
    location_nodes
):

    planning_graph = create_planning_graph(G)

    n = len(location_nodes)

    matrix = np.zeros(
        (n, n),
        dtype=float
    )

    for i in range(n):

        source = location_nodes[i]

        lengths = nx.single_source_dijkstra_path_length(
            planning_graph,
            source,
            weight="length"
        )

        for j in range(n):

            target = location_nodes[j]

            if target in lengths:

                matrix[i][j] = (
                    lengths[target] / 1000.0
                )

            else:

                matrix[i][j] = 1e9

    return matrix


# ============================================================
# PREPARE VRP PROBLEM
# ============================================================

def prepare_problem():

    print()
    print("=" * 60)
    print("FLUXROUTE - PSO BENCHMARK")
    print("=" * 60)

    print()
    print("Loading customer data...")

    customers_df = load_customers()

    # --------------------------------------------------------
    # DEPOT
    # --------------------------------------------------------

    depot_rows = customers_df[
        customers_df["type"]
        .str.lower()
        == "depot"
    ]

    if len(depot_rows) == 0:

        raise ValueError(
            "No depot found."
        )

    depot_id = depot_rows.iloc[0]["id"]

    # --------------------------------------------------------
    # CUSTOMERS
    # --------------------------------------------------------

    customer_rows = customers_df[
        customers_df["type"]
        .str.lower()
        != "depot"
    ]

    if len(customer_rows) == 0:

        raise ValueError(
            "No customers found."
        )

    # --------------------------------------------------------
    # LOCATIONS
    # --------------------------------------------------------

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
    # LOAD ROAD NETWORK
    # --------------------------------------------------------

    print()
    print(
        f"Loading road network: {PLACE}"
    )

    G = load_city_graph(
        PLACE
    )

    print(
        f"Road nodes: {len(G.nodes):,}"
    )

    print(
        f"Road edges: {len(G.edges):,}"
    )

    # --------------------------------------------------------
    # SNAP LOCATIONS
    # --------------------------------------------------------

    print()
    print(
        "Snapping locations..."
    )

    snapped = snap_locations(
        G,
        locations
    )

    print(
        f"Locations snapped: {len(snapped)}"
    )

    # --------------------------------------------------------
    # LOCATION ORDER
    # --------------------------------------------------------

    location_names = [
        depot_id
    ]

    for _, row in customer_rows.iterrows():

        location_names.append(
            row["id"]
        )

    # --------------------------------------------------------
    # ROAD NODES
    # --------------------------------------------------------

    location_nodes = []

    for name in location_names:

        if name not in snapped:

            raise ValueError(
                f"Location could not be snapped: {name}"
            )

        location_nodes.append(
            snapped[name]["node"]
        )

    # --------------------------------------------------------
    # DEMANDS
    # --------------------------------------------------------

    demands = {}

    for index, name in enumerate(
        location_names
    ):

        if index == 0:

            demands[index] = 0.0

        else:

            demands[index] = float(
                snapped[name]["demand"]
            )

    # --------------------------------------------------------
    # CAPACITY CHECK
    # --------------------------------------------------------

    total_demand = sum(
        demands[i]
        for i in range(
            1,
            len(location_names)
        )
    )

    print()
    print(
        f"Total demand: {total_demand:.2f} kg"
    )

    print(
        f"Vehicle capacity: "
        f"{NUM_VEHICLES * VEHICLE_CAPACITY:.2f} kg"
    )

    if total_demand > (
        NUM_VEHICLES
        * VEHICLE_CAPACITY
    ):

        raise ValueError(
            "Problem is infeasible: "
            "total demand exceeds capacity."
        )

    # --------------------------------------------------------
    # DISTANCE MATRIX
    # --------------------------------------------------------

    print()
    print(
        "Building distance matrix..."
    )

    distance_matrix = build_distance_matrix(
        G,
        location_nodes
    )

    if np.any(
        distance_matrix >= 1e9
    ):

        raise ValueError(
            "Some locations are disconnected."
        )

    customer_indices = list(
        range(
            1,
            len(location_names)
        )
    )

    return (
        distance_matrix,
        demands,
        customer_indices
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # PREPARE PROBLEM
    # --------------------------------------------------------

    (
        distance_matrix,
        demands,
        customer_indices
    ) = prepare_problem()

    # --------------------------------------------------------
    # CREATE BENCHMARK ENGINE
    # --------------------------------------------------------

    engine = BenchmarkEngine()

    # --------------------------------------------------------
    # FUNCTION TO RUN PSO
    # --------------------------------------------------------

    def run_pso():

        optimizer = PSO_VRP(

            distance_matrix=distance_matrix,

            demands=demands,

            customer_indices=customer_indices,

            depot_index=0,

            vehicle_capacity=VEHICLE_CAPACITY,

            num_vehicles=NUM_VEHICLES,

            num_particles=NUM_PARTICLES,

            max_iterations=MAX_ITERATIONS,

            inertia=INERTIA,

            cognitive=COGNITIVE,

            social=SOCIAL,

            seed=SEED

        )

        return optimizer.optimize()

    # --------------------------------------------------------
    # RUN BENCHMARK
    # --------------------------------------------------------

    print()
    print(
        "Running PSO..."
    )

    result = engine.run_algorithm(
        "PSO",
        run_pso
    )

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PSO BENCHMARK RESULT")
    print("=" * 60)

    print()

    print(
        f"Algorithm : {result.algorithm_name}"
    )

    print(
        f"Distance  : "
        f"{result.distance:.3f} km"
    )

    print(
        f"Runtime   : "
        f"{result.runtime:.4f} seconds"
    )

    print(
        f"Iterations: "
        f"{result.iterations}"
    )

    print(
        f"Vehicles  : "
        f"{len(result.routes)}"
    )

    # --------------------------------------------------------
    # ROUTES
    # --------------------------------------------------------

    print()
    print(
        "Routes:"
    )

    for vehicle_number, route in enumerate(
        result.routes
    ):

        route_text = (
            "DEPOT → "
            + " → ".join(
                str(customer)
                for customer in route
            )
            + " → DEPOT"
        )

        print(
            f"Vehicle {vehicle_number + 1}: "
            f"{route_text}"
        )

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    print()
    print(
        "Benchmark Table:"
    )

    print(
        engine.results_dataframe()
        .to_string(index=False)
    )