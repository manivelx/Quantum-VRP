import sys
import os
import time

# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

from benchmarks.benchmark import BenchmarkEngine

from algorithms.qpso import QPSO_VRP

from graph.osm_loader import load_city_graph
from graph.location_snapper import snap_locations

import pandas as pd
import numpy as np
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

PLACE = "Koregaon Park, Pune, India"

CUSTOMER_FILE = "data/customers_stress_1000.csv"

NUM_VEHICLES = 3

VEHICLE_CAPACITY = 200

NUM_PARTICLES = 30

MAX_ITERATIONS = 100

BETA = 0.75


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
# LOAD CUSTOMERS
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
            "Invalid customer data."
        )

    return df


# ============================================================
# PREPARE VRP PROBLEM
# ============================================================

def prepare_problem():

    print()
    print("=" * 60)
    print("PREPARING VRP PROBLEM")
    print("=" * 60)

    # --------------------------------------------------------
    # Load customer data
    # --------------------------------------------------------

    print("\nLoading customers...")

    customers_df = load_customers()

    # --------------------------------------------------------
    # Find depot
    # --------------------------------------------------------

    depot_rows = customers_df[
        customers_df["type"].str.lower()
        == "depot"
    ]

    if len(depot_rows) == 0:

        raise ValueError(
            "No depot found."
        )

    depot_id = depot_rows.iloc[0]["id"]

    # --------------------------------------------------------
    # Find customers
    # --------------------------------------------------------

    customer_rows = customers_df[
        customers_df["type"].str.lower()
        != "depot"
    ]

    if len(customer_rows) == 0:

        raise ValueError(
            "No customers found."
        )

    # --------------------------------------------------------
    # Create locations dictionary
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
    # Load OSM graph
    # --------------------------------------------------------

    print(
        f"\nLoading road network: {PLACE}"
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
    # Snap locations
    # --------------------------------------------------------

    print(
        "\nSnapping locations..."
    )

    snapped = snap_locations(
        G,
        locations
    )

    print(
        f"Locations snapped: {len(snapped)}"
    )

    # --------------------------------------------------------
    # Location ordering
    #
    # Index 0 = depot
    # Index 1+ = customers
    # --------------------------------------------------------

    location_names = [
        depot_id
    ]

    for _, row in customer_rows.iterrows():

        location_names.append(
            row["id"]
        )

    # --------------------------------------------------------
    # Convert IDs → road nodes
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
    # Create demand dictionary
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
    # Capacity check
    # --------------------------------------------------------

    total_demand = sum(
        demands[i]
        for i in range(
            1,
            len(location_names)
        )
    )

    print(
        f"\nTotal demand: "
        f"{total_demand:.2f} kg"
    )

    print(
        f"Vehicle capacity: "
        f"{NUM_VEHICLES * VEHICLE_CAPACITY:.2f} kg"
    )

    if total_demand > (
        NUM_VEHICLES * VEHICLE_CAPACITY
    ):

        raise ValueError(
            "Total demand exceeds vehicle capacity."
        )

    # --------------------------------------------------------
    # BUILD DISTANCE MATRIX
    # --------------------------------------------------------

    print(
        "\nBuilding distance matrix..."
    )

    start = time.time()

    distance_matrix = build_distance_matrix(
        G,
        location_nodes
    )

    matrix_time = time.time() - start

    print(
        f"Distance matrix created in "
        f"{matrix_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Connectivity check
    # --------------------------------------------------------

    if np.any(
        distance_matrix >= 1e9
    ):

        raise ValueError(
            "Some locations are disconnected."
        )

    # --------------------------------------------------------
    # Customer indices
    # --------------------------------------------------------

    customer_indices = list(
        range(
            1,
            len(location_names)
        )
    )

    return (
        distance_matrix,
        demands,
        customer_indices,
        location_names
    )


# ============================================================
# RUN QPSO
# ============================================================

def run_qpso():

    # --------------------------------------------------------
    # THIS IS THE IMPORTANT FIX
    #
    # distance_matrix is created HERE before QPSO uses it.
    # --------------------------------------------------------

    (
        distance_matrix,
        demands,
        customer_indices,
        location_names
    ) = prepare_problem()

    # --------------------------------------------------------
    # Create QPSO optimizer
    # --------------------------------------------------------

    optimizer = QPSO_VRP(

        distance_matrix=distance_matrix,

        demands=demands,

        customer_indices=customer_indices,

        depot_index=0,

        vehicle_capacity=VEHICLE_CAPACITY,

        num_vehicles=NUM_VEHICLES,

        num_particles=NUM_PARTICLES,

        max_iterations=MAX_ITERATIONS,

        beta=BETA,

        seed=42

    )

    # --------------------------------------------------------
    # Run optimization
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RUNNING QPSO")
    print("=" * 60)

    start = time.time()

    routes, distance, iterations = (
        optimizer.optimize()
    )

    runtime = time.time() - start

    # --------------------------------------------------------
    # Display route
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("QPSO RESULT")
    print("=" * 60)

    print(
        f"\nTotal Distance: "
        f"{distance:.3f} km"
    )

    print(
        f"Vehicles Used: "
        f"{len(routes)}"
    )

    print(
        f"Iterations: "
        f"{iterations}"
    )

    print(
        f"Runtime: "
        f"{runtime:.3f} seconds"
    )

    print("\nRoutes:")

    for vehicle_number, route in enumerate(
        routes
    ):

        route_names = [
            location_names[c]
            for c in route
        ]

        route_text = (
            "DEPOT → "
            + " → ".join(route_names)
            + " → DEPOT"
        )

        print(
            f"Vehicle {vehicle_number + 1}: "
            f"{route_text}"
        )

    # --------------------------------------------------------
    # RETURN RESULT TO BENCHMARK ENGINE
    # --------------------------------------------------------

    return (
        routes,
        distance,
        iterations
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("FLUXROUTE")
    print("QPSO BENCHMARK")
    print("=" * 60)

    engine = BenchmarkEngine()

    result = engine.run_algorithm(
        "QPSO",
        run_qpso
    )

    print()
    print("=" * 60)
    print("BENCHMARK RESULT")
    print("=" * 60)

    print()

    print(
        f"Algorithm       : "
        f"{result.algorithm_name}"
    )

    print(
        f"Distance        : "
        f"{result.distance:.3f} km"
    )

    print(
        f"Runtime         : "
        f"{result.runtime:.3f} seconds"
    )

    print(
        f"Iterations      : "
        f"{result.iterations}"
    )

    print(
        f"Vehicles Used   : "
        f"{len(result.routes)}"
    )

    print()
    print("Benchmark completed successfully.")