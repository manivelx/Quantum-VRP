import os
import time
import random

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox

from graph.osm_loader import load_city_graph
from graph.location_snapper import snap_locations


# ============================================================
# CONFIGURATION
# ============================================================

PLACE = "Koregaon Park, Pune, India"

CUSTOMER_FILE = "data/customers.csv"

NUM_VEHICLES = 2
VEHICLE_CAPACITY = 200

NUM_PARTICLES = 30
MAX_ITERATIONS = 100
BETA = 0.75

# Number of independent benchmark runs
NUM_RUNS = 10


# ============================================================
# QPSO ENGINE
# ============================================================

class QPSO_VRP:

    def __init__(
        self,
        distance_matrix,
        demands,
        customer_indices,
        depot_index,
        vehicle_capacity,
        num_vehicles,
        num_particles=30,
        max_iterations=100,
        beta=0.75,
        seed=42
    ):

        self.distance_matrix = np.asarray(
            distance_matrix,
            dtype=float
        )

        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index

        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.beta = beta

        random.seed(seed)
        np.random.seed(seed)

        self.dimension = len(customer_indices)

    # --------------------------------------------------------
    # Decode particle
    # --------------------------------------------------------

    def decode(self, position):

        order = np.argsort(position)

        ordered_customers = [
            self.customer_indices[i]
            for i in order
        ]

        routes = []

        current_route = []
        current_load = 0.0

        for customer in ordered_customers:

            demand = self.demands[customer]

            if (
                current_route
                and current_load + demand
                > self.vehicle_capacity
            ):

                routes.append(current_route)

                current_route = []
                current_load = 0.0

            current_route.append(customer)
            current_load += demand

        if current_route:
            routes.append(current_route)

        if len(routes) > self.num_vehicles:
            return routes, False

        return routes, True

    # --------------------------------------------------------
    # Route distance
    # --------------------------------------------------------

    def route_distance(self, route):

        if not route:
            return 0.0

        distance = 0.0

        previous = self.depot_index

        for customer in route:

            distance += self.distance_matrix[
                previous,
                customer
            ]

            previous = customer

        distance += self.distance_matrix[
            previous,
            self.depot_index
        ]

        return distance

    # --------------------------------------------------------
    # Fitness
    # --------------------------------------------------------

    def fitness(self, position):

        routes, feasible = self.decode(position)

        total_distance = sum(
            self.route_distance(route)
            for route in routes
        )

        penalty = 0.0

        if not feasible:

            excess_routes = max(
                0,
                len(routes) - self.num_vehicles
            )

            penalty += (
                excess_routes
                * self.vehicle_capacity
                * 1000
            )

        for route in routes:

            load = sum(
                self.demands[c]
                for c in route
            )

            if load > self.vehicle_capacity:

                penalty += (
                    load
                    - self.vehicle_capacity
                ) * 10000

        return total_distance + penalty

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    def optimize(self):

        if self.dimension == 0:
            return [], 0.0, 0

        particles = np.random.uniform(
            0,
            1,
            (
                self.num_particles,
                self.dimension
            )
        )

        personal_best = particles.copy()

        personal_best_fitness = np.array([
            self.fitness(p)
            for p in particles
        ])

        best_index = np.argmin(
            personal_best_fitness
        )

        global_best = personal_best[
            best_index
        ].copy()

        global_best_fitness = (
            personal_best_fitness[
                best_index
            ]
        )

        history = [
            global_best_fitness
        ]

        for iteration in range(
            self.max_iterations
        ):

            mbest = np.mean(
                personal_best,
                axis=0
            )

            for i in range(
                self.num_particles
            ):

                phi = np.random.uniform(
                    0,
                    1,
                    self.dimension
                )

                attractor = (
                    phi * personal_best[i]
                    + (1 - phi) * global_best
                )

                u = np.random.uniform(
                    0.000001,
                    1,
                    self.dimension
                )

                sign = np.where(
                    np.random.random(
                        self.dimension
                    ) < 0.5,
                    -1,
                    1
                )

                new_position = (
                    attractor
                    + sign
                    * self.beta
                    * np.abs(
                        mbest - particles[i]
                    )
                    * np.log(1 / u)
                )

                new_position = np.clip(
                    new_position,
                    0,
                    1
                )

                particles[i] = new_position

                current_fitness = self.fitness(
                    new_position
                )

                if (
                    current_fitness
                    < personal_best_fitness[i]
                ):

                    personal_best[i] = (
                        new_position.copy()
                    )

                    personal_best_fitness[i] = (
                        current_fitness
                    )

                if (
                    current_fitness
                    < global_best_fitness
                ):

                    global_best = (
                        new_position.copy()
                    )

                    global_best_fitness = (
                        current_fitness
                    )

            history.append(
                global_best_fitness
            )

        routes, feasible = self.decode(
            global_best
        )

        return (
            routes,
            global_best_fitness,
            len(history) - 1
        )


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

        try:

            lengths = nx.single_source_dijkstra_path_length(
                planning_graph,
                source,
                weight="length"
            )

        except Exception:

            lengths = {}

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
# LOAD CUSTOMER DATA
# ============================================================

def load_customers():

    if not os.path.exists(
        CUSTOMER_FILE
    ):

        raise FileNotFoundError(
            f"Customer file not found: "
            f"{CUSTOMER_FILE}"
        )

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
# PREPARE VRP DATA
# ============================================================

def prepare_problem():

    print()
    print("=" * 60)
    print("BLACK ROCK VRP - BENCHMARK")
    print("=" * 60)

    print()
    print("Loading customer data...")

    customers_df = load_customers()

    print(
        f"Customers found: "
        f"{len(customers_df)}"
    )

    # --------------------------------------------------------
    # Identify depot
    # --------------------------------------------------------

    depot_rows = customers_df[
        customers_df["type"]
        .str.lower()
        == "depot"
    ]

    if len(depot_rows) == 0:

        raise ValueError(
            "No depot found in customer data."
        )

    depot_id = depot_rows.iloc[0]["id"]

    # --------------------------------------------------------
    # Identify customers
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
    # Create locations
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
    # Load OSM network
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
    # Snap locations
    # --------------------------------------------------------

    print()
    print(
        "Snapping customer locations..."
    )

    snapped = snap_locations(
        G,
        locations
    )

    print(
        f"Locations snapped: "
        f"{len(snapped)}"
    )

    # --------------------------------------------------------
    # Location ordering
    # --------------------------------------------------------

    location_names = [
        depot_id
    ]

    for _, row in customer_rows.iterrows():

        location_names.append(
            row["id"]
        )

    location_nodes = []

    for name in location_names:

        if name not in snapped:

            raise ValueError(
                f"Location could not be snapped: "
                f"{name}"
            )

        location_nodes.append(
            snapped[name]["node"]
        )

    # --------------------------------------------------------
    # Demand dictionary
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
    # Capacity validation
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
        f"Total demand: "
        f"{total_demand:.2f} kg"
    )

    print(
        f"Total vehicle capacity: "
        f"{NUM_VEHICLES * VEHICLE_CAPACITY:.2f} kg"
    )

    if total_demand > (
        NUM_VEHICLES
        * VEHICLE_CAPACITY
    ):

        raise ValueError(
            "Problem is infeasible: "
            "total demand exceeds vehicle capacity."
        )

    # --------------------------------------------------------
    # Distance matrix
    # --------------------------------------------------------

    print()
    print(
        "Building road distance matrix..."
    )

    start = time.time()

    distance_matrix = build_distance_matrix(
        G,
        location_nodes
    )

    matrix_time = (
        time.time() - start
    )

    print(
        f"Distance matrix created in "
        f"{matrix_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Check connectivity
    # --------------------------------------------------------

    if np.any(
        distance_matrix >= 1e9
    ):

        raise ValueError(
            "Some locations are disconnected "
            "from the road network."
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
        customer_indices,
        location_names
    )


# ============================================================
# RUN ONE BENCHMARK
# ============================================================

def run_single_benchmark(
    distance_matrix,
    demands,
    customer_indices,
    seed
):

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

        seed=seed

    )

    start_time = time.time()

    routes, best_distance, iterations = (
        optimizer.optimize()
    )

    elapsed = (
        time.time()
        - start_time
    )

    total_load = sum(
        demands[c]
        for c in customer_indices
    )

    return {

        "seed":
            seed,

        "distance_km":
            best_distance,

        "vehicles_used":
            len(routes),

        "total_load_kg":
            total_load,

        "iterations":
            iterations,

        "execution_time_sec":
            elapsed,

        "feasible":
            len(routes) <= NUM_VEHICLES,

        "routes":
            routes
    }


# ============================================================
# BENCHMARK QPSO
# ============================================================

def run_benchmark():

    (
        distance_matrix,
        demands,
        customer_indices,
        location_names
    ) = prepare_problem()

    print()
    print("=" * 60)
    print("STARTING QPSO BENCHMARK")
    print("=" * 60)

    print()
    print(
        f"Runs        : {NUM_RUNS}"
    )

    print(
        f"Particles   : {NUM_PARTICLES}"
    )

    print(
        f"Iterations  : {MAX_ITERATIONS}"
    )

    print(
        f"Vehicles    : {NUM_VEHICLES}"
    )

    print(
        f"Capacity    : {VEHICLE_CAPACITY} kg"
    )

    results = []

    # --------------------------------------------------------
    # Multiple independent runs
    # --------------------------------------------------------

    for run in range(
        NUM_RUNS
    ):

        seed = 42 + run

        print()
        print(
            f"Run {run + 1}/{NUM_RUNS} "
            f"(seed={seed})"
        )

        result = run_single_benchmark(
            distance_matrix,
            demands,
            customer_indices,
            seed
        )

        results.append(
            result
        )

        print(
            f"Distance : "
            f"{result['distance_km']:.3f} km"
        )

        print(
            f"Vehicles : "
            f"{result['vehicles_used']}"
        )

        print(
            f"Time     : "
            f"{result['execution_time_sec']:.3f} sec"
        )

    # ========================================================
    # CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        [
            {
                "Run":
                    i + 1,

                "Seed":
                    r["seed"],

                "Distance (km)":
                    round(
                        r["distance_km"],
                        3
                    ),

                "Vehicles":
                    r["vehicles_used"],

                "Iterations":
                    r["iterations"],

                "Execution Time (sec)":
                    round(
                        r["execution_time_sec"],
                        3
                    ),

                "Feasible":
                    r["feasible"]
            }

            for i, r
            in enumerate(results)
        ]
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    distances = np.array(
        [
            r["distance_km"]
            for r in results
        ]
    )

    times = np.array(
        [
            r["execution_time_sec"]
            for r in results
        ]
    )

    best_index = np.argmin(
        distances
    )

    best_result = results[
        best_index
    ]

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print()

    print(
        f"Best Distance       : "
        f"{np.min(distances):.3f} km"
    )

    print(
        f"Worst Distance      : "
        f"{np.max(distances):.3f} km"
    )

    print(
        f"Average Distance    : "
        f"{np.mean(distances):.3f} km"
    )

    print(
        f"Std Deviation       : "
        f"{np.std(distances):.3f} km"
    )

    print(
        f"Average Runtime     : "
        f"{np.mean(times):.3f} sec"
    )

    print(
        f"Best Runtime        : "
        f"{np.min(times):.3f} sec"
    )

    print(
        f"Worst Runtime       : "
        f"{np.max(times):.3f} sec"
    )

    print(
        f"Feasible Runs       : "
        f"{sum(r['feasible'] for r in results)}"
        f"/{NUM_RUNS}"
    )

    # ========================================================
    # BEST ROUTE
    # ========================================================

    print()
    print("=" * 60)
    print("BEST ROUTE")
    print("=" * 60)

    for vehicle_number, route in enumerate(
        best_result["routes"]
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

        print()
        print(
            f"Vehicle {vehicle_number + 1}:"
        )

        print(
            f"  {route_text}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = (
        "benchmark_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Benchmark results saved to: "
        f"{output_file}"
    )

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_benchmark()

    except Exception as e:

        print()
        print("=" * 60)
        print("BENCHMARK FAILED")
        print("=" * 60)

        print()
        print(
            f"Error: {e}"
        )

        print()