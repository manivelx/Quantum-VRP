import sys
import os
import pandas as pd
import pickle


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
# IMPORT ALGORITHMS
# ============================================================

from algorithms.qpso import QPSO_VRP
from algorithms.hybrid_qpso_ga import HybridQPSOGA
from algorithms.pso import PSO_VRP
from algorithms.ga import GeneticAlgorithmVRP

from benchmarks.benchmark import BenchmarkEngine

from benchmarks.run_qpso import prepare_problem


# ============================================================
# CONFIGURATION
# ============================================================

NUM_VEHICLES = 3

VEHICLE_CAPACITY = 200

NUM_PARTICLES = 30

POPULATION_SIZE = 30

MAX_ITERATIONS = 100

BETA = 0.75

INERTIA = 0.7

COGNITIVE = 1.5

SOCIAL = 1.5

MUTATION_RATE = 0.1

SEED = 42


# ============================================================
# LOAD VRP PROBLEM
# ============================================================

print()

print("=" * 60)

print(
    "FLUXROUTE - ALGORITHM COMPARISON"
)

print("=" * 60)

print()

print(
    "Preparing VRP problem..."
)


(
    distance_matrix,
    demands,
    customer_indices,
    location_names
) = prepare_problem()


# ============================================================
# QPSO
# ============================================================

def run_qpso():

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

        seed=SEED

    )

    return optimizer.optimize()


def run_hybrid_qpso_ga():
    optimizer = HybridQPSOGA(
        distance_matrix=distance_matrix,
        demands=demands,
        customer_indices=customer_indices,
        depot_index=0,
        vehicle_capacity=VEHICLE_CAPACITY,
        num_vehicles=NUM_VEHICLES,
        num_particles=NUM_PARTICLES,
        population_size=POPULATION_SIZE,
        max_iterations=MAX_ITERATIONS,
        beta=BETA,
        mutation_rate=MUTATION_RATE,
        seed=SEED
    )
    return optimizer.optimize()


# ============================================================
# PSO
# ============================================================

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


# ============================================================
# GENETIC ALGORITHM
# ============================================================

def run_ga():

    optimizer = GeneticAlgorithmVRP(

        distance_matrix=distance_matrix,

        demands=demands,

        customer_indices=customer_indices,

        depot_index=0,

        vehicle_capacity=VEHICLE_CAPACITY,

        num_vehicles=NUM_VEHICLES,

        population_size=POPULATION_SIZE,

        max_iterations=MAX_ITERATIONS,

        mutation_rate=MUTATION_RATE,

        seed=SEED

    )

    return optimizer.optimize()


# ============================================================
# CREATE BENCHMARK ENGINE
# ============================================================

engine = BenchmarkEngine()


# ============================================================
# RUN QPSO
# ============================================================

print()

print(
    "Running QPSO..."
)

qpso_result = engine.run_algorithm(
    "QPSO",
    run_qpso
)

print()
print("Running Hybrid QPSO + GA...")
engine.run_algorithm("Hybrid QPSO + GA", run_hybrid_qpso_ga)


# ============================================================
# RUN PSO
# ============================================================

print()

print(
    "Running PSO..."
)

pso_result = engine.run_algorithm(
    "PSO",
    run_pso
)


# ============================================================
# RUN GENETIC ALGORITHM
# ============================================================

print()

print(
    "Running Genetic Algorithm..."
)

ga_result = engine.run_algorithm(
    "Genetic Algorithm",
    run_ga
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print()

print("=" * 60)

print(
    "ALGORITHM COMPARISON"
)

print("=" * 60)

results_df = engine.results_dataframe()

results_df["Distance Rank"] = results_df[
    "Total Distance (km)"
].rank(method="min", ascending=True).astype(int)
results_df["Runtime Rank"] = results_df[
    "Runtime (seconds)"
].rank(method="min", ascending=True).astype(int)
results_df["Distance/Runtime"] = (
    results_df["Total Distance (km)"]
    / results_df["Runtime (seconds)"].clip(lower=0.000001)
).round(2)

print()

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST ALGORITHM
# ============================================================

best = engine.best_result()
fastest = min(
    engine.get_results(),
    key=lambda result: result.runtime
)


print()

print("=" * 60)

print(
    "BEST ALGORITHM"
)

print("=" * 60)

print()


print(
    f"Algorithm : "
    f"{best.algorithm_name}"
)


print(
    f"Distance  : "
    f"{best.distance:.3f} km"
)


print(
    f"Runtime   : "
    f"{best.runtime:.4f} seconds"
)


print(
    f"Iterations: "
    f"{best.iterations}"
)


print(
    f"Vehicles  : "
    f"{len(best.routes)}"
)

print()
print("FASTEST ALGORITHM")
print("=" * 60)
print(f"Algorithm : {fastest.algorithm_name}")
print(f"Runtime   : {fastest.runtime:.4f} seconds")
print(f"Distance  : {fastest.distance:.3f} km")


# ============================================================
# BEST ROUTE
# ============================================================

print()

print("=" * 60)

print(
    "BEST ROUTE"
)

print("=" * 60)


for vehicle_number, route in enumerate(
    best.routes
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

        f"Vehicle {vehicle_number + 1}: "

        f"{route_text}"

    )


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

output_file = os.path.join(

    PROJECT_ROOT,

    "benchmark_comparison.csv"

)


results_df.to_csv(

    output_file,

    index=False

)


print()

print(

    f"Comparison saved to: "

    f"{output_file}"

)


# ============================================================
# SAVE BEST ROUTE
# ============================================================

best_route_data = {

    "algorithm": best.algorithm_name,

    "distance": float(
        best.distance
    ),

    "runtime": float(
        best.runtime
    ),

    "iterations": int(
        best.iterations
    ),

    "vehicles": len(
        best.routes
    ),

    "routes": best.routes,

    "location_names": location_names

}


best_route_file = os.path.join(

    PROJECT_ROOT,

    "best_route.pkl"

)


with open(

    best_route_file,

    "wb"

) as file:

    pickle.dump(

        best_route_data,

        file

    )


print()

print(

    f"Best route saved to: "

    f"{best_route_file}"

)


# ============================================================
# SAVE BEST ROUTE AS CSV
# ============================================================

best_route_rows = []


for vehicle_number, route in enumerate(

    best.routes

):

    route_names = [

        location_names[c]

        for c in route

    ]


    for stop_number, customer in enumerate(

        route,

        start=1

    ):

        best_route_rows.append({

            "Vehicle":

                vehicle_number + 1,

            "Stop":

                stop_number,

            "Location":

                location_names[customer],

            "Customer Index":

                customer,

            "Demand (kg)":

                demands[customer]

        })


best_route_df = pd.DataFrame(

    best_route_rows

)


best_route_csv = os.path.join(

    PROJECT_ROOT,

    "best_route.csv"

)


best_route_df.to_csv(

    best_route_csv,

    index=False

)


print()

print(

    f"Best route CSV saved to: "

    f"{best_route_csv}"

)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print("=" * 60)

print(
    "COMPARISON COMPLETED"
)

print("=" * 60)

print()

print(
    "Files generated:"
)

print(
    "1. benchmark_comparison.csv"
)

print(
    "2. best_route.pkl"
)

print(
    "3. best_route.csv"
)

print()

print(
    f"Best Algorithm : "
    f"{best.algorithm_name}"
)

print(
    f"Best Distance  : "
    f"{best.distance:.3f} km"
)

print(
    f"Vehicles Used  : "
    f"{len(best.routes)}"
)

print()