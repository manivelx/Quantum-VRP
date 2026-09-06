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
from algorithms.pso import PSO_VRP
from algorithms.ga import GeneticAlgorithmVRP
from algorithms.aco import ACOVRP
from algorithms.simulated_annealing import SimulatedAnnealingVRP
from algorithms.tabu_search import TabuSearchVRP
from algorithms.alns import ALNS_VRP

from benchmarks.benchmark import BenchmarkEngine

from benchmarks.run_qpso import prepare_problem


# ============================================================
# CONFIGURATION
# ============================================================

NUM_VEHICLES = 2

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
    "BLACK ROCK VRP - ALGORITHM COMPARISON"
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


def run_aco():
    optimizer = ACOVRP(
        distance_matrix=distance_matrix,
        demands=demands,
        customer_indices=customer_indices,
        depot_index=0,
        vehicle_capacity=VEHICLE_CAPACITY,
        num_vehicles=NUM_VEHICLES,
        num_ants=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        seed=SEED
    )
    return optimizer.optimize()


def run_sa():
    optimizer = SimulatedAnnealingVRP(
        distance_matrix=distance_matrix,
        demands=demands,
        customer_indices=customer_indices,
        depot_index=0,
        vehicle_capacity=VEHICLE_CAPACITY,
        num_vehicles=NUM_VEHICLES,
        max_iterations=MAX_ITERATIONS,
        seed=SEED
    )
    return optimizer.optimize()


def run_tabu():
    optimizer = TabuSearchVRP(
        distance_matrix=distance_matrix,
        demands=demands,
        customer_indices=customer_indices,
        depot_index=0,
        vehicle_capacity=VEHICLE_CAPACITY,
        num_vehicles=NUM_VEHICLES,
        max_iterations=MAX_ITERATIONS,
        seed=SEED
    )
    return optimizer.optimize()


def run_alns():
    optimizer = ALNS_VRP(
        distance_matrix=distance_matrix,
        demands=demands,
        customer_indices=customer_indices,
        depot_index=0,
        vehicle_capacity=VEHICLE_CAPACITY,
        num_vehicles=NUM_VEHICLES,
        max_iterations=MAX_ITERATIONS,
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

print()
print("Running Ant Colony Optimization...")
engine.run_algorithm("Ant Colony Optimization", run_aco)

print()
print("Running Simulated Annealing...")
engine.run_algorithm("Simulated Annealing", run_sa)

print()
print("Running Tabu Search...")
engine.run_algorithm("Tabu Search", run_tabu)

print()
print("Running ALNS...")
engine.run_algorithm("ALNS", run_alns)


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