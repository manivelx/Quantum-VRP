from algorithms.tabu_search import TabuSearchVRP
from benchmarks.benchmark import BenchmarkEngine

from benchmarks.run_qpso import prepare_problem


# ============================================================
# LOAD VRP PROBLEM
# ============================================================

(
    distance_matrix,
    demands,
    customer_indices,
    location_names
) = prepare_problem()


# ============================================================
# RUN TABU SEARCH
# ============================================================

def run_tabu():

    optimizer = TabuSearchVRP(

        distance_matrix=distance_matrix,

        demands=demands,

        customer_indices=customer_indices,

        depot_index=0,

        vehicle_capacity=200,

        num_vehicles=2,

        max_iterations=1000,

        tabu_tenure=10,

        neighborhood_size=50,

        seed=42

    )

    return optimizer.optimize()


# ============================================================
# BENCHMARK
# ============================================================

engine = BenchmarkEngine()

result = engine.run_algorithm(
    "Tabu Search",
    run_tabu
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 60)
print("TABU SEARCH RESULT")
print("=" * 60)

print(
    f"Distance : {result.distance:.3f} km"
)

print(
    f"Runtime   : {result.runtime:.4f} sec"
)

print(
    f"Vehicles  : {len(result.routes)}"
)

print(
    f"Iterations: {result.iterations}"
)

print()
print("Routes:")

for i, route in enumerate(
    result.routes
):

    names = [
        location_names[c]
        for c in route
    ]

    print(
        f"Vehicle {i + 1}: "
        + "DEPOT → "
        + " → ".join(names)
        + " → DEPOT"
    )

print("=" * 60)