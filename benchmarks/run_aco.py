from algorithms.aco import ACOVRP
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
# RUN ACO
# ============================================================

def run_aco():

    optimizer = ACOVRP(

        distance_matrix=distance_matrix,

        demands=demands,

        customer_indices=customer_indices,

        depot_index=0,

        vehicle_capacity=200,

        num_vehicles=2,

        num_ants=30,

        max_iterations=100,

        alpha=1.0,

        beta=2.0,

        evaporation=0.5,

        q=100.0,

        seed=42

    )

    return optimizer.optimize()


# ============================================================
# BENCHMARK
# ============================================================

engine = BenchmarkEngine()

result = engine.run_algorithm(
    "Ant Colony Optimization",
    run_aco
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 60)
print("ANT COLONY OPTIMIZATION RESULT")
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