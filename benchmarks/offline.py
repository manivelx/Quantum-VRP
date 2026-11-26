"""
Offline benchmark for the FluxRoute optimizers.

Runs QPSO, hybrid QPSO + GA, PSO and GA on the same capacitated VRP
instance without downloading a road network, so it finishes in seconds.

Distances are straight-line (haversine) distances scaled by a detour
factor. They are an approximation of road distance, not a replacement
for the OpenStreetMap graph used by the Streamlit app.

    python -m benchmarks.offline
    python -m benchmarks.offline --csv data/sample_customers.csv --runs 5
    python -m benchmarks.offline --generate 40 --seed 7
"""

import argparse
import math
import os
import time

import numpy as np
import pandas as pd

from algorithms.ga import GeneticAlgorithmVRP
from algorithms.hybrid_qpso_ga import HybridQPSOGA
from algorithms.pso import PSO_VRP
from algorithms.qpso import QPSO_VRP

EARTH_RADIUS_KM = 6371.0088
DETOUR_FACTOR = 1.3
DEFAULT_CSV = os.path.join("data", "sample_customers.csv")
REQUIRED_COLUMNS = ["id", "type", "latitude", "longitude", "demand"]


def haversine_matrix(latitudes, longitudes):
    """Pairwise great-circle distances in km."""
    lat = np.radians(np.asarray(latitudes, dtype=float))
    lon = np.radians(np.asarray(longitudes, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat)[:, None] * np.cos(lat)[None, :]
        * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_instance(csv_path, detour_factor=DETOUR_FACTOR):
    """
    Read a customer CSV with exactly one depot row.

    Returns (distance_matrix, demands, customer_indices, names) with the
    depot at index 0, which is what the optimizers expect.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["type"] = df["type"].astype(str).str.strip().str.lower()
    for column in ("latitude", "longitude", "demand"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["latitude", "longitude", "demand"]].isnull().any().any():
        raise ValueError("Invalid latitude, longitude or demand values.")

    depots = df[df["type"] == "depot"]
    if len(depots) != 1:
        raise ValueError(f"Expected exactly one depot, found {len(depots)}.")

    customers = df[df["type"] == "customer"]
    if customers.empty:
        raise ValueError("No customer rows found.")

    ordered = pd.concat([depots, customers], ignore_index=True)
    matrix = haversine_matrix(ordered["latitude"], ordered["longitude"])
    matrix = matrix * detour_factor
    demands = [float(x) for x in ordered["demand"]]
    return (
        matrix,
        demands,
        list(range(1, len(ordered))),
        list(ordered["id"]),
    )


def generate_csv(path, num_customers=30, seed=42):
    """Write a deterministic clustered instance around a Pune depot."""
    rng = np.random.default_rng(seed)
    depot_lat, depot_lon = 18.5362, 73.8958
    centers = rng.uniform(-0.04, 0.04, size=(4, 2))
    rows = [["DEPOT_01", "depot", depot_lat, depot_lon, 0]]
    for number in range(1, num_customers + 1):
        center = centers[rng.integers(len(centers))]
        lat = depot_lat + center[0] + rng.normal(0, 0.008)
        lon = depot_lon + center[1] + rng.normal(0, 0.008)
        demand = int(rng.integers(5, 26))
        rows.append(
            [f"LOC_{number:03d}", "customer",
             round(lat, 6), round(lon, 6), demand]
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    return path


def route_distance(matrix, depot, routes):
    total = 0.0
    for route in routes:
        previous = depot
        for customer in route:
            total += matrix[previous, customer]
            previous = customer
        if route:
            total += matrix[previous, depot]
    return total


def is_feasible(routes, demands, capacity, num_vehicles, customers):
    """Every customer exactly once, capacity and vehicle limits respected."""
    visited = [c for route in routes for c in route]
    if sorted(visited) != sorted(customers):
        return False
    if len(routes) > num_vehicles:
        return False
    return all(
        sum(demands[c] for c in route) <= capacity + 1e-9
        for route in routes
    )


def build_optimizers(matrix, demands, customers, capacity, vehicles,
                     iterations, particles, seed):
    common = dict(
        distance_matrix=matrix,
        demands=demands,
        customer_indices=customers,
        depot_index=0,
        vehicle_capacity=capacity,
        num_vehicles=vehicles,
        max_iterations=iterations,
        seed=seed,
    )
    return {
        "QPSO": QPSO_VRP(num_particles=particles, **common),
        "Hybrid QPSO + GA": HybridQPSOGA(
            num_particles=particles, population_size=particles, **common
        ),
        "PSO": PSO_VRP(num_particles=particles, **common),
        "Genetic Algorithm": GeneticAlgorithmVRP(
            population_size=particles, **common
        ),
    }


def run_benchmark(csv_path, vehicles, capacity, runs, iterations,
                  particles):
    matrix, demands, customers, _ = load_instance(csv_path)
    total_demand = sum(demands[c] for c in customers)
    if total_demand > vehicles * capacity:
        raise ValueError(
            f"Infeasible instance: total demand {total_demand:.0f} exceeds "
            f"fleet capacity {vehicles * capacity:.0f}."
        )

    rows = []
    names = list(build_optimizers(
        matrix, demands, customers, capacity, vehicles,
        iterations, particles, 0
    ))
    for name in names:
        raw, polished, times, feasible = [], [], [], True
        for run in range(runs):
            optimizer = build_optimizers(
                matrix, demands, customers, capacity, vehicles,
                iterations, particles, run
            )[name]
            started = time.perf_counter()
            routes = optimizer.optimize()[0]
            times.append(time.perf_counter() - started)
            feasible &= is_feasible(
                routes, demands, capacity, vehicles, customers
            )
            raw.append(route_distance(matrix, 0, routes))

            # Same 2-opt polish for every algorithm, so QPSO is not
            # credited for a local search the others do not get.
            helper = QPSO_VRP(
                matrix, demands, customers, 0, capacity, vehicles,
                num_particles=2, max_iterations=1, seed=run,
            )
            polished.append(route_distance(
                matrix, 0, [helper._two_opt_route(r) for r in routes]
            ))
        rows.append({
            "algorithm": name,
            "feasible": feasible,
            "mean_km_raw": float(np.mean(raw)),
            "best_km_raw": float(np.min(raw)),
            "std_km_raw": float(np.std(raw)),
            "mean_km_with_2opt": float(np.mean(polished)),
            "mean_seconds": float(np.mean(times)),
        })
    return pd.DataFrame(rows), len(customers)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--generate", type=int, metavar="N",
                        help="write a fresh N-customer CSV to --csv first")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for --generate")
    parser.add_argument("--vehicles", type=int, default=3)
    parser.add_argument("--capacity", type=float, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--out", default=os.path.join(
        "results", "offline_benchmark.csv"))
    args = parser.parse_args()

    if args.generate:
        generate_csv(args.csv, args.generate, args.seed)
        print(f"Wrote {args.generate} customers to {args.csv}")

    table, count = run_benchmark(
        args.csv, args.vehicles, args.capacity, args.runs,
        args.iterations, args.particles,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False)

    print(
        f"\n{count} customers, {args.vehicles} vehicles x "
        f"{args.capacity:g} capacity, {args.runs} runs, "
        f"{args.iterations} iterations\n"
    )
    print(table.round(3).to_string(index=False))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
