import numpy as np
import pytest

from algorithms.ga import GeneticAlgorithmVRP
from algorithms.hybrid_qpso_ga import HybridQPSOGA
from algorithms.pso import PSO_VRP
from algorithms.qpso import QPSO_VRP
from benchmarks.offline import (
    generate_csv,
    haversine_matrix,
    is_feasible,
    load_instance,
    route_distance,
)


def make_instance(num_customers=20, seed=3, num_vehicles=3):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 10, size=(num_customers + 1, 2))
    points[0] = (5, 5)
    matrix = np.linalg.norm(points[:, None] - points[None], axis=2)
    demands = [0.0] + [float(x) for x in rng.integers(5, 20, num_customers)]
    capacity = float(sum(demands) / num_vehicles * 1.3)
    customers = list(range(1, num_customers + 1))
    return matrix, demands, customers, capacity, num_vehicles


def common(instance, **extra):
    matrix, demands, customers, capacity, vehicles = instance
    return dict(
        distance_matrix=matrix,
        demands=demands,
        customer_indices=customers,
        depot_index=0,
        vehicle_capacity=capacity,
        num_vehicles=vehicles,
        max_iterations=30,
        seed=1,
        **extra,
    )


ALGORITHMS = [
    (QPSO_VRP, dict(num_particles=15)),
    (PSO_VRP, dict(num_particles=15)),
    (GeneticAlgorithmVRP, dict(population_size=15)),
    (HybridQPSOGA, dict(num_particles=15, population_size=15)),
]


@pytest.mark.parametrize("cls,extra", ALGORITHMS)
def test_solution_is_feasible_and_visits_every_customer_once(cls, extra):
    instance = make_instance()
    routes, distance, *_ = cls(**common(instance, **extra)).optimize()
    matrix, demands, customers, capacity, vehicles = instance
    assert is_feasible(routes, demands, capacity, vehicles, customers)
    assert distance == pytest.approx(route_distance(matrix, 0, routes))


@pytest.mark.parametrize("cls,extra", ALGORITHMS)
def test_same_seed_gives_same_result(cls, extra):
    instance = make_instance()
    first = cls(**common(instance, **extra)).optimize()[0]
    second = cls(**common(instance, **extra)).optimize()[0]
    assert first == second


def test_two_opt_never_worsens_and_keeps_customers():
    instance = make_instance(num_customers=25)
    optimizer = QPSO_VRP(**common(instance, num_particles=5))
    rng = np.random.default_rng(0)
    route = list(rng.permutation(np.arange(1, 26)))
    improved = optimizer._two_opt_route(route)
    assert sorted(improved) == sorted(route)
    assert optimizer.route_distance(improved) <= optimizer.route_distance(route)


def test_fast_two_opt_matches_exact_version():
    instance = make_instance(num_customers=15)
    optimizer = QPSO_VRP(**common(instance, num_particles=5))
    rng = np.random.default_rng(5)
    for _ in range(10):
        route = list(rng.permutation(np.arange(1, 16)))
        fast = optimizer._two_opt_route(route)
        exact = optimizer._two_opt_route_exact(route)
        assert optimizer.route_distance(fast) == pytest.approx(
            optimizer.route_distance(exact)
        )


def test_asymmetric_matrix_uses_exact_fallback():
    matrix, demands, customers, capacity, vehicles = make_instance(10)
    matrix = matrix.copy()
    matrix[1, 2] += 5
    optimizer = QPSO_VRP(
        matrix, demands, customers, 0, capacity, vehicles,
        num_particles=5, max_iterations=5, seed=1,
    )
    route = [3, 1, 2, 5, 4]
    result = optimizer._two_opt_route(route)
    assert sorted(result) == sorted(route)


def test_decode_respects_capacity():
    matrix, demands, customers, capacity, vehicles = make_instance()
    optimizer = QPSO_VRP(**common(make_instance(), num_particles=5))
    position = np.random.default_rng(2).uniform(0, 1, len(customers))
    routes, _ = optimizer.decode(position)
    for route in routes:
        assert sum(demands[c] for c in route) <= capacity + 1e-9


def test_haversine_known_distance():
    # Pune to Mumbai is roughly 120 km in a straight line.
    matrix = haversine_matrix([18.5204, 19.0760], [73.8567, 72.8777])
    assert 115 < matrix[0, 1] < 125
    assert matrix[0, 0] == 0
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])


def test_generated_csv_loads_with_depot_first(tmp_path):
    path = tmp_path / "customers.csv"
    generate_csv(str(path), num_customers=12, seed=9)
    matrix, demands, customers, names = load_instance(str(path))
    assert matrix.shape == (13, 13)
    assert names[0] == "DEPOT_01"
    assert demands[0] == 0
    assert customers == list(range(1, 13))


def test_load_instance_rejects_missing_depot(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "id,type,latitude,longitude,demand\nA,customer,18.5,73.9,10\n"
    )
    with pytest.raises(ValueError, match="depot"):
        load_instance(str(path))
