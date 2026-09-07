"""Multi-objective cost utilities for traffic-aware CVRP optimization."""

import numpy as np


def build_multi_objective_matrix(
    distance_matrix,
    traffic_matrix=None,
    distance_weight=0.45,
    time_weight=0.35,
    congestion_weight=0.20,
):
    """Build a normalized weighted cost matrix.

    Distance is the base road cost, time is represented by the traffic-adjusted
    cost, and congestion is the relative traffic penalty on each segment.
    """
    base = np.asarray(distance_matrix, dtype=float)
    traffic = base if traffic_matrix is None else np.asarray(traffic_matrix, dtype=float)
    if base.shape != traffic.shape or base.ndim != 2 or base.shape[0] != base.shape[1]:
        raise ValueError("distance and traffic matrices must be equally sized squares")
    if min(distance_weight, time_weight, congestion_weight) < 0:
        raise ValueError("objective weights must be non-negative")
    weight_total = distance_weight + time_weight + congestion_weight
    if weight_total <= 0:
        raise ValueError("at least one objective weight must be positive")

    valid = (base > 0) & np.isfinite(base) & np.isfinite(traffic)
    if not np.any(valid):
        return base.copy()

    distance_term = base / np.mean(base[valid])
    time_term = traffic / np.mean(traffic[valid])
    congestion_term = np.zeros_like(base)
    congestion_term[valid] = np.maximum(
        0.0, traffic[valid] / np.maximum(base[valid], 1e-12) - 1.0
    )
    congestion_mean = np.mean(congestion_term[valid])
    if congestion_mean > 0:
        congestion_term /= congestion_mean

    result = (
        distance_weight * distance_term
        + time_weight * time_term
        + congestion_weight * congestion_term
    ) / weight_total
    result[~valid] = base[~valid]
    np.fill_diagonal(result, 0.0)
    return result


def evaluate_route_objectives(routes, base_matrix, traffic_matrix, depot_index=0):
    """Return distance, traffic-time cost, and congestion exposure."""
    base = np.asarray(base_matrix, dtype=float)
    traffic = np.asarray(traffic_matrix, dtype=float)
    totals = {"distance": 0.0, "time": 0.0, "congestion": 0.0}
    for route in routes:
        previous = depot_index
        for customer in [*route, depot_index]:
            distance = base[previous, customer]
            adjusted = traffic[previous, customer]
            totals["distance"] += distance
            totals["time"] += adjusted
            if distance > 0:
                totals["congestion"] += max(0.0, adjusted / distance - 1.0)
            previous = customer
    return totals