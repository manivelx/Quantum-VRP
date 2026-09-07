"""Deterministic simulated incident and re-routing helpers."""

import numpy as np


def simulate_road_closure(distance_matrix, from_index, to_index, penalty=1e9):
    """Return a copy of the matrix with one directed road segment closed."""
    matrix = np.asarray(distance_matrix, dtype=float).copy()
    if not (0 <= from_index < matrix.shape[0] and 0 <= to_index < matrix.shape[1]):
        raise ValueError("closure indices must be valid matrix positions")
    if from_index == to_index:
        raise ValueError("a road closure must affect two different locations")
    matrix[from_index, to_index] = penalty
    return matrix


def route_uses_segment(routes, from_index, to_index):
    """Check whether any route traverses a directed location segment."""
    for route in routes:
        sequence = [0, *route, 0]
        if any(
            sequence[index] == from_index and sequence[index + 1] == to_index
            for index in range(len(sequence) - 1)
        ):
            return True
    return False