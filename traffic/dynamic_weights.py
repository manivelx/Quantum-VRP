"""
Dynamic Edge Weight System for Real-Time Route Optimization.

Takes a base distance matrix and traffic conditions,
returns a traffic-adjusted distance matrix.
Supports re-optimization when traffic changes.
"""

import numpy as np
from traffic.traffic_simulator import TrafficSimulator


class DynamicWeights:
    """
    Manages dynamic edge weights for the VRP based on
    real-time traffic conditions.

    Usage:
        dw = DynamicWeights(base_distance_matrix)
        adjusted = dw.get_adjusted_matrix(hour=8, weather="rain")
        comparison = dw.compare_static_vs_dynamic(hour=17)
    """

    def __init__(self, base_distance_matrix, seed=None):
        self.base_matrix = np.asarray(
            base_distance_matrix, dtype=float
        )
        if self.base_matrix.ndim != 2 or self.base_matrix.shape[0] != self.base_matrix.shape[1]:
            raise ValueError("base_distance_matrix must be square")
        if not np.all(np.isfinite(self.base_matrix)) or np.any(self.base_matrix < 0):
            raise ValueError("base_distance_matrix must contain finite non-negative values")
        self.simulator = TrafficSimulator(seed=seed)
        self.n = self.base_matrix.shape[0]

    # ========================================================
    # GET TRAFFIC-ADJUSTED MATRIX
    # ========================================================

    def get_adjusted_matrix(self, hour=None, weather="clear",
                            congestion_probability=0.15):
        """
        Returns a traffic-adjusted distance matrix.

        Args:
            hour: time of day (0-23)
            weather: "clear", "rain", "heavy_rain", "fog"
            congestion_probability: probability of congestion per segment

        Returns:
            (adjusted_matrix, traffic_info)
        """
        return self.simulator.apply_traffic(
            self.base_matrix,
            hour=hour,
            congestion_probability=congestion_probability,
            weather=weather
        )

    # ========================================================
    # COMPARE STATIC VS DYNAMIC
    # ========================================================

    def compare_static_vs_dynamic(self, hour=None, weather="clear",
                                   congestion_probability=0.15):
        """
        Compare base (static) distances with traffic-adjusted distances.

        Returns:
            dict with comparison statistics
        """
        adjusted, traffic_info = self.get_adjusted_matrix(
            hour=hour,
            weather=weather,
            congestion_probability=congestion_probability
        )

        # Mask out diagonal and infinity values
        mask = (self.base_matrix > 0) & (self.base_matrix < 1e9)

        if not np.any(mask):
            return {
                "avg_increase_pct": 0.0,
                "max_increase_pct": 0.0,
                "traffic_info": traffic_info,
            }

        base_vals = self.base_matrix[mask]
        adjusted_vals = adjusted[mask]

        increases = (adjusted_vals - base_vals) / base_vals * 100

        return {
            "avg_increase_pct": round(float(np.mean(increases)), 2),
            "max_increase_pct": round(float(np.max(increases)), 2),
            "min_increase_pct": round(float(np.min(increases)), 2),
            "median_increase_pct": round(float(np.median(increases)), 2),
            "traffic_info": traffic_info,
        }

    # ========================================================
    # ROUTE DISTANCE COMPARISON
    # ========================================================

    def route_distance_comparison(self, routes, location_indices=None,
                                  hour=None, weather="clear",
                                  congestion_probability=0.15):
        """
        Compare total route distance under static vs dynamic conditions.

        Args:
            routes: list of vehicle routes [[c1, c2], [c3, c4]]
            location_indices: maps route indices to matrix indices

        Returns:
            dict with static_distance and dynamic_distance
        """
        adjusted, traffic_info = self.get_adjusted_matrix(
            hour=hour,
            weather=weather,
            congestion_probability=congestion_probability
        )

        if location_indices is None:
            location_indices = list(range(self.n))

        if len(location_indices) != self.n:
            raise ValueError("location_indices must map every matrix location")

        def calc_distance(matrix, routes):
            total = 0.0
            for route in routes:
                if not route:
                    continue
                prev = location_indices[0]
                for c in route:
                    if not isinstance(c, (int, np.integer)) or not 0 <= c < len(location_indices):
                        raise ValueError(
                            "route customer IDs must be integer indices into location_indices"
                        )
                    node = location_indices[c]
                    total += matrix[prev, node]
                    prev = node
                total += matrix[prev, location_indices[0]]
            return total

        static_dist = calc_distance(self.base_matrix, routes)
        dynamic_dist = calc_distance(adjusted, routes)

        return {
            "static_distance": round(static_dist, 3),
            "dynamic_distance": round(dynamic_dist, 3),
            "increase_km": round(dynamic_dist - static_dist, 3),
            "increase_pct": round(
                (dynamic_dist - static_dist) / static_dist * 100
                if static_dist > 0 else 0, 2
            ),
            "traffic_info": traffic_info,
        }
