"""
Traffic Simulator for Dynamic Route Optimization.

Simulates real-world traffic conditions:
    - Time-of-day traffic multipliers (rush hour peaks)
    - Random congestion events on road segments
    - Weather impact multipliers
"""

import numpy as np
import random
from datetime import datetime


class TrafficSimulator:
    """
    Generates realistic traffic conditions that modify
    edge weights in the road network distance matrix.
    """

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ========================================================
    # TIME-OF-DAY TRAFFIC PROFILES
    # ========================================================

    # Hourly traffic multipliers (0-23 hours)
    # 1.0 = normal traffic, >1.0 = congested, <1.0 = free-flow
    HOURLY_MULTIPLIERS = {
        0: 0.7,   1: 0.6,   2: 0.6,   3: 0.6,
        4: 0.7,   5: 0.8,   6: 1.0,   7: 1.4,
        8: 1.8,   9: 1.6,  10: 1.2,  11: 1.1,
        12: 1.2,  13: 1.2,  14: 1.1,  15: 1.2,
        16: 1.5,  17: 1.9,  18: 1.8,  19: 1.5,
        20: 1.2,  21: 1.0,  22: 0.9,  23: 0.8,
    }

    # ========================================================
    # GET TIME-OF-DAY MULTIPLIER
    # ========================================================

    def get_time_multiplier(self, hour=None):
        """
        Returns a traffic multiplier for the given hour (0-23).
        If no hour is given, uses the current system time.
        """
        if hour is None:
            hour = datetime.now().hour
        hour = int(hour) % 24
        return self.HOURLY_MULTIPLIERS.get(hour, 1.0)

    # ========================================================
    # GENERATE CONGESTION EVENTS
    # ========================================================

    def generate_congestion(self, n, congestion_probability=0.15,
                            severity_range=(1.3, 2.5)):
        """
        Generate a matrix of congestion multipliers.

        Args:
            n: number of locations (matrix dimension)
            congestion_probability: probability that any segment is congested
            severity_range: (min_multiplier, max_multiplier) for congested segments

        Returns:
            n×n matrix of congestion multipliers (1.0 for normal, >1.0 for congested)
        """
        if n < 1:
            raise ValueError("n must be a positive matrix dimension")
        if not 0 <= congestion_probability <= 1:
            raise ValueError("congestion_probability must be between 0 and 1")
        if len(severity_range) != 2 or severity_range[0] < 1 or severity_range[1] < severity_range[0]:
            raise ValueError("severity_range must contain valid multipliers >= 1")

        congestion_matrix = np.ones((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                if i != j and random.random() < congestion_probability:
                    severity = random.uniform(*severity_range)
                    congestion_matrix[i, j] = severity

        return congestion_matrix

    # ========================================================
    # WEATHER IMPACT
    # ========================================================

    def get_weather_multiplier(self, condition="clear"):
        """
        Returns a global multiplier based on weather conditions.

        Args:
            condition: "clear", "rain", "heavy_rain", "fog"

        Returns:
            float multiplier
        """
        weather_multipliers = {
            "clear": 1.0,
            "rain": 1.3,
            "heavy_rain": 1.6,
            "fog": 1.4,
        }
        return weather_multipliers.get(condition.lower(), 1.0)

    # ========================================================
    # APPLY TRAFFIC TO DISTANCE MATRIX
    # ========================================================

    def apply_traffic(self, distance_matrix, hour=None,
                      congestion_probability=0.15,
                      weather="clear"):
        """
        Apply all traffic factors to a base distance matrix.

        Args:
            distance_matrix: n×n numpy array of base distances (km)
            hour: time of day (0-23), None for current time
            congestion_probability: probability of congestion per segment
            weather: "clear", "rain", "heavy_rain", "fog"

        Returns:
            (traffic_matrix, traffic_info) where:
            - traffic_matrix: n×n array with traffic-adjusted distances
            - traffic_info: dict with applied factors
        """
        dm = np.asarray(distance_matrix, dtype=float)
        if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
            raise ValueError("distance_matrix must be a square 2-D array")
        n = dm.shape[0]

        # Time-of-day factor
        time_mult = self.get_time_multiplier(hour)

        # Weather factor
        weather_mult = self.get_weather_multiplier(weather)

        # Random congestion events
        congestion = self.generate_congestion(
            n, congestion_probability
        )

        # Combined traffic matrix
        traffic_matrix = dm * time_mult * weather_mult * congestion

        # Preserve diagonal zeros
        np.fill_diagonal(traffic_matrix, 0.0)

        # Count congested segments
        congested_count = int(np.sum(congestion > 1.05))

        traffic_info = {
            "hour": hour if hour is not None else datetime.now().hour,
            "time_multiplier": round(time_mult, 2),
            "weather": weather,
            "weather_multiplier": round(weather_mult, 2),
            "congested_segments": max(0, congested_count),
            "total_segments": n * (n - 1),
            "avg_congestion": round(float(np.mean(congestion)), 3),
        }

        return traffic_matrix, traffic_info

    # ========================================================
    # GET TRAFFIC SUMMARY
    # ========================================================

    def get_traffic_level(self, hour=None):
        """
        Returns a human-readable traffic level string.
        """
        mult = self.get_time_multiplier(hour)
        if mult <= 0.8:
            return "🟢 Free Flow"
        elif mult <= 1.1:
            return "🟡 Normal"
        elif mult <= 1.5:
            return "🟠 Moderate"
        else:
            return "🔴 Heavy"

    def get_hourly_profile(self):
        """
        Returns a list of (hour, multiplier) tuples for charting.
        """
        return [
            (h, self.HOURLY_MULTIPLIERS[h])
            for h in range(24)
        ]
