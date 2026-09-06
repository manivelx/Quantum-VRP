from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Vehicle:

    vehicle_id: str

    capacity: float

    max_distance: float = float("inf")

    max_time: float = float("inf")

    available: bool = True


@dataclass
class VRPProblem:

    depot_id: str

    customers: List[str]

    demands: Dict[str, float]

    vehicles: Dict[str, Vehicle]

    distance_matrix: Dict[str, Dict[str, float]]

    time_matrix: Dict[str, Dict[str, float]]

    service_time: Dict[str, float] = field(
        default_factory=dict
    )

    time_windows: Dict[str, tuple] = field(
        default_factory=dict
    )