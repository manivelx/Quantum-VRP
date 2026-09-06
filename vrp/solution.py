from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Route:

    vehicle_id: str

    customers: List[str] = field(
        default_factory=list
    )

    def full_route(self, depot_id: str):

        return [
            depot_id,
            *self.customers,
            depot_id
        ]


@dataclass
class VRPSolution:

    routes: List[Route] = field(
        default_factory=list
    )

    total_distance: float = 0.0

    total_time: float = 0.0

    total_penalty: float = 0.0

    objective_value: float = float("inf")

    feasible: bool = False

    def copy(self):

        copied_routes = [
            Route(
                vehicle_id=route.vehicle_id,
                customers=list(route.customers)
            )
            for route in self.routes
        ]

        return VRPSolution(
            routes=copied_routes,
            total_distance=self.total_distance,
            total_time=self.total_time,
            total_penalty=self.total_penalty,
            objective_value=self.objective_value,
            feasible=self.feasible
        )

    def get_all_customers(self):

        customers = []

        for route in self.routes:

            customers.extend(
                route.customers
            )

        return customers

    def number_of_routes(self):

        return len(self.routes)

    def summary(self, depot_id: str):

        result = {}

        for route in self.routes:

            result[route.vehicle_id] = (
                route.full_route(depot_id)
            )

        return result