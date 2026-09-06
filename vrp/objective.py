from typing import Dict, List

from vrp.problem import VRPProblem
from vrp.constraints import ConstraintChecker


class ObjectiveEvaluator:
    """
    Calculates the fitness/objective value
    of a VRP solution.

    Lower objective value = better solution.
    """

    def __init__(
        self,
        problem: VRPProblem,
        constraint_checker: ConstraintChecker
    ):

        self.problem = problem
        self.constraint_checker = constraint_checker

    # ========================================================
    # ROUTE DISTANCE
    # ========================================================

    def calculate_route_distance(
        self,
        route_customers: List[str]
    ):

        depot = self.problem.depot_id

        full_route = [
            depot,
            *route_customers,
            depot
        ]

        total_distance = 0.0

        for i in range(len(full_route) - 1):

            source = full_route[i]
            destination = full_route[i + 1]

            total_distance += (
                self.problem.distance_matrix[source][destination]
            )

        return total_distance

    # ========================================================
    # ROUTE TIME
    # ========================================================

    def calculate_route_time(
        self,
        route_customers: List[str]
    ):

        depot = self.problem.depot_id

        full_route = [
            depot,
            *route_customers,
            depot
        ]

        total_time = 0.0

        for i in range(len(full_route) - 1):

            source = full_route[i]
            destination = full_route[i + 1]

            total_time += (
                self.problem.time_matrix[source][destination]
            )

        # Add customer service times
        for customer in route_customers:

            total_time += (
                self.problem.service_time.get(
                    customer,
                    0.0
                )
            )

        return total_time

    # ========================================================
    # TOTAL DISTANCE
    # ========================================================

    def calculate_total_distance(
        self,
        solution
    ):

        total = 0.0

        for route in solution.routes:

            total += self.calculate_route_distance(
                route.customers
            )

        return total

    # ========================================================
    # TOTAL TIME
    # ========================================================

    def calculate_total_time(
        self,
        solution
    ):

        total = 0.0

        for route in solution.routes:

            total += self.calculate_route_time(
                route.customers
            )

        return total

    # ========================================================
    # FINAL FITNESS
    # ========================================================

    def evaluate(self, solution):

        total_distance = (
            self.calculate_total_distance(solution)
        )

        total_time = (
            self.calculate_total_time(solution)
        )

        penalty = (
            self.constraint_checker.calculate_penalty(
                solution
            )
        )

        # ----------------------------------------------------
        # OBJECTIVE WEIGHTS
        # ----------------------------------------------------

        distance_weight = 1.0
        time_weight = 1.0

        objective = (
            distance_weight * total_distance
            +
            time_weight * total_time
            +
            penalty
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        solution.total_distance = total_distance
        solution.total_time = total_time
        solution.total_penalty = penalty
        solution.objective_value = objective

        solution.feasible = (
            penalty == 0
        )

        return solution