import random
import math
from typing import List, Dict, Tuple


class QPSO:
    """
    Quantum-Inspired Particle Swarm Optimization
    for Vehicle Routing Problems.

    This is a classical-computer implementation.
    No quantum computer is required.
    """

    def __init__(
        self,
        distance_matrix,
        demands,
        vehicle_capacity,
        num_vehicles,
        depot_index=0,
        population_size=30,
        iterations=100,
        beta=0.75,
        seed=42
    ):

        self.distance_matrix = distance_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.depot_index = depot_index

        self.population_size = population_size
        self.iterations = iterations
        self.beta = beta

        random.seed(seed)

        self.num_customers = len(distance_matrix) - 1

    # ========================================================
    # CREATE INITIAL SOLUTION
    # ========================================================

    def create_solution(self):

        customers = list(range(1, self.num_customers + 1))

        random.shuffle(customers)

        routes = [[] for _ in range(self.num_vehicles)]

        # Greedy initial distribution
        for customer in customers:

            best_vehicle = None
            best_load = float("inf")

            for vehicle in range(self.num_vehicles):

                current_load = sum(
                    self.demands[c]
                    for c in routes[vehicle]
                )

                if current_load < best_load:

                    best_load = current_load
                    best_vehicle = vehicle

            routes[best_vehicle].append(customer)

        return routes

    # ========================================================
    # SOLUTION → POSITION
    # ========================================================

    def solution_to_position(self, routes):

        position = []

        for route in routes:
            position.extend(route)

        return position

    # ========================================================
    # POSITION → SOLUTION
    # ========================================================

    def position_to_solution(self, position):

        routes = [[] for _ in range(self.num_vehicles)]

        loads = [0] * self.num_vehicles

        for customer in position:

            demand = self.demands[customer]

            # Find a vehicle that can accept customer
            candidates = [
                i
                for i in range(self.num_vehicles)
                if loads[i] + demand <= self.vehicle_capacity
            ]

            if candidates:

                vehicle = min(
                    candidates,
                    key=lambda x: loads[x]
                )

            else:

                # If no vehicle can accept,
                # put into least-loaded vehicle.
                vehicle = min(
                    range(self.num_vehicles),
                    key=lambda x: loads[x]
                )

            routes[vehicle].append(customer)

            loads[vehicle] += demand

        return routes

    # ========================================================
    # ROUTE DISTANCE
    # ========================================================

    def route_distance(self, route):

        if not route:
            return 0

        distance = 0

        previous = self.depot_index

        for customer in route:

            distance += self.distance_matrix[
                previous
            ][customer]

            previous = customer

        # Return to depot
        distance += self.distance_matrix[
            previous
        ][self.depot_index]

        return distance

    # ========================================================
    # CAPACITY VIOLATION
    # ========================================================

    def capacity_violation(self, route):

        load = sum(
            self.demands[c]
            for c in route
        )

        return max(
            0,
            load - self.vehicle_capacity
        )

    # ========================================================
    # FITNESS FUNCTION
    # ========================================================

    def fitness(self, routes):

        total_distance = 0

        penalty = 0

        for route in routes:

            total_distance += self.route_distance(
                route
            )

            violation = self.capacity_violation(
                route
            )

            penalty += violation * 1000

        return total_distance + penalty

    # ========================================================
    # QUANTUM POSITION UPDATE
    # ========================================================

    def quantum_update(
        self,
        current_position,
        personal_best,
        global_best
    ):

        new_position = current_position.copy()

        if not new_position:
            return new_position

        # Quantum-inspired contraction coefficient
        alpha = random.random()

        # Mean best position
        mean_best = (
            sum(personal_best) /
            len(personal_best)
        )

        for i in range(len(new_position)):

            r = random.random()

            # Quantum-inspired attraction
            if r < 0.5:

                target = personal_best[i]

            else:

                target = global_best[i]

            # Quantum delta
            distance = abs(
                target - new_position[i]
            )

            quantum_step = (
                self.beta *
                math.log(
                    1.0 / max(alpha, 1e-10)
                ) *
                distance
            )

            # Decide whether to move toward
            # personal/global best
            if random.random() < 0.5:

                if new_position[i] < target:

                    new_position[i] += int(
                        quantum_step + 0.5
                    )

                else:

                    new_position[i] -= int(
                        quantum_step + 0.5
                    )

        # ====================================================
        # IMPORTANT:
        # Routing is permutation-based.
        # Therefore we repair the solution after
        # quantum-inspired movement.
        # ====================================================

        valid_customers = list(
            range(
                1,
                self.num_customers + 1
            )
        )

        # Keep only valid customers
        new_position = [
            x
            for x in new_position
            if x in valid_customers
        ]

        # Remove duplicates
        repaired = []

        for customer in new_position:

            if customer not in repaired:

                repaired.append(customer)

        # Add missing customers
        for customer in valid_customers:

            if customer not in repaired:

                repaired.append(customer)

        return repaired

    # ========================================================
    # LOCAL SEARCH
    # ========================================================

    def local_search(self, routes):

        best_routes = [
            route.copy()
            for route in routes
        ]

        best_cost = self.fitness(
            best_routes
        )

        for _ in range(10):

            candidate = [
                route.copy()
                for route in best_routes
            ]

            non_empty_routes = [
                r
                for r in candidate
                if r
            ]

            if not non_empty_routes:
                break

            route = random.choice(
                non_empty_routes
            )

            if len(route) < 2:
                continue

            i, j = random.sample(
                range(len(route)),
                2
            )

            route[i], route[j] = (
                route[j],
                route[i]
            )

            candidate_cost = self.fitness(
                candidate
            )

            if candidate_cost < best_cost:

                best_routes = candidate
                best_cost = candidate_cost

        return best_routes

    # ========================================================
    # MAIN QPSO SOLVER
    # ========================================================

    def solve(self):

        population = []

        personal_best = []

        personal_best_costs = []

        # ====================================================
        # INITIAL POPULATION
        # ====================================================

        for _ in range(
            self.population_size
        ):

            solution = self.create_solution()

            position = self.solution_to_position(
                solution
            )

            population.append(position)

            personal_best.append(
                position.copy()
            )

            personal_best_costs.append(
                self.fitness(solution)
            )

        # ====================================================
        # FIND GLOBAL BEST
        # ====================================================

        best_index = min(
            range(
                len(personal_best_costs)
            ),
            key=lambda i:
            personal_best_costs[i]
        )

        global_best = personal_best[
            best_index
        ].copy()

        global_best_cost = (
            personal_best_costs[
                best_index
            ]
        )

        convergence = []

        # ====================================================
        # QPSO ITERATIONS
        # ====================================================

        for iteration in range(
            self.iterations
        ):

            # Gradually reduce beta
            beta = (
                0.9 -
                (
                    0.5 *
                    iteration /
                    max(
                        1,
                        self.iterations - 1
                    )
                )
            )

            self.beta = beta

            for i in range(
                self.population_size
            ):

                new_position = (
                    self.quantum_update(
                        population[i],
                        personal_best[i],
                        global_best
                    )
                )

                routes = (
                    self.position_to_solution(
                        new_position
                    )
                )

                # Constraint repair
                routes = self.repair_capacity(
                    routes
                )

                # Local improvement
                routes = self.local_search(
                    routes
                )

                cost = self.fitness(
                    routes
                )

                population[i] = (
                    self.solution_to_position(
                        routes
                    )
                )

                # Update personal best
                if cost < personal_best_costs[i]:

                    personal_best[i] = (
                        population[i].copy()
                    )

                    personal_best_costs[i] = cost

                # Update global best
                if cost < global_best_cost:

                    global_best = (
                        population[i].copy()
                    )

                    global_best_cost = cost

            convergence.append(
                global_best_cost
            )

            print(
                f"Iteration "
                f"{iteration + 1}/{self.iterations} "
                f"| Best Cost: "
                f"{global_best_cost:.2f}"
            )

        # ====================================================
        # FINAL SOLUTION
        # ====================================================

        final_routes = (
            self.position_to_solution(
                global_best
            )
        )

        final_routes = self.repair_capacity(
            final_routes
        )

        return {
            "routes": final_routes,
            "cost": global_best_cost,
            "convergence": convergence
        }

    # ========================================================
    # CAPACITY REPAIR
    # ========================================================

    def repair_capacity(self, routes):

        repaired_routes = [
            []
            for _ in range(
                self.num_vehicles
            )
        ]

        vehicle_loads = [
            0
            for _ in range(
                self.num_vehicles
            )
        ]

        # Flatten customers
        customers = []

        for route in routes:

            customers.extend(route)

        # Reinsert customers while respecting capacity
        for customer in customers:

            demand = self.demands[
                customer
            ]

            possible = [
                v
                for v in range(
                    self.num_vehicles
                )
                if (
                    vehicle_loads[v]
                    + demand
                    <= self.vehicle_capacity
                )
            ]

            if possible:

                vehicle = min(
                    possible,
                    key=lambda v:
                    vehicle_loads[v]
                )

                repaired_routes[
                    vehicle
                ].append(customer)

                vehicle_loads[
                    vehicle
                ] += demand

            else:

                # Keep customer in least-loaded
                # vehicle if total capacity is insufficient.
                vehicle = min(
                    range(
                        self.num_vehicles
                    ),
                    key=lambda v:
                    vehicle_loads[v]
                )

                repaired_routes[
                    vehicle
                ].append(customer)

                vehicle_loads[
                    vehicle
                ] += demand

        return repaired_routes