import random
import numpy as np


class ACOVRP:
    """
    Ant Colony Optimization for Capacitated Vehicle Routing Problem.

    Each ant creates a customer sequence.
    The sequence is then split into vehicle routes
    according to vehicle capacity.
    """

    def __init__(
        self,
        distance_matrix,
        demands,
        customer_indices,
        depot_index,
        vehicle_capacity,
        num_vehicles,
        num_ants=30,
        max_iterations=100,
        alpha=1.0,
        beta=2.0,
        evaporation=0.5,
        q=100.0,
        seed=42
    ):

        self.distance_matrix = np.asarray(
            distance_matrix,
            dtype=float
        )

        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index

        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        self.num_ants = num_ants
        self.max_iterations = max_iterations

        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q

        random.seed(seed)
        np.random.seed(seed)

        self.dimension = len(customer_indices)

        self.pheromone = np.ones(
            (
                self.dimension + 1,
                self.dimension + 1
            ),
            dtype=float
        )

    # ========================================================
    # ROUTE DISTANCE
    # ========================================================

    def route_distance(self, route):

        if not route:
            return 0.0

        distance = 0.0

        previous = self.depot_index

        for customer in route:

            distance += self.distance_matrix[
                previous,
                customer
            ]

            previous = customer

        distance += self.distance_matrix[
            previous,
            self.depot_index
        ]

        return distance

    # ========================================================
    # TOTAL DISTANCE
    # ========================================================

    def total_distance(self, routes):

        return sum(
            self.route_distance(route)
            for route in routes
        )

    # ========================================================
    # ROUTE LOAD
    # ========================================================

    def route_load(self, route):

        return sum(
            self.demands[c]
            for c in route
        )

    # ========================================================
    # DECODE CUSTOMER ORDER
    # ========================================================

    def decode(self, customer_order):

        routes = []

        current_route = []
        current_load = 0.0

        for customer in customer_order:

            demand = self.demands[
                customer
            ]

            if (
                current_route
                and current_load + demand
                > self.vehicle_capacity
            ):

                routes.append(
                    current_route
                )

                current_route = []
                current_load = 0.0

            current_route.append(
                customer
            )

            current_load += demand

        if current_route:

            routes.append(
                current_route
            )

        return routes

    # ========================================================
    # FITNESS
    # ========================================================

    def fitness(self, routes):

        distance = self.total_distance(
            routes
        )

        penalty = 0.0

        if len(routes) > self.num_vehicles:

            penalty += (
                len(routes)
                - self.num_vehicles
            ) * self.vehicle_capacity * 1000

        for route in routes:

            load = self.route_load(
                route
            )

            if load > self.vehicle_capacity:

                penalty += (
                    load
                    - self.vehicle_capacity
                ) * 10000

        return distance + penalty

    # ========================================================
    # SELECT NEXT CUSTOMER
    # ========================================================

    def select_next_customer(
        self,
        current,
        unvisited
    ):

        probabilities = []

        total_probability = 0.0

        for customer in unvisited:

            distance = self.distance_matrix[
                current,
                customer
            ]

            if distance <= 0:
                distance = 0.000001

            pheromone = self.pheromone[
                current,
                customer
            ]

            heuristic = 1.0 / distance

            probability = (
                pheromone ** self.alpha
                *
                heuristic ** self.beta
            )

            probabilities.append(
                probability
            )

            total_probability += probability

        if total_probability <= 0:

            return random.choice(
                list(unvisited)
            )

        probabilities = [
            p / total_probability
            for p in probabilities
        ]

        return random.choices(
            list(unvisited),
            weights=probabilities,
            k=1
        )[0]

    # ========================================================
    # BUILD ANT SOLUTION
    # ========================================================

    def build_ant_solution(self):

        unvisited = set(
            self.customer_indices
        )

        customer_order = []

        current = self.depot_index

        while unvisited:

            next_customer = (
                self.select_next_customer(
                    current,
                    unvisited
                )
            )

            customer_order.append(
                next_customer
            )

            unvisited.remove(
                next_customer
            )

            current = next_customer

        routes = self.decode(
            customer_order
        )

        return routes

    # ========================================================
    # PHEROMONE EVAPORATION
    # ========================================================

    def evaporate_pheromone(self):

        self.pheromone *= (
            1.0 - self.evaporation
        )

        self.pheromone = np.maximum(
            self.pheromone,
            0.000001
        )

    # ========================================================
    # PHEROMONE UPDATE
    # ========================================================

    def deposit_pheromone(
        self,
        routes,
        distance
    ):

        if distance <= 0:
            return

        amount = self.q / distance

        for route in routes:

            previous = self.depot_index

            for customer in route:

                self.pheromone[
                    previous,
                    customer
                ] += amount

                self.pheromone[
                    customer,
                    previous
                ] += amount

                previous = customer

            self.pheromone[
                previous,
                self.depot_index
            ] += amount

            self.pheromone[
                self.depot_index,
                previous
            ] += amount

    # ========================================================
    # OPTIMIZATION
    # ========================================================

    def optimize(self):

        if self.dimension == 0:

            return [], 0.0, 0

        best_routes = None
        best_distance = float("inf")

        # ====================================================
        # MAIN ACO LOOP
        # ====================================================

        for iteration in range(
            self.max_iterations
        ):

            iteration_best_routes = None
            iteration_best_distance = float(
                "inf"
            )

            # ------------------------------------------------
            # Create solutions for all ants
            # ------------------------------------------------

            for _ in range(
                self.num_ants
            ):

                routes = (
                    self.build_ant_solution()
                )

                distance = self.fitness(
                    routes
                )

                if (
                    distance
                    < iteration_best_distance
                ):

                    iteration_best_distance = (
                        distance
                    )

                    iteration_best_routes = (
                        routes
                    )

                if (
                    distance
                    < best_distance
                ):

                    best_distance = distance

                    best_routes = [
                        route.copy()
                        for route in routes
                    ]

            # ------------------------------------------------
            # Evaporate
            # ------------------------------------------------

            self.evaporate_pheromone()

            # ------------------------------------------------
            # Update using iteration best
            # ------------------------------------------------

            if iteration_best_routes is not None:

                self.deposit_pheromone(
                    iteration_best_routes,
                    iteration_best_distance
                )

        # ====================================================
        # RETURN BEST SOLUTION
        # ====================================================

        return (
            best_routes,
            best_distance,
            self.max_iterations
        )