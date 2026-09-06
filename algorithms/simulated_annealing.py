import random
import math
import numpy as np


class SimulatedAnnealingVRP:
    """
    Simulated Annealing (SA) for Capacitated Vehicle Routing Problem.

    Representation:
        A solution is a list of routes.

        Example:
        [
            [1, 4, 2],
            [3, 5]
        ]

        This means:

        Vehicle 1:
        DEPOT -> C1 -> C4 -> C2 -> DEPOT

        Vehicle 2:
        DEPOT -> C3 -> C5 -> DEPOT
    """

    def __init__(
        self,
        distance_matrix,
        demands,
        customer_indices,
        depot_index,
        vehicle_capacity,
        num_vehicles,
        max_iterations=1000,
        initial_temperature=1000.0,
        cooling_rate=0.995,
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

        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

        random.seed(seed)
        np.random.seed(seed)

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
            self.demands[customer]
            for customer in route
        )

    # ========================================================
    # CHECK CAPACITY
    # ========================================================

    def is_feasible(self, routes):

        if len(routes) > self.num_vehicles:
            return False

        for route in routes:

            load = self.route_load(route)

            if load > self.vehicle_capacity:
                return False

        return True

    # ========================================================
    # FITNESS
    # ========================================================

    def fitness(self, routes):

        distance = self.total_distance(routes)

        penalty = 0.0

        # Too many vehicles/routes
        if len(routes) > self.num_vehicles:

            excess_routes = (
                len(routes)
                - self.num_vehicles
            )

            penalty += (
                excess_routes
                * self.vehicle_capacity
                * 1000
            )

        # Capacity violation
        for route in routes:

            load = self.route_load(route)

            if load > self.vehicle_capacity:

                penalty += (
                    load
                    - self.vehicle_capacity
                ) * 10000

        return distance + penalty

    # ========================================================
    # CREATE INITIAL SOLUTION
    # ========================================================

    def create_initial_solution(self):

        customers = self.customer_indices.copy()

        random.shuffle(customers)

        routes = []

        current_route = []
        current_load = 0.0

        for customer in customers:

            demand = self.demands[customer]

            # Start a new route when capacity is exceeded
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
    # COPY SOLUTION
    # ========================================================

    def copy_solution(self, routes):

        return [
            route.copy()
            for route in routes
        ]

    # ========================================================
    # GENERATE NEIGHBOUR
    # ========================================================

    def generate_neighbor(self, routes):

        neighbor = self.copy_solution(
            routes
        )

        if not neighbor:
            return neighbor

        # Choose random neighborhood operation
        operation = random.choice(
            [
                "swap",
                "relocate",
                "reverse"
            ]
        )

        # ----------------------------------------------------
        # SWAP
        # ----------------------------------------------------

        if operation == "swap":

            non_empty_routes = [
                route
                for route in neighbor
                if route
            ]

            if len(non_empty_routes) >= 1:

                route_a = random.choice(
                    non_empty_routes
                )

                route_b = random.choice(
                    non_empty_routes
                )

                if route_a and route_b:

                    index_a = random.randrange(
                        len(route_a)
                    )

                    index_b = random.randrange(
                        len(route_b)
                    )

                    route_a[index_a], route_b[index_b] = (
                        route_b[index_b],
                        route_a[index_a]
                    )

        # ----------------------------------------------------
        # RELOCATE
        # ----------------------------------------------------

        elif operation == "relocate":

            non_empty_routes = [
                route
                for route in neighbor
                if route
            ]

            if non_empty_routes:

                source = random.choice(
                    non_empty_routes
                )

                customer = random.choice(
                    source
                )

                source.remove(
                    customer
                )

                target = random.choice(
                    neighbor
                )

                insert_position = random.randint(
                    0,
                    len(target)
                )

                target.insert(
                    insert_position,
                    customer
                )

                # Remove empty routes
                neighbor = [
                    route
                    for route in neighbor
                    if route
                ]

        # ----------------------------------------------------
        # REVERSE
        # ----------------------------------------------------

        elif operation == "reverse":

            route = random.choice(
                neighbor
            )

            if len(route) >= 2:

                start = random.randint(
                    0,
                    len(route) - 2
                )

                end = random.randint(
                    start + 1,
                    len(route) - 1
                )

                route[start:end + 1] = reversed(
                    route[start:end + 1]
                )

        return neighbor

    # ========================================================
    # OPTIMIZE
    # ========================================================

    def optimize(self):

        if not self.customer_indices:
            return [], 0.0, 0, []

        # Initial solution
        current_solution = self.create_initial_solution()
        current_fitness = self.fitness(current_solution)

        best_solution = self.copy_solution(current_solution)
        best_fitness = current_fitness

        temperature = self.initial_temperature

        # Convergence history
        history = [best_fitness]

        # Simulated Annealing loop
        for iteration in range(self.max_iterations):

            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.fitness(neighbor)

            delta = neighbor_fitness - current_fitness

            # Accept better solution
            if delta < 0:
                current_solution = neighbor
                current_fitness = neighbor_fitness

            # Accept worse solution probabilistically
            else:
                if temperature > 0:
                    probability = math.exp(-delta / temperature)
                else:
                    probability = 0.0

                if random.random() < probability:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness

            # Update global best
            if current_fitness < best_fitness:
                best_solution = self.copy_solution(current_solution)
                best_fitness = current_fitness

            # Cool temperature
            temperature *= self.cooling_rate

            if temperature < 1e-10:
                temperature = 1e-10

            # Track convergence
            history.append(best_fitness)

        return (
            best_solution,
            best_fitness,
            self.max_iterations,
            history
        )