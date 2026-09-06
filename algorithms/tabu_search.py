import random
import numpy as np


class TabuSearchVRP:
    """
    Tabu Search for Capacitated Vehicle Routing Problem (CVRP).

    Uses:
    - Customer permutation representation
    - Swap neighborhood
    - Tabu list
    - Aspiration criterion
    - Vehicle capacity constraint
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
        tabu_tenure=10,
        neighborhood_size=50,
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
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size

        random.seed(seed)
        np.random.seed(seed)

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

            current_route.append(customer)
            current_load += demand

        if current_route:
            routes.append(
                current_route
            )

        return routes

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
    # CHECK FEASIBILITY
    # ========================================================

    def is_feasible(self, routes):

        if len(routes) > self.num_vehicles:
            return False

        for route in routes:

            if (
                self.route_load(route)
                > self.vehicle_capacity
            ):

                return False

        return True

    # ========================================================
    # FITNESS
    # ========================================================

    def fitness(self, routes):

        distance = self.total_distance(routes)

        penalty = 0.0

        # Too many vehicles
        if len(routes) > self.num_vehicles:

            penalty += (
                len(routes) - self.num_vehicles
            ) * self.vehicle_capacity * 1000

        # Capacity violation
        for route in routes:

            load = self.route_load(route)

            if load > self.vehicle_capacity:

                penalty += (
                    load - self.vehicle_capacity
                ) * 10000

        return distance + penalty

    # ========================================================
    # COPY SOLUTION
    # ========================================================

    def copy_solution(self, routes):

        return [
            route.copy()
            for route in routes
        ]

    # ========================================================
    # GENERATE SWAP NEIGHBOR
    # ========================================================

    def generate_neighbor(
        self,
        routes
    ):

        neighbor = self.copy_solution(
            routes
        )

        non_empty_routes = [
            route
            for route in neighbor
            if route
        ]

        if not non_empty_routes:
            return neighbor, None

        route_a = random.choice(
            non_empty_routes
        )

        route_b = random.choice(
            non_empty_routes
        )

        index_a = random.randrange(
            len(route_a)
        )

        index_b = random.randrange(
            len(route_b)
        )

        customer_a = route_a[index_a]
        customer_b = route_b[index_b]

        route_a[index_a] = customer_b
        route_b[index_b] = customer_a

        move = (
            min(customer_a, customer_b),
            max(customer_a, customer_b)
        )

        return neighbor, move

    # ========================================================
    # OPTIMIZE
    # ========================================================

    def optimize(self):

        if not self.customer_indices:

            return [], 0.0, 0

        # ----------------------------------------------------
        # Initial solution
        # ----------------------------------------------------

        current_solution = (
            self.create_initial_solution()
        )

        current_fitness = self.fitness(
            current_solution
        )

        best_solution = (
            self.copy_solution(
                current_solution
            )
        )

        best_fitness = current_fitness

        # ----------------------------------------------------
        # Tabu list
        # ----------------------------------------------------

        tabu_list = {}

        # ----------------------------------------------------
        # Main search
        # ----------------------------------------------------

        for iteration in range(
            self.max_iterations
        ):

            best_neighbor = None
            best_neighbor_fitness = float(
                "inf"
            )
            best_move = None

            # ------------------------------------------------
            # Generate neighborhood
            # ------------------------------------------------

            for _ in range(
                self.neighborhood_size
            ):

                neighbor, move = (
                    self.generate_neighbor(
                        current_solution
                    )
                )

                neighbor_fitness = (
                    self.fitness(
                        neighbor
                    )
                )

                is_tabu = (
                    move in tabu_list
                )

                # ------------------------------------------------
                # Aspiration criterion
                # ------------------------------------------------

                aspiration = (
                    neighbor_fitness
                    < best_fitness
                )

                # ------------------------------------------------
                # Accept candidate
                # ------------------------------------------------

                if (
                    not is_tabu
                    or aspiration
                ):

                    if (
                        neighbor_fitness
                        < best_neighbor_fitness
                    ):

                        best_neighbor = (
                            neighbor
                        )

                        best_neighbor_fitness = (
                            neighbor_fitness
                        )

                        best_move = move

            # ----------------------------------------------------
            # No neighbor found
            # ----------------------------------------------------

            if best_neighbor is None:
                continue

            # ----------------------------------------------------
            # Move to best neighbor
            # ----------------------------------------------------

            current_solution = (
                best_neighbor
            )

            current_fitness = (
                best_neighbor_fitness
            )

            # ----------------------------------------------------
            # Add move to tabu list
            # ----------------------------------------------------

            if best_move is not None:

                tabu_list[best_move] = (
                    iteration
                    + self.tabu_tenure
                )

            # ----------------------------------------------------
            # Remove expired tabu moves
            # ----------------------------------------------------

            expired_moves = [

                move
                for move, expiry
                in tabu_list.items()

                if expiry <= iteration

            ]

            for move in expired_moves:

                del tabu_list[move]

            # ----------------------------------------------------
            # Update global best
            # ----------------------------------------------------

            if (
                current_fitness
                < best_fitness
            ):

                best_solution = (
                    self.copy_solution(
                        current_solution
                    )
                )

                best_fitness = (
                    current_fitness
                )

        return (
            best_solution,
            best_fitness,
            self.max_iterations
        )