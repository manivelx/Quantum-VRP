import random
import numpy as np


class QPSO_VRP:
    """
    Quantum-Inspired Particle Swarm Optimization
    for Capacitated Vehicle Routing Problem.

    Representation:
        Each particle contains a continuous random-key
        vector.

        Example:
            [0.72, 0.11, 0.93, 0.35, 0.52]

        Sorting the values creates a customer sequence.

        The sequence is then divided into vehicle routes
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
        num_particles=30,
        max_iterations=100,
        beta=0.75,
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

        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.beta = beta

        random.seed(seed)
        np.random.seed(seed)

        self.dimension = len(customer_indices)

    # ========================================================
    # DECODE PARTICLE
    # ========================================================

    def decode(self, position):

        order = np.argsort(position)

        ordered_customers = [
            self.customer_indices[i]
            for i in order
        ]

        routes = []

        current_route = []
        current_load = 0.0

        for customer in ordered_customers:

            demand = self.demands[customer]

            if (
                current_route
                and current_load + demand
                > self.vehicle_capacity
            ):
                routes.append(current_route)

                current_route = []
                current_load = 0.0

            current_route.append(customer)
            current_load += demand

        if current_route:
            routes.append(current_route)

        if len(routes) > self.num_vehicles:
            return routes, False

        return routes, True

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

        # Return to depot
        distance += self.distance_matrix[
            previous,
            self.depot_index
        ]

        return distance

    # ========================================================
    # FITNESS FUNCTION
    # ========================================================

    def fitness(self, position):

        routes, feasible = self.decode(position)

        total_distance = sum(
            self.route_distance(route)
            for route in routes
        )

        penalty = 0.0

        # ----------------------------------------------------
        # Vehicle constraint
        # ----------------------------------------------------

        if not feasible:

            excess_routes = max(
                0,
                len(routes) - self.num_vehicles
            )

            penalty += (
                excess_routes
                * self.vehicle_capacity
                * 1000
            )

        # ----------------------------------------------------
        # Capacity constraint
        # ----------------------------------------------------

        for route in routes:

            load = sum(
                self.demands[c]
                for c in route
            )

            if load > self.vehicle_capacity:

                penalty += (
                    load
                    - self.vehicle_capacity
                ) * 10000

        return total_distance + penalty

    # ========================================================
    # QPSO OPTIMIZATION
    # ========================================================

    def optimize(self):

        if self.dimension == 0:

            return [], 0.0, 0

        # ----------------------------------------------------
        # Initialize particles
        # ----------------------------------------------------

        particles = np.random.uniform(
            0,
            1,
            (
                self.num_particles,
                self.dimension
            )
        )

        # ----------------------------------------------------
        # Personal best
        # ----------------------------------------------------

        personal_best = particles.copy()

        personal_best_fitness = np.array([
            self.fitness(particle)
            for particle in particles
        ])

        # ----------------------------------------------------
        # Global best
        # ----------------------------------------------------

        best_index = np.argmin(
            personal_best_fitness
        )

        global_best = personal_best[
            best_index
        ].copy()

        global_best_fitness = (
            personal_best_fitness[
                best_index
            ]
        )

        # ----------------------------------------------------
        # Optimization history
        # ----------------------------------------------------

        history = [
            global_best_fitness
        ]

        # ====================================================
        # QPSO ITERATIONS
        # ====================================================

        for iteration in range(
            self.max_iterations
        ):

            # ------------------------------------------------
            # Mean best position
            # ------------------------------------------------

            mbest = np.mean(
                personal_best,
                axis=0
            )

            # ------------------------------------------------
            # Update every particle
            # ------------------------------------------------

            for i in range(
                self.num_particles
            ):

                # Random coefficient
                phi = np.random.uniform(
                    0,
                    1,
                    self.dimension
                )

                # Local attractor
                attractor = (
                    phi * personal_best[i]
                    + (1 - phi) * global_best
                )

                # Random value
                u = np.random.uniform(
                    0.000001,
                    1,
                    self.dimension
                )

                # Random direction
                sign = np.where(
                    np.random.random(
                        self.dimension
                    ) < 0.5,
                    -1,
                    1
                )

                # ------------------------------------------------
                # QPSO position update
                # ------------------------------------------------

                new_position = (
                    attractor
                    + sign
                    * self.beta
                    * np.abs(
                        mbest - particles[i]
                    )
                    * np.log(1 / u)
                )

                # Keep random keys inside [0, 1]
                new_position = np.clip(
                    new_position,
                    0,
                    1
                )

                particles[i] = new_position

                # ------------------------------------------------
                # Evaluate particle
                # ------------------------------------------------

                current_fitness = self.fitness(
                    new_position
                )

                # ------------------------------------------------
                # Update personal best
                # ------------------------------------------------

                if (
                    current_fitness
                    < personal_best_fitness[i]
                ):

                    personal_best[i] = (
                        new_position.copy()
                    )

                    personal_best_fitness[i] = (
                        current_fitness
                    )

                # ------------------------------------------------
                # Update global best
                # ------------------------------------------------

                if (
                    current_fitness
                    < global_best_fitness
                ):

                    global_best = (
                        new_position.copy()
                    )

                    global_best_fitness = (
                        current_fitness
                    )

            history.append(
                global_best_fitness
            )

        # ====================================================
        # FINAL SOLUTION
        # ====================================================

        routes, feasible = self.decode(
            global_best
        )

        return (
            routes,
            global_best_fitness,
            len(history) - 1
        )