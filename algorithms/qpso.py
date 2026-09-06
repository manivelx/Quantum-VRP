import random
import numpy as np


class QPSO_VRP:
    """
    Quantum-Inspired Particle Swarm Optimization
    for Capacitated Vehicle Routing Problem (CVRP).

    Random-key representation:

        Particle:
        [0.72, 0.11, 0.93, 0.35, 0.52]

    Sorting the particle gives the customer visiting order.

    Example:

        C2 -> C4 -> C5 -> C1 -> C3

    The ordered customers are then split into routes
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

        # ----------------------------------------------------
        # Problem data
        # ----------------------------------------------------

        self.distance_matrix = np.asarray(
            distance_matrix,
            dtype=float
        )

        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index

        # ----------------------------------------------------
        # Vehicle constraints
        # ----------------------------------------------------

        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        # ----------------------------------------------------
        # QPSO parameters
        # ----------------------------------------------------

        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.beta = beta

        # ----------------------------------------------------
        # Random seed
        # ----------------------------------------------------

        random.seed(seed)
        np.random.seed(seed)

        # Number of customers
        self.dimension = len(
            customer_indices
        )

    # ========================================================
    # DECODE PARTICLE
    # ========================================================

    def decode(self, position):
        """
        Convert a random-key particle into VRP routes.

        Example particle:

            [0.8, 0.2, 0.5, 0.1]

        Sorting gives:

            Customer 4
            Customer 2
            Customer 3
            Customer 1

        Then customers are divided according
        to vehicle capacity.
        """

        # ----------------------------------------------------
        # Sort particle values
        # ----------------------------------------------------

        order = np.argsort(position)

        # ----------------------------------------------------
        # Convert sorted positions to customer IDs
        # ----------------------------------------------------

        ordered_customers = [
            self.customer_indices[i]
            for i in order
        ]

        # ----------------------------------------------------
        # Create routes
        # ----------------------------------------------------

        routes = []

        current_route = []
        current_load = 0.0

        # ----------------------------------------------------
        # Assign customers to vehicles
        # ----------------------------------------------------

        for customer in ordered_customers:

            demand = self.demands[customer]

            # If adding this customer exceeds capacity,
            # close the current route.
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

        # ----------------------------------------------------
        # Add final route
        # ----------------------------------------------------

        if current_route:

            routes.append(
                current_route
            )

        # ----------------------------------------------------
        # Check number of vehicles
        # ----------------------------------------------------

        if len(routes) > self.num_vehicles:

            return routes, False

        return routes, True

    # ========================================================
    # ROUTE DISTANCE
    # ========================================================

    def route_distance(self, route):
        """
        Calculate total distance of one vehicle route.

        Route:

            DEPOT -> C1 -> C2 -> C3 -> DEPOT
        """

        if not route:

            return 0.0

        distance = 0.0

        # Start at depot
        previous = self.depot_index

        # ----------------------------------------------------
        # Travel through customers
        # ----------------------------------------------------

        for customer in route:

            distance += self.distance_matrix[
                previous,
                customer
            ]

            previous = customer

        # ----------------------------------------------------
        # Return to depot
        # ----------------------------------------------------

        distance += self.distance_matrix[
            previous,
            self.depot_index
        ]

        return distance

    # ========================================================
    # FITNESS FUNCTION
    # ========================================================

    def fitness(self, position):
        """
        Calculate the fitness of a particle.

        Lower distance = better solution.

        Penalties are added if:

        1. Number of routes exceeds available vehicles.
        2. Vehicle capacity is exceeded.
        """

        # ----------------------------------------------------
        # Decode particle
        # ----------------------------------------------------

        routes, feasible = self.decode(
            position
        )

        # ----------------------------------------------------
        # Calculate total distance
        # ----------------------------------------------------

        total_distance = sum(
            self.route_distance(route)
            for route in routes
        )

        penalty = 0.0

        # ----------------------------------------------------
        # Vehicle-count penalty
        # ----------------------------------------------------

        if not feasible:

            excess_routes = max(
                0,
                len(routes)
                - self.num_vehicles
            )

            penalty += (
                excess_routes
                * self.vehicle_capacity
                * 1000
            )

        # ----------------------------------------------------
        # Capacity penalty
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

        # ----------------------------------------------------
        # Final fitness
        # ----------------------------------------------------

        return (
            total_distance
            + penalty
        )

    # ========================================================
    # QPSO OPTIMIZATION
    # ========================================================

    def optimize(self):
        """
        Run Quantum-Inspired Particle Swarm Optimization.

        Returns:

            routes
            best_distance
            iterations
        """

        # ----------------------------------------------------
        # No customers
        # ----------------------------------------------------

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
        # Personal best positions
        # ----------------------------------------------------

        personal_best = particles.copy()

        # ----------------------------------------------------
        # Personal best fitness
        # ----------------------------------------------------

        personal_best_fitness = np.array(
            [
                self.fitness(p)
                for p in particles
            ]
        )

        # ----------------------------------------------------
        # Find initial global best
        # ----------------------------------------------------

        best_index = np.argmin(
            personal_best_fitness
        )

        global_best = (
            personal_best[
                best_index
            ].copy()
        )

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
        # MAIN QPSO LOOP
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

                # ------------------------------------------------
                # Local attractor
                # ------------------------------------------------

                attractor = (
                    phi
                    * personal_best[i]
                    +
                    (1 - phi)
                    * global_best
                )

                # ------------------------------------------------
                # Random value
                # ------------------------------------------------

                u = np.random.uniform(
                    0.000001,
                    1,
                    self.dimension
                )

                # ------------------------------------------------
                # Random direction
                # ------------------------------------------------

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
                    +
                    sign
                    * self.beta
                    * np.abs(
                        mbest
                        - particles[i]
                    )
                    * np.log(
                        1 / u
                    )
                )

                # ------------------------------------------------
                # Keep particle inside [0, 1]
                # ------------------------------------------------

                new_position = np.clip(
                    new_position,
                    0,
                    1
                )

                # ------------------------------------------------
                # Update particle
                # ------------------------------------------------

                particles[i] = (
                    new_position
                )

                # ------------------------------------------------
                # Evaluate new particle
                # ------------------------------------------------

                current_fitness = (
                    self.fitness(
                        new_position
                    )
                )

                # ------------------------------------------------
                # Update personal best
                # ------------------------------------------------

                if (
                    current_fitness
                    <
                    personal_best_fitness[i]
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
                    <
                    global_best_fitness
                ):

                    global_best = (
                        new_position.copy()
                    )

                    global_best_fitness = (
                        current_fitness
                    )

            # ------------------------------------------------
            # Store iteration result
            # ------------------------------------------------

            history.append(
                global_best_fitness
            )

        # ====================================================
        # DECODE FINAL BEST PARTICLE
        # ====================================================

        routes, feasible = self.decode(
            global_best
        )

        return (
            routes,
            global_best_fitness,
            len(history) - 1
        )