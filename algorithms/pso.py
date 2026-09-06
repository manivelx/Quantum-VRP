import random
import numpy as np


class PSO_VRP:
    """
    Particle Swarm Optimization for
    Capacitated Vehicle Routing Problem.

    Uses random-key representation.
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
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        seed=42
    ):
        self.distance_matrix = np.asarray(
            distance_matrix, dtype=float
        )
        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index

        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        self.num_particles = num_particles
        self.max_iterations = max_iterations

        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

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

        feasible = len(routes) <= self.num_vehicles
        return routes, feasible

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
                previous, customer
            ]
            previous = customer

        distance += self.distance_matrix[
            previous, self.depot_index
        ]
        return distance

    # ========================================================
    # FITNESS
    # ========================================================

    def fitness(self, position):
        routes, feasible = self.decode(position)

        total_distance = sum(
            self.route_distance(route)
            for route in routes
        )

        penalty = 0.0
        if not feasible:
            excess_routes = (
                len(routes) - self.num_vehicles
            )
            penalty += (
                excess_routes
                * self.vehicle_capacity
                * 1000
            )

        return total_distance + penalty

    # ========================================================
    # OPTIMIZE
    # ========================================================

    def optimize(self):
        if self.dimension == 0:
            return [], 0.0, 0, []

        # Initialize particles
        particles = np.random.uniform(
            0, 1,
            (self.num_particles, self.dimension)
        )

        # Initial velocities
        velocities = np.random.uniform(
            -0.1, 0.1,
            (self.num_particles, self.dimension)
        )

        # Personal best
        personal_best = particles.copy()
        personal_best_fitness = np.array([
            self.fitness(p) for p in particles
        ])

        # Global best
        best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[best_index].copy()
        global_best_fitness = personal_best_fitness[best_index]

        # Convergence history
        history = [global_best_fitness]

        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                r1 = np.random.random(self.dimension)
                r2 = np.random.random(self.dimension)

                # Update velocity
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.cognitive * r1 * (
                        personal_best[i] - particles[i]
                    )
                    + self.social * r2 * (
                        global_best - particles[i]
                    )
                )

                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)

                # Fitness
                current_fitness = self.fitness(particles[i])

                # Personal best update
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = current_fitness

                # Global best update
                if current_fitness < global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = current_fitness

            history.append(global_best_fitness)

        # Final routes
        routes, feasible = self.decode(global_best)

        return (
            routes,
            global_best_fitness,
            self.max_iterations,
            history
        )