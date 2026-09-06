import random
import numpy as np


class QPSO_VRP:
    """
    Enhanced Quantum-Inspired Particle Swarm Optimization
    for Capacitated Vehicle Routing Problem (CVRP).

    Improvements over basic QPSO:
        1. Adaptive β decay (exploration → exploitation)
        2. Nearest-neighbor seeded initialization
        3. 2-opt local search post-optimization
        4. Stagnation detection with partial restart

    Random-key representation:
        Particle: [0.72, 0.11, 0.93, 0.35, 0.52]
    Sorting the particle gives the customer visiting order.
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
        # Problem data
        self.distance_matrix = np.asarray(
            distance_matrix, dtype=float
        )
        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index

        # Vehicle constraints
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        # QPSO parameters — adaptive β range
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.beta_max = max(beta + 0.25, 1.0)  # Start high for exploration
        self.beta_min = max(beta - 0.25, 0.4)  # End low for exploitation

        # Stagnation control
        self.stagnation_limit = max(20, max_iterations // 5)

        # Random seed
        random.seed(seed)
        np.random.seed(seed)

        # Number of customers
        self.dimension = len(customer_indices)

    # ========================================================
    # DECODE PARTICLE
    # ========================================================

    def decode(self, position):
        """
        Convert a random-key particle into VRP routes.
        Sorting gives customer visiting order, then
        customers are divided according to vehicle capacity.
        """
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
        """Calculate total distance of one vehicle route."""
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
    # FITNESS FUNCTION
    # ========================================================

    def fitness(self, position):
        """
        Calculate the fitness of a particle.
        Lower distance = better solution.
        Penalties for exceeding vehicles or capacity.
        """
        routes, feasible = self.decode(position)

        total_distance = sum(
            self.route_distance(route)
            for route in routes
        )

        penalty = 0.0

        # Vehicle-count penalty
        if not feasible:
            excess_routes = max(
                0, len(routes) - self.num_vehicles
            )
            penalty += (
                excess_routes
                * self.vehicle_capacity
                * 1000
            )

        # Capacity penalty
        for route in routes:
            load = sum(
                self.demands[c] for c in route
            )
            if load > self.vehicle_capacity:
                penalty += (
                    load - self.vehicle_capacity
                ) * 10000

        return total_distance + penalty

    # ========================================================
    # NEAREST-NEIGHBOR SEEDED PARTICLE
    # ========================================================

    def _nn_seeded_particle(self):
        """
        Create a particle using nearest-neighbor heuristic.
        Produces a greedy customer ordering, then converts
        to random-key representation.
        """
        unvisited = set(range(self.dimension))
        order = []

        # Start from a random customer
        current_idx = random.choice(list(unvisited))
        unvisited.remove(current_idx)
        order.append(current_idx)

        current_customer = self.customer_indices[current_idx]

        while unvisited:
            # Find nearest unvisited
            best_idx = None
            best_dist = float("inf")

            for idx in unvisited:
                cust = self.customer_indices[idx]
                d = self.distance_matrix[current_customer, cust]
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            unvisited.remove(best_idx)
            order.append(best_idx)
            current_customer = self.customer_indices[best_idx]

        # Convert order to random-key: position[order[i]] should
        # sort to give the NN order
        position = np.zeros(self.dimension)
        for rank, idx in enumerate(order):
            # Evenly spaced + small noise to avoid ties
            position[idx] = (rank + 0.5) / self.dimension
            position[idx] += np.random.uniform(-0.01, 0.01)

        return np.clip(position, 0, 1)

    # ========================================================
    # 2-OPT LOCAL SEARCH
    # ========================================================

    def _two_opt_route(self, route):
        """Apply 2-opt improvement to a single route."""
        if len(route) < 3:
            return route

        improved = True
        best_route = list(route)
        best_dist = self.route_distance(best_route)

        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    new_route = (
                        best_route[:i]
                        + best_route[i:j+1][::-1]
                        + best_route[j+1:]
                    )
                    new_dist = self.route_distance(new_route)
                    if new_dist < best_dist - 1e-10:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def _apply_two_opt(self, position):
        """
        Decode particle, apply 2-opt to each route,
        then re-encode back to random-key.
        """
        routes, feasible = self.decode(position)

        # Apply 2-opt to each route
        improved_routes = []
        for route in routes:
            improved_routes.append(self._two_opt_route(route))

        # Re-encode: build a flat customer order from improved routes
        flat_order = []
        for route in improved_routes:
            flat_order.extend(route)

        # Map customer indices back to dimension indices
        cust_to_dim = {
            c: i for i, c in enumerate(self.customer_indices)
        }

        new_position = np.zeros(self.dimension)
        for rank, cust in enumerate(flat_order):
            dim_idx = cust_to_dim[cust]
            new_position[dim_idx] = (rank + 0.5) / self.dimension

        return np.clip(new_position, 0, 1)

    # ========================================================
    # QPSO OPTIMIZATION
    # ========================================================

    def optimize(self):
        """
        Run Enhanced Quantum-Inspired Particle Swarm Optimization.
        Returns: (routes, best_distance, iterations, history)
        """
        if self.dimension == 0:
            return [], 0.0, 0, []

        # --- Initialize particles ---
        # 80% random, 20% nearest-neighbor seeded
        num_seeded = max(1, self.num_particles // 5)
        num_random = self.num_particles - num_seeded

        random_particles = np.random.uniform(
            0, 1, (num_random, self.dimension)
        )

        seeded_particles = np.array([
            self._nn_seeded_particle()
            for _ in range(num_seeded)
        ])

        particles = np.vstack([seeded_particles, random_particles])

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

        # Stagnation counter
        stagnation_count = 0
        prev_best = global_best_fitness

        # Main QPSO loop
        for iteration in range(self.max_iterations):
            # --- Adaptive β: linear decay ---
            progress = iteration / max(1, self.max_iterations - 1)
            beta = self.beta_max - (self.beta_max - self.beta_min) * progress

            # Mean best position
            mbest = np.mean(personal_best, axis=0)

            for i in range(self.num_particles):
                # Random coefficient
                phi = np.random.uniform(
                    0, 1, self.dimension
                )

                # Local attractor
                attractor = (
                    phi * personal_best[i]
                    + (1 - phi) * global_best
                )

                # Random value
                u = np.random.uniform(
                    0.000001, 1, self.dimension
                )

                # Random direction
                sign = np.where(
                    np.random.random(self.dimension) < 0.5,
                    -1, 1
                )

                # QPSO position update
                new_position = (
                    attractor
                    + sign * beta
                    * np.abs(mbest - particles[i])
                    * np.log(1 / u)
                )

                # Clamp to [0, 1]
                new_position = np.clip(new_position, 0, 1)
                particles[i] = new_position

                # Evaluate
                current_fitness = self.fitness(new_position)

                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = new_position.copy()
                    personal_best_fitness[i] = current_fitness

                # Update global best
                if current_fitness < global_best_fitness:
                    global_best = new_position.copy()
                    global_best_fitness = current_fitness

            # --- Stagnation detection & partial restart ---
            if abs(global_best_fitness - prev_best) < 1e-10:
                stagnation_count += 1
            else:
                stagnation_count = 0
                prev_best = global_best_fitness

            if stagnation_count >= self.stagnation_limit:
                # Reinitialize worst 30% of particles
                num_restart = max(1, self.num_particles * 3 // 10)
                worst_indices = np.argsort(personal_best_fitness)[-num_restart:]

                for idx in worst_indices:
                    if random.random() < 0.5:
                        particles[idx] = self._nn_seeded_particle()
                    else:
                        particles[idx] = np.random.uniform(0, 1, self.dimension)
                    personal_best[idx] = particles[idx].copy()
                    personal_best_fitness[idx] = self.fitness(particles[idx])

                stagnation_count = 0

            history.append(global_best_fitness)

        # --- Post-optimization: 2-opt local search ---
        refined_position = self._apply_two_opt(global_best)
        refined_fitness = self.fitness(refined_position)

        if refined_fitness < global_best_fitness:
            global_best = refined_position
            global_best_fitness = refined_fitness
            history.append(global_best_fitness)

        # Decode final best
        routes, feasible = self.decode(global_best)

        return (
            routes,
            global_best_fitness,
            len(history) - 1,
            history
        )