import random


class GeneticAlgorithmVRP:

    def __init__(
        self,
        distance_matrix,
        demands,
        customer_indices,
        depot_index,
        vehicle_capacity,
        num_vehicles,
        population_size=30,
        max_iterations=100,
        mutation_rate=0.1,
        seed=42
    ):

        self.distance_matrix = distance_matrix
        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index

        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles

        self.population_size = population_size
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate

        random.seed(seed)

    # ========================================================
    # CREATE INITIAL SOLUTION
    # ========================================================

    def create_solution(self):

        solution = self.customer_indices.copy()

        random.shuffle(solution)

        return solution

    # ========================================================
    # DECODE SOLUTION
    # ========================================================

    def decode(self, solution):

        routes = []

        current_route = []
        current_load = 0

        for customer in solution:

            demand = self.demands[customer]

            if (
                current_route
                and current_load + demand
                > self.vehicle_capacity
            ):

                routes.append(current_route)

                current_route = []
                current_load = 0

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
            return 0

        distance = 0

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
    # FITNESS
    # ========================================================

    def fitness(self, solution):

        routes, feasible = self.decode(solution)

        distance = sum(
            self.route_distance(route)
            for route in routes
        )

        penalty = 0

        if not feasible:

            excess = len(routes) - self.num_vehicles

            penalty += excess * 100000

        return distance + penalty

    # ========================================================
    # SELECTION
    # ========================================================

    def selection(self, population):

        tournament = random.sample(
            population,
            min(3, len(population))
        )

        return min(
            tournament,
            key=self.fitness
        )

    # ========================================================
    # CROSSOVER
    # ========================================================

    def crossover(self, parent1, parent2):

        size = len(parent1)

        start, end = sorted(
            random.sample(
                range(size),
                2
            )
        )

        child = [None] * size

        child[start:end] = parent1[start:end]

        remaining = [
            customer
            for customer in parent2
            if customer not in child
        ]

        pointer = 0

        for i in range(size):

            if child[i] is None:

                child[i] = remaining[pointer]

                pointer += 1

        return child

    # ========================================================
    # MUTATION
    # ========================================================

    def mutate(self, solution):

        if random.random() < self.mutation_rate:

            i, j = random.sample(
                range(len(solution)),
                2
            )

            solution[i], solution[j] = (
                solution[j],
                solution[i]
            )

        return solution

    # ========================================================
    # OPTIMIZE
    # ========================================================

    def optimize(self):

        population = [
            self.create_solution()
            for _ in range(
                self.population_size
            )
        ]

        best_solution = min(
            population,
            key=self.fitness
        )

        best_fitness = self.fitness(
            best_solution
        )

        for iteration in range(
            self.max_iterations
        ):

            new_population = []

            # Keep best solution
            new_population.append(
                best_solution.copy()
            )

            while len(new_population) < self.population_size:

                parent1 = self.selection(
                    population
                )

                parent2 = self.selection(
                    population
                )

                child = self.crossover(
                    parent1,
                    parent2
                )

                child = self.mutate(
                    child
                )

                new_population.append(
                    child
                )

            population = new_population

            current_best = min(
                population,
                key=self.fitness
            )

            current_fitness = self.fitness(
                current_best
            )

            if current_fitness < best_fitness:

                best_solution = (
                    current_best.copy()
                )

                best_fitness = current_fitness

        routes, feasible = self.decode(
            best_solution
        )

        return (
            routes,
            best_fitness,
            self.max_iterations
        )