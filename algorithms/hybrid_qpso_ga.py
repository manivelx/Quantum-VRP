import random

import numpy as np

from algorithms.ga import GeneticAlgorithmVRP
from algorithms.qpso import QPSO_VRP


class HybridQPSOGA:
    """Hybrid optimizer that seeds a GA with an elite QPSO solution."""

    def __init__(
        self,
        distance_matrix,
        demands,
        customer_indices,
        depot_index,
        vehicle_capacity,
        num_vehicles,
        num_particles=30,
        population_size=30,
        max_iterations=100,
        beta=0.75,
        mutation_rate=0.1,
        seed=42
    ):
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.demands = demands
        self.customer_indices = customer_indices
        self.depot_index = depot_index
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        self.num_particles = num_particles
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.beta = beta
        self.mutation_rate = mutation_rate
        self.seed = seed

    def optimize(self):
        """Run QPSO and GA operators against one shared population."""
        qpso = QPSO_VRP(
            distance_matrix=self.distance_matrix,
            demands=self.demands,
            customer_indices=self.customer_indices,
            depot_index=self.depot_index,
            vehicle_capacity=self.vehicle_capacity,
            num_vehicles=self.num_vehicles,
            num_particles=self.num_particles,
            max_iterations=self.max_iterations,
            beta=self.beta,
            seed=self.seed,
        )
        ga = GeneticAlgorithmVRP(
            distance_matrix=self.distance_matrix,
            demands=self.demands,
            customer_indices=self.customer_indices,
            depot_index=self.depot_index,
            vehicle_capacity=self.vehicle_capacity,
            num_vehicles=self.num_vehicles,
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            mutation_rate=self.mutation_rate,
            seed=self.seed,
        )

        if qpso.dimension == 0:
            return [], 0.0, 0, []

        num_seeded = max(1, self.num_particles // 5)
        particles = np.vstack([
            np.array([
                qpso._nn_seeded_particle()
                for _ in range(num_seeded)
            ]),
            np.random.uniform(
                0, 1, (self.num_particles - num_seeded, qpso.dimension)
            )
        ])
        personal_best = particles.copy()
        personal_best_fitness = np.array([
            qpso.fitness(particle) for particle in particles
        ])
        best_index = np.argmin(personal_best_fitness)
        global_best = personal_best[best_index].copy()
        global_fitness = personal_best_fitness[best_index]
        history = [global_fitness]

        def encode(solution):
            position = np.zeros(qpso.dimension)
            customer_to_dimension = {
                customer: index
                for index, customer in enumerate(qpso.customer_indices)
            }
            for rank, customer in enumerate(solution):
                position[customer_to_dimension[customer]] = (
                    rank + 0.5
                ) / qpso.dimension
            return position

        def decode_solution(position):
            routes, _ = qpso.decode(position)
            return [customer for route in routes for customer in route]

        incumbent_solution = decode_solution(global_best)

        for iteration in range(self.max_iterations):
            progress = iteration / max(1, self.max_iterations - 1)
            beta = qpso.beta_max - (
                qpso.beta_max - qpso.beta_min
            ) * progress
            mbest = np.mean(personal_best, axis=0)

            # QPSO explores the shared candidate pool.
            for index in range(self.num_particles):
                phi = np.random.uniform(0, 1, qpso.dimension)
                attractor = (
                    phi * personal_best[index]
                    + (1 - phi) * global_best
                )
                random_values = np.random.uniform(
                    0.000001, 1, qpso.dimension
                )
                direction = np.where(
                    np.random.random(qpso.dimension) < 0.5,
                    -1,
                    1,
                )
                particles[index] = np.clip(
                    attractor
                    + direction * beta
                    * np.abs(mbest - particles[index])
                    * np.log(1 / random_values),
                    0,
                    1,
                )
                fitness = qpso.fitness(particles[index])
                if fitness < personal_best_fitness[index]:
                    personal_best[index] = particles[index].copy()
                    personal_best_fitness[index] = fitness
                if fitness < global_fitness:
                    global_best = particles[index].copy()
                    global_fitness = fitness

            # GA recombines the same candidates and returns them to QPSO.
            population = [
                [
                    qpso.customer_indices[index]
                    for index in np.argsort(particle)
                ]
                for particle in particles
            ]
            population.insert(0, incumbent_solution.copy())
            population.sort(key=ga.fitness)
            population = population[:self.population_size]
            while len(population) < self.population_size:
                candidate = population[0].copy()
                random.shuffle(candidate)
                population.append(candidate)

            next_population = [population[0].copy()]
            while len(next_population) < self.population_size:
                parent1 = ga.selection(population)
                parent2 = ga.selection(population)
                child = ga.crossover(parent1, parent2)
                next_population.append(ga.mutate(child))

            # Improve a few elite permutations with route-level 2-opt.
            for elite_index in range(min(3, len(next_population))):
                elite_position = encode(next_population[elite_index])
                refined_position = qpso._apply_two_opt(elite_position)
                if qpso.fitness(refined_position) < ga.fitness(
                    next_population[elite_index]
                ):
                    next_population[elite_index] = decode_solution(
                        refined_position
                    )

            population = next_population
            candidate_positions = [
                encode(solution) for solution in population
            ]
            for index, particle in enumerate(candidate_positions):
                fitness = qpso.fitness(particle)
                if fitness < personal_best_fitness[index % self.num_particles]:
                    personal_best[index % self.num_particles] = particle.copy()
                    personal_best_fitness[index % self.num_particles] = fitness
                if fitness < global_fitness:
                    global_best = particle.copy()
                    global_fitness = fitness

            particles = np.array(candidate_positions[:self.num_particles])
            while len(particles) < self.num_particles:
                particles = np.vstack([particles, particles[0]])

            incumbent_solution = decode_solution(global_best)
            history.append(global_fitness)

        # Apply the same route-level refinement used by standalone QPSO.
        refined_position = qpso._apply_two_opt(global_best)
        refined_fitness = qpso.fitness(refined_position)
        if refined_fitness < global_fitness:
            global_best = refined_position
            global_fitness = refined_fitness
            history.append(global_fitness)

        routes, _ = qpso.decode(global_best)
        return routes, global_fitness, self.max_iterations, history