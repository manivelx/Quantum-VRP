import random
import numpy as np


class ALNS_VRP:
    """
    Adaptive Large Neighborhood Search (ALNS)
    for the Capacitated Vehicle Routing Problem (CVRP).

    Representation:
        routes = [
            [customer1, customer2, ...],
            [customer3, customer4, ...]
        ]

    ALNS repeatedly:
        1. Destroys part of the current solution
        2. Repairs the solution
        3. Evaluates the new solution
        4. Adapts destroy/repair operator weights
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
        destroy_fraction=0.2,
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
        self.destroy_fraction = destroy_fraction

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

        distance = self.total_distance(
            routes
        )

        penalty = 0.0

        # Too many vehicles
        if len(routes) > self.num_vehicles:

            penalty += (
                len(routes)
                - self.num_vehicles
            ) * self.vehicle_capacity * 1000

        # Capacity violation
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
    # CREATE INITIAL SOLUTION
    # ========================================================

    def create_initial_solution(self):

        customers = (
            self.customer_indices.copy()
        )

        random.shuffle(customers)

        routes = []

        current_route = []
        current_load = 0.0

        for customer in customers:

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
    # COPY SOLUTION
    # ========================================================

    def copy_solution(self, routes):

        return [
            route.copy()
            for route in routes
        ]

    # ========================================================
    # REMOVE CUSTOMER
    # ========================================================

    def remove_customer(
        self,
        routes,
        customer
    ):

        for route in routes:

            if customer in route:

                route.remove(
                    customer
                )

                return

    # ========================================================
    # RANDOM DESTROY
    # ========================================================

    def random_destroy(
        self,
        routes,
        num_remove
    ):

        destroyed = self.copy_solution(
            routes
        )

        all_customers = [
            customer
            for route in destroyed
            for customer in route
        ]

        num_remove = min(
            num_remove,
            len(all_customers)
        )

        removed = random.sample(
            all_customers,
            num_remove
        )

        for customer in removed:

            self.remove_customer(
                destroyed,
                customer
            )

        # Remove empty routes
        destroyed = [
            route
            for route in destroyed
            if route
        ]

        return destroyed, removed

    # ========================================================
    # WORST ROUTE DESTROY
    # ========================================================

    def worst_destroy(
        self,
        routes,
        num_remove
    ):

        destroyed = self.copy_solution(
            routes
        )

        customer_costs = []

        for route in destroyed:

            for index, customer in enumerate(
                route
            ):

                previous = (
                    self.depot_index
                    if index == 0
                    else route[index - 1]
                )

                next_node = (
                    self.depot_index
                    if index == len(route) - 1
                    else route[index + 1]
                )

                cost = (
                    self.distance_matrix[
                        previous,
                        customer
                    ]
                    +
                    self.distance_matrix[
                        customer,
                        next_node
                    ]
                )

                customer_costs.append(
                    (customer, cost)
                )

        customer_costs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        num_remove = min(
            num_remove,
            len(customer_costs)
        )

        removed = [
            customer
            for customer, _ in
            customer_costs[:num_remove]
        ]

        for customer in removed:

            self.remove_customer(
                destroyed,
                customer
            )

        destroyed = [
            route
            for route in destroyed
            if route
        ]

        return destroyed, removed

    # ========================================================
    # RANDOM REPAIR
    # ========================================================

    def random_repair(
        self,
        routes,
        removed
    ):

        repaired = self.copy_solution(
            routes
        )

        random.shuffle(
            removed
        )

        for customer in removed:

            best_route = None
            best_position = None
            best_cost = float("inf")

            for route_index, route in enumerate(
                repaired
            ):

                current_load = (
                    self.route_load(route)
                )

                if (
                    current_load
                    + self.demands[customer]
                    > self.vehicle_capacity
                ):

                    continue

                for position in range(
                    len(route) + 1
                ):

                    new_route = (
                        route.copy()
                    )

                    new_route.insert(
                        position,
                        customer
                    )

                    old_cost = (
                        self.route_distance(
                            route
                        )
                    )

                    new_cost = (
                        self.route_distance(
                            new_route
                        )
                    )

                    insertion_cost = (
                        new_cost
                        - old_cost
                    )

                    if (
                        insertion_cost
                        < best_cost
                    ):

                        best_cost = (
                            insertion_cost
                        )

                        best_route = (
                            route_index
                        )

                        best_position = (
                            position
                        )

            if best_route is not None:

                repaired[
                    best_route
                ].insert(
                    best_position,
                    customer
                )

            else:

                # Create another route
                repaired.append(
                    [customer]
                )

        return repaired

    # ========================================================
    # REGRET REPAIR
    # ========================================================

    def regret_repair(
        self,
        routes,
        removed
    ):

        repaired = self.copy_solution(
            routes
        )

        remaining = removed.copy()

        while remaining:

            best_customer = None
            best_route = None
            best_position = None
            highest_regret = -float("inf")

            for customer in remaining:

                insertion_options = []

                for route_index, route in enumerate(
                    repaired
                ):

                    current_load = (
                        self.route_load(route)
                    )

                    if (
                        current_load
                        + self.demands[customer]
                        > self.vehicle_capacity
                    ):

                        continue

                    for position in range(
                        len(route) + 1
                    ):

                        new_route = (
                            route.copy()
                        )

                        new_route.insert(
                            position,
                            customer
                        )

                        cost = (
                            self.route_distance(
                                new_route
                            )
                            -
                            self.route_distance(
                                route
                            )
                        )

                        insertion_options.append(
                            (
                                cost,
                                route_index,
                                position
                            )
                        )

                if not insertion_options:
                    continue

                insertion_options.sort(
                    key=lambda x: x[0]
                )

                best_cost = (
                    insertion_options[0][0]
                )

                if len(insertion_options) > 1:

                    second_cost = (
                        insertion_options[1][0]
                    )

                else:

                    second_cost = (
                        best_cost
                        + abs(best_cost)
                        + 1
                    )

                regret = (
                    second_cost
                    - best_cost
                )

                if regret > highest_regret:

                    highest_regret = regret

                    best_customer = customer

                    best_route = (
                        insertion_options[0][1]
                    )

                    best_position = (
                        insertion_options[0][2]
                    )

            if best_customer is None:

                # No feasible insertion.
                # Create a new route.

                customer = remaining.pop(
                    random.randrange(
                        len(remaining)
                    )
                )

                repaired.append(
                    [customer]
                )

            else:

                repaired[
                    best_route
                ].insert(
                    best_position,
                    best_customer
                )

                remaining.remove(
                    best_customer
                )

        return repaired

    # ========================================================
    # ADAPTIVE DESTROY OPERATOR
    # ========================================================

    def select_destroy_operator(
        self,
        weights
    ):

        return random.choices(
            [
                "random",
                "worst"
            ],
            weights=weights,
            k=1
        )[0]

    # ========================================================
    # ADAPTIVE REPAIR OPERATOR
    # ========================================================

    def select_repair_operator(
        self,
        weights
    ):

        return random.choices(
            [
                "random",
                "regret"
            ],
            weights=weights,
            k=1
        )[0]

    # ========================================================
    # OPTIMIZATION
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

        best_fitness = (
            current_fitness
        )

        # ----------------------------------------------------
        # Operator weights
        # ----------------------------------------------------

        destroy_weights = [
            1.0,
            1.0
        ]

        repair_weights = [
            1.0,
            1.0
        ]

        # ----------------------------------------------------
        # ALNS parameters
        # ----------------------------------------------------

        reaction_factor = 0.2

        # ----------------------------------------------------
        # Main ALNS loop
        # ----------------------------------------------------

        for iteration in range(
            self.max_iterations
        ):

            num_customers = len(
                self.customer_indices
            )

            num_remove = max(
                1,
                int(
                    num_customers
                    * self.destroy_fraction
                )
            )

            # ------------------------------------------------
            # Select destroy operator
            # ------------------------------------------------

            destroy_operator = (
                self.select_destroy_operator(
                    destroy_weights
                )
            )

            # ------------------------------------------------
            # Destroy
            # ------------------------------------------------

            if destroy_operator == "random":

                destroyed, removed = (
                    self.random_destroy(
                        current_solution,
                        num_remove
                    )
                )

            else:

                destroyed, removed = (
                    self.worst_destroy(
                        current_solution,
                        num_remove
                    )
                )

            # ------------------------------------------------
            # Select repair operator
            # ------------------------------------------------

            repair_operator = (
                self.select_repair_operator(
                    repair_weights
                )
            )

            # ------------------------------------------------
            # Repair
            # ------------------------------------------------

            if repair_operator == "random":

                candidate = (
                    self.random_repair(
                        destroyed,
                        removed
                    )
                )

            else:

                candidate = (
                    self.regret_repair(
                        destroyed,
                        removed
                    )
                )

            candidate_fitness = (
                self.fitness(candidate)
            )

            # ------------------------------------------------
            # Adaptive scoring
            # ------------------------------------------------

            score = 0.0

            if candidate_fitness < best_fitness:

                score = 5.0

            elif candidate_fitness < current_fitness:

                score = 3.0

            else:

                score = 1.0

            # ------------------------------------------------
            # Update destroy weight
            # ------------------------------------------------

            if destroy_operator == "random":

                destroy_index = 0

            else:

                destroy_index = 1

            destroy_weights[
                destroy_index
            ] = (
                (1 - reaction_factor)
                * destroy_weights[
                    destroy_index
                ]
                +
                reaction_factor
                * score
            )

            # ------------------------------------------------
            # Update repair weight
            # ------------------------------------------------

            if repair_operator == "random":

                repair_index = 0

            else:

                repair_index = 1

            repair_weights[
                repair_index
            ] = (
                (1 - reaction_factor)
                * repair_weights[
                    repair_index
                ]
                +
                reaction_factor
                * score
            )

            # ------------------------------------------------
            # Acceptance
            # ------------------------------------------------

            if (
                candidate_fitness
                <= current_fitness
            ):

                current_solution = (
                    candidate
                )

                current_fitness = (
                    candidate_fitness
                )

            # ------------------------------------------------
            # Update best solution
            # ------------------------------------------------

            if (
                candidate_fitness
                < best_fitness
            ):

                best_solution = (
                    self.copy_solution(
                        candidate
                    )
                )

                best_fitness = (
                    candidate_fitness
                )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return (
            best_solution,
            best_fitness,
            self.max_iterations
        )