class ConstraintChecker:

    def __init__(self, problem):

        self.problem = problem

    # ========================================================
    # CAPACITY
    # ========================================================

    def capacity_violation(self, route):

        vehicle = self.problem.vehicles[
            route.vehicle_id
        ]

        load = 0.0

        for customer in route.customers:

            load += self.problem.demands[
                customer
            ]

        if load > vehicle.capacity:

            return load - vehicle.capacity

        return 0.0

    # ========================================================
    # CUSTOMER VISIT
    # ========================================================

    def customer_visit_violation(self, solution):

        expected = set(
            self.problem.customers
        )

        visited = solution.get_all_customers()

        violation = 0

        # Duplicate visits
        violation += (
            len(visited) - len(set(visited))
        )

        # Missing customers
        violation += len(
            expected - set(visited)
        )

        # Unknown customers
        violation += len(
            set(visited) - expected
        )

        return violation

    # ========================================================
    # ROUTE DISTANCE
    # ========================================================

    def route_distance_violation(self, route):

        vehicle = self.problem.vehicles[
            route.vehicle_id
        ]

        depot = self.problem.depot_id

        full_route = [
            depot,
            *route.customers,
            depot
        ]

        distance = 0.0

        for i in range(
            len(full_route) - 1
        ):

            source = full_route[i]

            destination = full_route[i + 1]

            distance += (
                self.problem.distance_matrix[
                    source
                ][destination]
            )

        if distance > vehicle.max_distance:

            return distance - vehicle.max_distance

        return 0.0

    # ========================================================
    # ROUTE TIME
    # ========================================================

    def route_time_violation(self, route):

        vehicle = self.problem.vehicles[
            route.vehicle_id
        ]

        depot = self.problem.depot_id

        full_route = [
            depot,
            *route.customers,
            depot
        ]

        time = 0.0

        for i in range(
            len(full_route) - 1
        ):

            source = full_route[i]

            destination = full_route[i + 1]

            time += (
                self.problem.time_matrix[
                    source
                ][destination]
            )

        for customer in route.customers:

            time += self.problem.service_time.get(
                customer,
                0.0
            )

        if time > vehicle.max_time:

            return time - vehicle.max_time

        return 0.0

    # ========================================================
    # TOTAL PENALTY
    # ========================================================

    def calculate_penalty(self, solution):

        penalty = 0.0

        # Customer visit penalty
        visit_violation = (
            self.customer_visit_violation(
                solution
            )
        )

        penalty += (
            visit_violation * 10000
        )

        # Route constraints
        for route in solution.routes:

            capacity = (
                self.capacity_violation(route)
            )

            distance = (
                self.route_distance_violation(
                    route
                )
            )

            route_time = (
                self.route_time_violation(
                    route
                )
            )

            penalty += capacity * 1000

            penalty += distance * 100

            penalty += route_time * 100

        return penalty