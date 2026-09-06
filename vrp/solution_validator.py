def validate_solution(
    routes,
    customer_indices,
    demands,
    vehicle_capacity,
    num_vehicles
):
    """
    Validate a VRP solution.

    Returns:
        {
            "valid": bool,
            "errors": list[str]
        }
    """

    errors = []

    # --------------------------------------------------------
    # Check number of vehicles
    # --------------------------------------------------------

    if len(routes) > num_vehicles:

        errors.append(
            f"Solution uses {len(routes)} vehicles "
            f"but only {num_vehicles} are available."
        )

    # --------------------------------------------------------
    # Customers appearing in solution
    # --------------------------------------------------------

    visited = []

    for route in routes:
        visited.extend(route)

    # --------------------------------------------------------
    # Check missing customers
    # --------------------------------------------------------

    expected = set(customer_indices)
    actual = set(visited)

    missing = expected - actual

    if missing:

        errors.append(
            "Missing customers: "
            + ", ".join(
                str(customer)
                for customer in sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Check duplicate customers
    # --------------------------------------------------------

    duplicates = [
        customer
        for customer in set(visited)
        if visited.count(customer) > 1
    ]

    if duplicates:

        errors.append(
            "Duplicate customers: "
            + ", ".join(
                str(customer)
                for customer in sorted(duplicates)
            )
        )

    # --------------------------------------------------------
    # Check individual vehicle capacity
    # --------------------------------------------------------

    for vehicle_number, route in enumerate(routes):

        load = sum(
            demands[customer]
            for customer in route
        )

        if load > vehicle_capacity:

            errors.append(
                f"Vehicle {vehicle_number + 1} "
                f"exceeds capacity: "
                f"{load:.2f} > "
                f"{vehicle_capacity:.2f} kg"
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }