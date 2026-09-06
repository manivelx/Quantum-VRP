# navigation/eta.py

def calculate_eta(
    distance_km,
    average_speed_kmh=30.0
):
    """
    Calculate estimated travel time.

    Parameters
    ----------
    distance_km : float
        Route distance in kilometers.

    average_speed_kmh : float
        Average vehicle speed in km/h.

    Returns
    -------
    float
        Estimated travel time in minutes.
    """

    if distance_km <= 0:
        return 0.0

    if average_speed_kmh <= 0:
        raise ValueError(
            "Average speed must be greater than zero."
        )

    time_hours = (
        distance_km
        / average_speed_kmh
    )

    time_minutes = (
        time_hours * 60
    )

    return time_minutes


def format_eta(minutes):
    """
    Convert ETA in minutes into
    a human-readable format.
    """

    minutes = float(minutes)

    if minutes < 1:
        return "< 1 min"

    total_minutes = round(minutes)

    hours = total_minutes // 60
    remaining_minutes = total_minutes % 60

    if hours > 0:

        if remaining_minutes > 0:

            return (
                f"{hours} hr "
                f"{remaining_minutes} min"
            )

        return f"{hours} hr"

    return f"{total_minutes} min"