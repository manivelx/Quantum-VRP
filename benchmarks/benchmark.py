import time
from typing import Callable, Dict, Any


class BenchmarkResult:
    """
    Stores the result of one optimization algorithm.
    """

    def __init__(
        self,
        algorithm_name: str,
        routes,
        distance: float,
        runtime: float,
        iterations: int = 0
    ):
        self.algorithm_name = algorithm_name
        self.routes = routes
        self.distance = distance
        self.runtime = runtime
        self.iterations = iterations

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert benchmark result into a dictionary.
        """

        return {
            "Algorithm": self.algorithm_name,
            "Total Distance (km)": round(
                self.distance,
                2
            ),
            "Runtime (seconds)": round(
                self.runtime,
                4
            ),
            "Iterations": self.iterations,
            "Vehicles Used": len(self.routes)
        }


class BenchmarkEngine:
    """
    Common benchmark engine for FLUXROUTE.

    It allows different optimization algorithms
    to be evaluated using the same VRP problem.

    Each algorithm must provide a callable that
    returns:

        routes, distance, iterations
    """

    def __init__(self):
        self.results = []

    # --------------------------------------------------------
    # RUN ONE ALGORITHM
    # --------------------------------------------------------

    def run_algorithm(
        self,
        algorithm_name: str,
        optimizer_function: Callable,
    ) -> BenchmarkResult:
        """
        Run one optimization algorithm and record:

        - solution routes
        - total distance
        - runtime
        - iterations
        """

        start_time = time.time()

        result = optimizer_function()

        runtime = time.time() - start_time

        # ----------------------------------------------------
        # Support different return formats
        # ----------------------------------------------------

        if isinstance(result, tuple):

            if len(result) == 4:

                routes, distance, iterations, history = result

            elif len(result) == 3:

                routes, distance, iterations = result
                history = []

            elif len(result) == 2:

                routes, distance = result
                iterations = 0
                history = []

            else:

                raise ValueError(
                    "Optimizer must return "
                    "(routes, distance) or "
                    "(routes, distance, iterations)."
                )

        else:

            raise ValueError(
                "Optimizer function must return "
                "a tuple containing the solution."
            )

        benchmark_result = BenchmarkResult(

            algorithm_name=algorithm_name,

            routes=routes,

            distance=float(distance),

            runtime=float(runtime),

            iterations=int(iterations)

        )

        self.results.append(
            benchmark_result
        )

        return benchmark_result

    # --------------------------------------------------------
    # GET RESULTS
    # --------------------------------------------------------

    def get_results(self):
        """
        Return all benchmark results.
        """

        return self.results

    # --------------------------------------------------------
    # CONVERT RESULTS TO TABLE
    # --------------------------------------------------------

    def results_dataframe(self):
        """
        Convert benchmark results into a pandas
        DataFrame for Streamlit visualization.
        """

        import pandas as pd

        if not self.results:

            return pd.DataFrame(
                columns=[
                    "Algorithm",
                    "Total Distance (km)",
                    "Runtime (seconds)",
                    "Iterations",
                    "Vehicles Used"
                ]
            )

        return pd.DataFrame(
            [
                result.to_dict()
                for result in self.results
            ]
        )

    # --------------------------------------------------------
    # BEST ALGORITHM
    # --------------------------------------------------------

    def best_result(self):
        """
        Return the algorithm with the shortest
        total route distance.
        """

        if not self.results:

            return None

        return min(
            self.results,
            key=lambda result: result.distance
        )

    # --------------------------------------------------------
    # CLEAR RESULTS
    # --------------------------------------------------------

    def clear(self):
        """
        Remove previous benchmark results.
        """

        self.results.clear()