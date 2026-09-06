import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_FILE = "benchmark_comparison.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not os.path.exists(RESULT_FILE):

        raise FileNotFoundError(
            f"Result file not found: {RESULT_FILE}\n"
            "Run compare_algorithms.py first."
        )

    df = pd.read_csv(
        RESULT_FILE
    )

    required_columns = [
        "Algorithm",
        "Total Distance (km)",
        "Runtime (seconds)",
        "Iterations",
        "Vehicles Used"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# DISPLAY DATA
# ============================================================

def display_results(df):

    print()
    print("=" * 60)
    print("BENCHMARK COMPARISON DATA")
    print("=" * 60)

    print()

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================
# DISTANCE COMPARISON
# ============================================================

def plot_distance(df):

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        df["Algorithm"],
        df["Total Distance (km)"]
    )

    plt.title(
        "Algorithm Comparison - Total Distance"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Total Distance (km)"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        "algorithm_distance_comparison.png",
        dpi=300
    )

    plt.show()


# ============================================================
# RUNTIME COMPARISON
# ============================================================

def plot_runtime(df):

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        df["Algorithm"],
        df["Runtime (seconds)"]
    )

    plt.title(
        "Algorithm Comparison - Runtime"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Runtime (seconds)"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        "algorithm_runtime_comparison.png",
        dpi=300
    )

    plt.show()


# ============================================================
# ITERATION COMPARISON
# ============================================================

def plot_iterations(df):

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        df["Algorithm"],
        df["Iterations"]
    )

    plt.title(
        "Algorithm Comparison - Iterations"
    )

    plt.xlabel(
        "Algorithm"
    )

    plt.ylabel(
        "Iterations"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()

    plt.savefig(
        "algorithm_iterations_comparison.png",
        dpi=300
    )

    plt.show()


# ============================================================
# BEST ALGORITHM
# ============================================================

def find_best_algorithm(df):

    best_index = df[
        "Total Distance (km)"
    ].idxmin()

    best = df.loc[
        best_index
    ]

    print()
    print("=" * 60)
    print("BEST ALGORITHM")
    print("=" * 60)

    print()

    print(
        f"Algorithm : {best['Algorithm']}"
    )

    print(
        f"Distance  : "
        f"{best['Total Distance (km)']:.3f} km"
    )

    print(
        f"Runtime   : "
        f"{best['Runtime (seconds)']:.4f} seconds"
    )

    print(
        f"Iterations: "
        f"{int(best['Iterations'])}"
    )

    print(
        f"Vehicles  : "
        f"{int(best['Vehicles Used'])}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("FLUXROUTE - RESULT VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load benchmark results
    # --------------------------------------------------------

    df = load_results()

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_results(
        df
    )

    # --------------------------------------------------------
    # Find best algorithm
    # --------------------------------------------------------

    find_best_algorithm(
        df
    )

    # --------------------------------------------------------
    # Generate charts
    # --------------------------------------------------------

    print()
    print("Generating distance comparison...")

    plot_distance(
        df
    )

    print()
    print("Generating runtime comparison...")

    plot_runtime(
        df
    )

    print()
    print("Generating iteration comparison...")

    plot_iterations(
        df
    )

    # --------------------------------------------------------
    # Completed
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)

    print()
    print(
        "Generated files:"
    )

    print(
        "1. algorithm_distance_comparison.png"
    )

    print(
        "2. algorithm_runtime_comparison.png"
    )

    print(
        "3. algorithm_iterations_comparison.png"
    )

    print()