import subprocess
import sys


# ============================================================
# RUN QPSO
# ============================================================

print()
print("=" * 60)
print("1. RUNNING QPSO")
print("=" * 60)

subprocess.run(
    [
        sys.executable,
        "-m",
        "benchmarks.run_qpso"
    ],
    check=True
)


# ============================================================
# RUN PSO
# ============================================================

print()
print("=" * 60)
print("2. RUNNING PSO")
print("=" * 60)

subprocess.run(
    [
        sys.executable,
        "-m",
        "benchmarks.run_pso"
    ],
    check=True
)


# ============================================================
# RUN GENETIC ALGORITHM
# ============================================================

print()
print("=" * 60)
print("3. RUNNING GENETIC ALGORITHM")
print("=" * 60)

subprocess.run(
    [
        sys.executable,
        "-m",
        "benchmarks.run_ga"
    ],
    check=True
)


# ============================================================
# RUN COMPARISON
# ============================================================

print()
print("=" * 60)
print("4. RUNNING ALGORITHM COMPARISON")
print("=" * 60)

subprocess.run(
    [
        sys.executable,
        "-m",
        "benchmarks.compare_algorithms"
    ],
    check=True
)


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 60)
print("BLACK ROCK VRP - COMPLETE BENCHMARK")
print("=" * 60)

print()
print("QPSO        : COMPLETED")
print("PSO         : COMPLETED")
print("GA          : COMPLETED")
print("COMPARISON  : COMPLETED")

print()
print("All benchmark operations completed successfully.")
print("=" * 60)