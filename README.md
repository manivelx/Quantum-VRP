# BLACK ROCK: Intelligent Traffic Route Optimization

SIH26137 implementation for capacitated vehicle route optimization using
quantum-inspired and classical metaheuristics. The Streamlit application
optimizes routes on an OpenStreetMap road graph and can account for traffic
conditions that change the effective edge costs.

## Highlights

- Enhanced QPSO with adaptive beta decay, nearest-neighbor seeding, 2-opt
	refinement, and stagnation restarts.
- Dynamic traffic weighting from time of day, weather, and reproducible
	congestion events.
- Shared benchmark across QPSO, PSO, genetic algorithm, ACO, simulated
	annealing, tabu search, and ALNS.
- Capacity-aware CVRP routes with depot return and route-distance reporting.

## Run The App

Install dependencies, then start Streamlit from the repository root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

In the Route Optimization section, enable dynamic traffic conditions and
choose the hour, weather, and congestion probability. The selected traffic
scenario is applied to both individual optimization and the benchmark so the
results are comparable.

## Run The Benchmark

The comparison runner executes all seven algorithms on the same road-distance
matrix and writes `benchmark_comparison.csv` plus the best route output:

```powershell
python -m benchmarks.compare_algorithms
```

The QPSO result includes a convergence history. Its contraction-expansion
coefficient decays from an exploratory value toward an exploitative value,
and the final solution is refined with 2-opt. The benchmark engine accepts
both legacy three-value optimizer results and four-value results containing a
history.

## Project Layout

- `algorithms/`: QPSO, PSO, GA, ACO, SA, Tabu Search, and ALNS.
- `traffic/`: traffic simulation and dynamic edge-weight management.
- `benchmarks/`: reusable benchmark engine and runners.
- `graph/`: road graph loading, snapping, and routing.
- `vrp/`: CVRP data structures, constraints, and validation.