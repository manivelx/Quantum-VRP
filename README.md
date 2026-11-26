# FluxRoute: Intelligent Traffic Route Optimization

SIH26137 implementation for capacitated vehicle route optimization using
quantum-inspired and classical metaheuristics. The Streamlit application
optimizes routes on an OpenStreetMap road graph and can account for traffic
conditions that change the effective edge costs.

## Highlights

- Enhanced QPSO with adaptive beta decay, nearest-neighbor seeding, 2-opt
	refinement, and stagnation restarts.
- Dynamic traffic weighting from time of day, weather, and reproducible
	congestion events.
- Hybrid QPSO + GA optimizer, combining quantum exploration with genetic
	refinement.
- Multi-objective routing that balances distance, traffic-adjusted time, and
	congestion exposure.
- Incident response mode that closes a live route segment and re-optimizes
	around the simulated road closure.
- Shared benchmark across QPSO, hybrid QPSO + GA, PSO, and genetic algorithm.
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

This is scenario-based dynamic routing: the road-cost matrix is regenerated
when an optimization or benchmark is started, and the optimizer then produces
a route for that traffic state. It does not yet consume a live traffic API or
reroute a vehicle during delivery. `TrafficSimulator` provides the simulation
boundary where live speed, incident, or weather feeds can be connected later.

Traffic inputs are validated before use. Hours wrap into `0..23`, supported
weather values are `clear`, `rain`, `heavy_rain`, and `fog`, and congestion is
sampled on directed non-diagonal road segments. Passing `seed=42` to
`DynamicWeights` makes a scenario reproducible.

## Run The Benchmark

The comparison runner executes all four algorithms on the same road-distance
matrix and writes `benchmark_comparison.csv` plus the best route output. The
comparison reports distance rank, runtime rank, and distance-per-runtime so
route quality is not confused with raw execution speed:

```powershell
python -m benchmarks.compare_algorithms
```

The QPSO result includes a convergence history. Its contraction-expansion
coefficient decays from an exploratory value toward an exploitative value,
and the final solution is refined with 2-opt. The benchmark engine accepts
both legacy three-value optimizer results and four-value results containing a
history.

### Sample benchmark results

From `benchmark_comparison.csv` (single run, 100 iterations, 3 vehicles):

| Algorithm | Distance (km) | Runtime (s) |
|---|---|---|
| QPSO | 10.25 | 0.0655 |
| Hybrid QPSO + GA | 10.25 | 0.2165 |
| PSO | 11.13 | 0.0435 |
| Genetic Algorithm | 10.46 | 0.0955 |

## Project Layout

- `algorithms/`: QPSO, hybrid QPSO + GA, PSO, and GA.
- `traffic/`: traffic simulation and dynamic edge-weight management.
- `benchmarks/`: reusable benchmark engine and runners.
- `graph/`: road graph loading, snapping, and routing.
- `vrp/`: CVRP data structures, constraints, and validation.