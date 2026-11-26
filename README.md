# FluxRoute: quantum-inspired vehicle routing prototype

A Smart India Hackathon 2026 prototype (problem statement SIH26137) for
capacitated vehicle routing (CVRP), built by a team. It compares four
optimizers on the same problem and includes a Streamlit map app that adds
traffic scenarios on top of an OpenStreetMap road graph.

**Status:** hackathon prototype, cleaned up after the event. The optimizers
and the offline benchmark are tested. The Streamlit app needs a network
connection to download map data and has not been re-tested in this cleanup.

## What is and is not original

The optimizers are standard published algorithms applied to CVRP. They are
not new algorithms.

| Component | Source |
|---|---|
| Particle swarm optimization (PSO) | Kennedy and Eberhart, 1995 |
| Quantum-behaved PSO (QPSO) update rule | Sun, Feng and Xu, IEEE CEC 2004 |
| Genetic algorithm, tournament selection | Classical GA |
| Random-key encoding of routes | Bean, 1994 |
| 2-opt local search | Croes, 1958 |

"Quantum-inspired" means QPSO borrows a probability-cloud position update
from quantum mechanics. It runs on an ordinary computer and has no quantum
speedup.

What this repository adds: a CVRP decoder, nearest-neighbor seeding,
adaptive beta decay, stagnation restarts, a hybrid QPSO + GA loop,
scenario-based traffic weights, and a shared benchmark.

## Quick start (no map download, runs in seconds)

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m benchmarks.offline
```

The benchmark reads `data/sample_customers.csv` (30 customers around Pune),
uses haversine distance times 1.3 as a rough road-detour estimate, and runs
each algorithm over five seeds. Make your own instance with
`python -m benchmarks.offline --generate 40 --seed 7`.

### Sample result

30 customers, 3 vehicles with capacity 200, 5 seeds, 100 iterations, on one
laptop-class CPU. Distances are in km and cover all routes.

| Algorithm | Mean km | Best km | Mean km with 2-opt added | Mean seconds |
|---|---|---|---|---|
| QPSO | 49.06 | 46.69 | 49.06 | 0.08 |
| Hybrid QPSO + GA | 45.91 | 44.90 | 45.91 | 0.33 |
| PSO | 71.22 | 61.57 | 62.19 | 0.05 |
| Genetic Algorithm | 66.31 | 57.96 | 56.73 | 0.15 |

Read this carefully:

- QPSO and the hybrid include 2-opt local search. PSO and GA do not, so the
  raw columns compare different methods. The "2-opt added" column applies the
  same polish to every algorithm.
- This is one small instance with five seeds. It does not show that QPSO is
  better than PSO or GA in general.

## Why it was slow, and what changed

- 2-opt re-summed the full route for every candidate move. It now scores each
  move from the four edges it changes. Results are identical, and the
  tests check that against the old method. On 300 customers QPSO went from
  4.7 s to 0.7 s, and the hybrid from over 9 minutes to about 38 s.
- The old default input was a 1,000-customer file on a small-neighborhood
  road graph, which meant about 1,000 shortest-path runs plus an instance that
  3 vehicles cannot serve. That file is now `data/customers_stress_1000.csv`
  and is no longer the default.
- The hybrid is still the slowest optimizer. Its GA step uses plain Python
  lists and could be vectorized.

## Map app (needs network)

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

Traffic is scenario-based: the road-cost matrix is regenerated for a chosen
hour, weather and congestion level. It does not use a live traffic feed and
does not reroute a vehicle mid-delivery. `traffic/traffic_simulator.py` is
the place to connect a real feed.

## Layout

- `algorithms/`: QPSO, hybrid QPSO + GA, PSO, GA
- `benchmarks/offline.py`: fast offline benchmark; the other files here use the road graph
- `traffic/`: scenario traffic simulation and dynamic edge weights
- `graph/`, `routing/`, `navigation/`, `visualization/`: road graph and map code used by the app
- `vrp/`: CVRP data structures and validators
- `tests/`: pytest suite

## Contributors

Hackathon team project. Contributor names and roles to be added by the team.

## License

MIT, see `LICENSE`. Confirm all contributors agree before publishing.
