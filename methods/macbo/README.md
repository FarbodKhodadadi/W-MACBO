# Waypoint-based memory-augmented circulation barrier optimization

This additive implementation uses the existing `robot_env` simulator, wheel inputs, robot footprint, clearance, and experiment format. No environment code changes are needed. Install the existing method dependencies with `python -m pip install -r methods/requirements.txt`.

```python
from robot_env import Environment
from methods import MACBO, MACBOConfig, make_method, run_method

env = Environment(bounds=(-1, 1, -1, 1))
env.add("U", width=.45, height=.55, thickness=.12, angle=.3)
env.reset((-.8, -.1, 0))
controller = make_method("macbo", (.8, .15, .7), grid_spacing=.2, k_c=.8)
experiment = run_method(env, controller, horizon=60)
print(experiment.metrics())
experiment.save("results/my_macbo_run")
# Equivalent: MACBO(goal, MACBOConfig(grid_spacing=.2, k_c=.8))
```

`run_method` copies the environment and resets controller memory by default. Its goal tolerances default to the controller's position and heading tolerances. Inspect `experiment.rollout.status` and `experiment.failure`; reaching a horizon is not success. `controller.waypoints`, `controller.stage_entries`, and `experiment.diagnostics` expose the plan and execution. The controller also works with `simulate`; apply exactly one wheel command per configured `dt`. Changing the map requires preparing/resetting again. A repeated call without advancing time is rejected rather than advancing memory twice.

## Manuscript fidelity and the differential-drive adaptation

`clf_mode="strict"` implements the supplied intermediate position CLF, final full-pose CLF, transformed time term, bounded slack, memory/circulation field, and hard component barriers. Heading differences use the local wrapped angular coordinate. The strict optimization can be infeasible even without obstacles: at `(0,0,0)` targeting `(0,.3,0)`, both forward/backward motion have zero instantaneous position-CLF derivative. Yet the finite-rate constraint requires a strictly negative derivative because `c_bar < c`. This is a limitation of the supplied formulation on this robot, not a solver issue. Strict mode reports `qp_infeasible` and applies no command.

The default `clf_mode="hybrid"` is an explicitly modified practical controller. It uses stationary heading CLFs before driving, waypoint-position CLFs while driving, and a stationary terminal heading CLF after arriving within the configured position tolerance. It adds a weighted yaw-tracking objective, clamps desired heading relative to waypoint bearing, and uses heading hysteresis. Memory and circulation persist across waypoint changes. Hard barriers and bounded finite-rate slack remain in every phase. This mode is not an exact reproduction of the manuscript's final full-state CLF and does not establish its theorem.

## Implementation map

- `geometry.py`: circumscribed circle polygons, exact polygon obstacles, convex decomposition preserving T/star/U cavities, projections, squared-clearance barriers, deterministic R/W/S obstacle selection and nearby components.
- `planner.py`: incoming-direction A*, axis edges, visible endpoint connections, Manhattan travel cost plus turn delay, admissible heuristic, clearance-certified shortening, deterministic refinement, explicit resource limits.
- `controller.py`: waypoint/goal/circulation fields; persistent tangent sign; exponential held-forcing memory update; physical wheel-input QP; analytic objective gradient; solver residual checks; independent whole-interval validation.
- `config.py`: immutable validated hyperparameters. Wheel decision variables are normalized for numerical conditioning; objectives and Lie derivatives retain physical units.

The safety radius is the environment's body radius plus clearance. `route_margin` adds planning clearance. Hybrid monitoring activates components one maximum translation step earlier. Circle approximation is conservative. Convex component barriers use unique Euclidean projections and `2*(p-projection)` gradients. Memory uses exact exponential integration of the sampled forcing; it approximates continuously changing forcing over each held-input interval.

## Hyperparameters

All parameters can be passed to `make_method("macbo", goal, **parameters)` or `MACBOConfig`. Units are meters, radians, and seconds.

| Group | Parameters |
|---|---|
| Route | `grid_spacing`, `max_edge_steps`, `nominal_speed`, `turn_delay`, `route_margin`, `shorten_route`, `refinement_factor`, `max_refinements`, `grid_max_nodes`, `max_expansions`, `circle_sides` |
| Field and memory | `k_w`, `k_g`, `k_c`, `beta`, `ell`, `lambda_memory`, `initial_memory`, `initial_sigma`, `epsilon_a`, `epsilon_w` |
| Barrier and CLF | `alpha`, `c`, `c_bar`, `q`, `epsilon_V`, `clf_mode` |
| Objective | `mu_v`, `mu_t`, `mu_u`, `mu_delta_u`, `mu_delta`, hybrid `mu_heading` |
| Hybrid heading | `heading_gain`, `max_turn_speed`, `heading_enter`, `heading_exit`, `max_field_heading_offset` |
| Execution | `dt`, `target_position_tolerance`, `target_heading_tolerance`, `max_iterations`, `solver_tolerance`, `feasibility_tolerance` |

Use a smaller grid for narrow passages; resource bounds prevent unbounded allocation. Large finite-rate gains or insufficient clearance can make the bounded-wheel QP infeasible. No failure silently changes the gains, removes barriers, substitutes another method, or rescales a feasible control. Unsafe sampled intervals stop with an explicit failure before application. Position tolerance must be set below the desired final positional accuracy; numerical arrival is not exact equality to a mathematical point.

## Reproducibility and validation

`experiment.save` records the initial scene, full configuration, numerical-library versions, goal, tolerances, commands, states, solver diagnostics, and interval acceptance. Load `scene.json` with `Environment.load`, recreate the method from `report.json`, and rerun its `run_options`. Exact trajectory replay is tested on the same numerical stack; runtime timings vary and cross-platform solver equality is not promised.

Run these suites from the project root:

```sh
python -m unittest discover -s tests
python -m unittest discover -s tests/methods
python -m unittest discover -s tests/macbo
```

The 24 MACBO tests cover equation/objective checks, A* versus Dijkstra, conservative geometry, concave cavities, failure reporting, repeated-call detection, configuration validation, reset/replay, QP inequalities, and navigation around every obstacle type, rotated obstacles, mixed maps, cavity exit, sideways/reversed/heading-only goals, and six seeded varied trials. Navigation tests independently evaluate exact unicycle positions at 21 points per accepted interval. In addition, every applied interval passes conservative polygon and original environment swept-motion guards. These tests provide evidence for the tested cases; arbitrary maps, infeasible goals, or parameter choices are not guaranteed to reach the goal. No finite test suite proves absence of all bugs.

See `../../notebooks/macbo_method.ipynb` for executed examples, plots, strict-mode behavior, and reproducible export/replay.
