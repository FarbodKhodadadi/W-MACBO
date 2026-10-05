# Seven-method, four-map benchmark

Open `../notebooks/benchmarking.ipynb`, install `python -m pip install -r benchmarks/requirements.txt` from the project root, then restart the kernel and Run All. The delivered notebook defaults to `RECOMPUTE=False` to load completed fingerprint-validated trials; set `RECOMPUTE=True` for fresh controller runs and timings. The notebook runs all 28 trials, exports figures/tables/GIFs, runs regression suites, and replays MACBO's U + T trajectory. Execution takes several minutes, depending on hardware. Times New Roman must be installed; the notebook checks instead of silently falling back.

This package adds benchmark orchestration around the existing APIs. It does not modify the physical environment or change a failed method to another controller. Source attribution/adaptations are in `../methods/README.md` and `../methods/macbo/README.md`.

## Files and entry point

- `scenarios.py`: deterministic, shared geometry and endpoints.
- `config.py`: one parameter set per method, reused unchanged on all four maps.
- `runner.py`: `run_benchmark(output_directory)` returns scenes, lightweight trials, a pandas per-trial frame, four-map means, and work proxies; saves trials immediately.
- `metrics.py`: independent exact-motion sampling, physical-time averages, common objective, and explicit treatment of undefined means.
- `plotting.py`: Seaborn styling/plots, Times New Roman, 600 dpi PNG + vector PDF exports, GIF motion reconstruction, and pandas table styling.

```python
from benchmarks.runner import run_benchmark
scenes, trials, per_trial, means, work = run_benchmark('results/benchmarking')
```

Settings in `config.py` are development choices tuned using these same four maps. This is a deterministic comparison, not an independent tuning/test split or a statistical significance claim. Do not silently change individual maps or stop tolerances per method.

## Shared scenarios

All methods start at `(-2,-2,0)` and target `(2,2,pi/2)`. Bounds are `[-3,3]²`, providing room behind the U. They are finite planning bounds; the simulator does not treat them as physical walls. Physical horizon is 120 s, position tolerance .08 m, heading tolerance .2 rad. The robot retains the default .021 m wheel radius, .0884 m axle length, 10 rad/s wheel bounds, .07 m body radius, and .02 m clearance.

1. Three circles centered at `(-1,-1)`, `(0,0)`, `(1,1)`, with radii .25, .38, .29 m. Adjacent boundary gaps are .7842 and .7442 m, greater than the .18 m safety diameter.
2. U centered at the start: width/height 1.1 m, thickness .18 m, rotation 3pi/4. Opening faces southwest, away from the target. T near the goal at `(1.15,1.15)`, width 1 m, height 1.05 m, thickness .22 m, rotation -pi/4. A valid route initially increases goal distance before going forward.
3. Central five-point star: radius .95 m, inner radius .43 m, rotation pi/10.
4. Two .9 m squares rotated pi/4, centered at ±.63 times the diagonal normal. Their boundary gap is .36 m, with .18 m admissible centerline width after safety inflation. Going through the passage or around the squares is permitted.

The unchanged obstacle APIs preserve star/U concavity. Huang's CLF–CBF implementation uses enclosing circles and therefore loses U cavities; its U starting state is inside its conservative circle. This limitation is reported, not repaired by replacing its geometry.

## Configuration choices

All seven methods receive the same robot/map/start/goal/horizon/arrival tolerances. Integration intervals differ and are recorded: MACBO/CLF–CBF .05 s, RRT* .2 s, MPC .3 s, HJ .6 s. Time averages are weighted in physical seconds, never by sample count. Different intervals remain a numerical-design difference, not an exact same-discretization experiment.

- MACBO is the explicitly documented hybrid controller, not the manuscript's strict full-state CLF. The .10 m grid searches positive and negative axis directions. .005 rad pre-drive heading tolerance and stronger yaw tracking (`mu_heading=20`, field heading offset .02 rad) avoid accumulated lateral drift in the narrow passage. No CBF or bounded-slack constraints are removed. Original graph points and shortened waypoints are saved separately.
- RRT*: fixed seed, 600 samples, physical rotate/drive/rotate steering, reverse allowed. It plans and then executes its own wheel controls.
- NMPC/composite remain local, with `route_guidance=False`; DHOCBF retains its own A* guidance. Route assistance is a method difference identified in the notebook, not a hidden fallback. Local failures in the U/star are retained.
- HJ uses a 61x61x24 grid (.1 m planar spacing), nine controls, 200 steps at .6 s (120 s total), periodic headings, and a 512 MiB configured memory bound. Goal coordinates and heading are represented on this grid. Its 3 value tensors occupy approximately 205.4 MiB; transition/work arrays add memory. These numerical value sets are not certified continuous reach-avoid sets.

## Metric definitions

Controls are physical wheel speeds `[omega_R,omega_L]`, norm in rad/s. Trajectory norm means `||(x,y)||_2` in meters, relative to the coordinate origin; it is not a mixed position/angle norm. Mean goal distance is also exported. End position error is Euclidean; heading error is wrapped. Path length sums accepted position chords; it is not exact arc length.

Control means use exact held-input widths. Trajectory/goal-distance means use five-point Gauss–Legendre quadrature of independently reconstructed exact held-twist positions. Dense safety validation uses 21 positions per interval, in addition to the original whole-interval conservative guard. Recorded failed attempts are not included as applied inputs.

Successful runs receive a declared zero-input terminal hold at their final pose to 120 s for scoring and signal plots. Failure trajectories stop at termination; no control or motion is invented beyond it. Their averages are censored and coverage is reported. An initial failure has undefined mean control/trajectory/objective. The primary four-map summary uses `mean(skipna=False)`: these undefined values propagate, and a count of defined scenarios is shown. End errors/path lengths/statuses still include all four maps. Short partial-horizon scores must not be interpreted as better performance.

### Common MACBO objective

Each candidate input is plugged into the same **goal-referenced final-stage** manuscript objective: `w_j = p_goal`, `V=.5||x-x_goal||²`, wrapped heading coordinate, source field/visibility/circulation, and shared MACBO weights. A separate virtual memory starts at zero for each method and is replayed from that method's states; this evaluator performs no control optimization and changes no actual trajectory. This convention removes dependence on a method's private route and is not the hybrid controller's actual stage cost.

The score is

`mu_v/2 ||v_p-v_d||² + mu_t*(1-q)*(V+epsilon_V)^(-q)*dotV + mu_u/2 ||u||² + mu_delta_u/2 ||u-u_prev||² + mu_delta/2 delta_req²`,

where `delta_req=max(0,dotV+c*V^q)`. This is the smallest nonnegative CLF slack needed for the candidate derivative. If it exceeds `c_bar*V^q`, the score is still calculable but the strict QP constraint is infeasible; the violation fraction is saved. No violation is hidden by clipping the required slack. Numerical terminal holds can also violate a strict finite-rate inequality away from exact equality. The transformed derivative can be negative, so scores are not necessarily nonnegative.

Stage costs use left-endpoint quadrature on recorded intervals. Long post-success holds are subdivided to at most .6 s for virtual memory integration. `common_objective.csv` records each value, weight, memory, required slack, and bound. Time-weighted means are then averaged with equal weights across the four scenarios. This convention must accompany the objective column in any publication.

### Runtime and computational work

Online wall time comes from the existing runner and includes unsuccessful controller calls. Setup is separate; setup + summed online times measures controller computation, excluding rendering/scoring/simulation guards. Mean and p95 online milliseconds are saved. CPU/hardware load changes timings across runs.

Optimizer iterations per call count SLSQP iterations; DHOCBF sums inner-QP iterations. RRT/HJ have no online optimizer iteration count and display undefined values. A separate table records RRT nodes/rewires, MACBO A* expansions, HJ grid nodes and scalar Bellman candidates, MPC variable count, and HJ tensor storage. These operations are different work units, not a uniform complexity score or theoretical Big-O estimate.

## Artifacts

`results/benchmarking/` contains:

- `trials/<scenario>/<method>/`: scene, original experiment report, rollout arrays; common objective trace when defined; MACBO `route.npz` with original graph/shortened points.
- `manifest.json`: full configurations including defaults, source SHA-256 fingerprints, numerical-library versions, endpoints/tolerances, and scoring convention.
- `per_trial_metrics.csv`, `mean_metrics.csv`, `work_proxies.csv`: machine-readable pandas results.
- `summary_table.html/.csv/.tex`, `coverage_and_timing.csv`: publication-oriented summary. HTML tables use Times New Roman; LaTeX table font follows the paper's document font.
- `figures/<scenario>/trajectory`, `control_norm`, `orientation`, each PNG at 600 dpi and vector PDF with embedded fonts; a 600 dpi four-map overview.
- `gifs/u_and_t_<method>.gif`: all seven motion outcomes. GIFs are 90 dpi playback media; the 600 dpi requirement applies to static figures. Adaptive playback is labelled. Zero-step failures produce a labelled static animation rather than fictitious motion.

All figures have titles, legends, axis names, and grids. Small filled MACBO dots are the original A* graph points, and open markers are shortened waypoints. Failed endpoints are marked by crosses. Wrapped-angle lines break at branch cuts.

## Verification

The notebook runs the 7 environment + 18 literature + 24 MACBO + 9 benchmark/dynamics/animation + 2 smoothing tests (60 total). New tests check fixed maps, independent exact dynamics, every common objective term, success-only terminal extension, and undefined-value propagation. The notebook also validates every accepted trial's dense clearance, physical wheel bounds, interval acceptance, MACBO rate/barrier residuals, reverse-then-forward U route, exact U + T replay, PNG DPI, and GIF frame count.

Passing these finite tests validates these cases; it does not establish arbitrary-map convergence or universal absence of bugs. Full statuses are reported before any summary averages.

## Approved smoothing update and actual dynamics

The main benchmark now adopts the successful `gentler_motion` trial: `k_w=.4`,
`k_g=.02`, `mu_delta_u=1`, `max_turn_speed=1`, `heading_gain=3`. The hybrid
controller, dense route grid, hard safety constraints, and physical model are
unchanged. All 28 controllers are rerun; the four promoted MACBO rollouts must
match the saved gentler trial exactly. Common-objective weights also change
for every method, so old and new objective scores are different scoring designs.
The previous results, notebook, manifest, and original package source snapshot
are retained in `results/benchmarking_before_smoothing`. The smoothing notebook
reads this explicit historical archive and generates candidates from its saved
baseline configuration. Normal `load_benchmark` still rejects stale sources and
settings; `historical=True` is solely for reading archived designs.

`dynamics.py` computes actual `xdot=f(x)+g(x)u` from applied controls. Autonomous
`f(x)=0` is the differential-drive robot's physical model, not a zero-dynamics
simulation switch. A nonzero drift would require a different specified model.
The new two-panel figure shows mean actual state-derivative norm and mean input
matrix Frobenius norm. CSVs include autonomous drift, planar speed, yaw rate,
status, and scored time. Full-state norms mix the original position/angle units;
separate speed/yaw columns retain physical units. Held-input derivative norms
are exactly constant over each interval for this robot, so time weighting is
exact. Initial failures have undefined actual-motion means and known model
norms. Four-map means propagate undefined values.

GIFs now reconstruct exact poses with physical sample gaps at most .4 s and
accumulated absolute yaw at most .25 rad between frames. Adaptive playback
spends extra frames on turns; the physical clock is authoritative, and playback
speed is not uniform. Endpoint outcomes pause for 1.5 playback seconds. Failure
messages and zero-command outcomes are visible. JSON sidecars preserve sample
times and poses for validation; no trajectory interpolation or fictitious
continuation is used. Static publication figures remain 600 dpi PNG/vector PDF.
