# Six literature methods for the existing robot environment

This package is additive: the original README, `robot_env/`, original notebook,
requirements, manuscript, and baseline tests are unchanged.

## Install and run

From the project root, in your notebook's Python environment:

```sh
python -m pip install -r methods/requirements.txt
jupyter lab notebooks/literature_methods.ipynb
```

```python
from robot_env import Environment
from methods import make_method, run_method

env = Environment(bounds=(-0.65, 0.65, -0.65, 0.65))
env.add('circle', radius=0.08)
env.reset((-0.45, -0.18, 0))

method = make_method('nmpc', goal=[0.45, 0.18, 0],
                     dt=0.2, horizon_steps=8, max_iterations=60, seed=7)
experiment = run_method(env, method, horizon=16,
                        position_tolerance=0.08, heading_tolerance=0.3)
print(experiment.metrics())
experiment.save('results/my_nmpc_experiment')
```

All methods accept a full `[x,y,theta]` goal and an immutable dataclass config.
`make_method(name, goal, **hyperparameters)` constructs that config automatically.
Unknown hyperparameters fail immediately. `run_method` resets method memory,
copies the environment, performs precomputation, executes exact held wheel-speed
inputs, and records explicit outcomes. The original `robot_env.simulate` also
accepts these objects with `mode='wheels', dt=method.config.dt`; expected method
failures raise `MethodFailure` there instead of producing an `Experiment`.

Use one object per experiment; the runner resets it each time. Changing geometry,
robot parameters, or planning bounds after preparation requires resetting the
method. All methods operate on the **known static map**. Moving-obstacle sensing,
uncertain dynamics, and disturbance games are not represented by the core model.

## Method names and configurations

| Factory name | Class / config | Important hyperparameters |
|---|---|---|
| `kinodynamic_rrt_star` | `KinodynamicRRTStar` / `RRTStarConfig` | `iterations`, `goal_bias`, `extension_length`, `neighbor_radius`, `nominal_speed`, `angular_speed`, `allow_reverse`, `safety_margin`, `seed` |
| `clf_cbf_qp` | `CLFCBFQP` / `CLFCBFConfig` | `alpha`, `gamma`, `rho`, `W1`, `W2`, `P`, solver tolerances |
| `nmpc` | `NonlinearMPC` / `NMPCConfig` | `horizon_steps`, `Q`, `R`, `terminal_weights`, `max_iterations`, `collision_substeps`, `route_guidance` |
| `mpc_dhocbf` | `DiscreteHOCBFMPC` / `DHOCBFConfig` | MPC options plus `gammas`, `convex_iterations`, `trust_radius`, `convex_tolerance` |
| `hj_reach_avoid` | `HJReachAvoid` / `HJConfig` | `grid_shape`, `horizon_steps`, `control_levels`, target tolerances, `memory_limit_mb` |
| `composite_mpc` | `CompositeBarrierMPC` / `CompositeConfig` | MPC options plus `eta`, `gamma`, `slack_penalty`, `max_slack` |

Every config exposes `dt` (seconds) and `seed`. Read all defaults with
`dataclasses.asdict(Config())`. Q, R, W1, W2 and terminal weights are positive
**diagonal entries**, supplied as tuples. P is a row-major nine-entry symmetric
positive-definite matrix. MPC solves normalized wheel inputs in [-1,1], so R
weights are applied to wheel speed divided by the physical wheel limit. This
scaling is part of the benchmark definition, not a published parameter choice.

MPC `route_guidance` defaults to False for NMPC/composite and True for DHOCBF.
It supplies an A* waypoint reference, not a backup control law. Additional
route assistance on a method must be identified in comparisons. `enforce_bounds`
is False by default; the original environment has no physical workspace walls.
RRT and HJ need finite computational bounds. HJ treats exit from its grid as
inadmissible, which is an extra computational constraint.

## What is implemented, and what is adapted

These are executable, equation-based **adapted research baselines**, not claims
of exact reproduction of the cited experiments or their guarantees. The supplied
literature review deliberately summarizes paradigms; its displayed equations
do not fully specify all six original algorithms. Source-specific differences
are recorded here and in `sources.json` so benchmark attribution stays accurate.

### Kinodynamic RRT*

Seeded pose sampling, bounded extension, cheapest collision-free parent
selection, directed rewiring, and propagation of changed descendant costs.
Edges are **physical wheel-input trajectories**, not straight holonomic edges
followed by another controller. A local rotate/translate/rotate connection
reaches each endpoint pose; optional backward translation chooses the cheaper
of two primitive sequences. Phases have integer multiples of dt and their
speeds are reduced to hit the endpoint exactly. Cost is executed duration.

The adaptation uses differential-drive primitives instead of the source's
Dubins/double-integrator steering. It does not solve an optimal steering boundary
value problem, and does not claim asymptotic optimality for this quantized
primitive class. Randomness is local to a NumPy Generator. Tree states, edges,
costs, rewires, plan controls, and predicted physical poses are inspectable.

### Huang CLF–CBF QP

Implements the cross-term quadratic pose function (11), offset-point circular
barriers (16)/(18), and the CLF-relaxed, CBF-hard quadratic program (24), with
wheel bounds mapped into the source's twist cost. Static obstacles have zero
time derivative. Negative CLF slack is permitted as in the source program.

Each polygon is enclosed by a circle centered at its configured shape center.
This intentionally preserves the source's circular-obstacle assumption; U/star
cavities cannot be used by this method. The radius is further inflated by the
lookahead offset, so protecting the offset point protects the physical center
disk. This conservative inflation is an adaptation to the existing pose model.
The source's rear-axle interpretation does not directly equal this environment's
center coordinates. Heading is wrapped, with a branch cut at +/-pi. A relaxed
pose function is not a proof of global convergence for a nonholonomic robot;
local deadlock or horizon exhaustion is a valid reported outcome.

### Ismael NMPC

Exact physical forward rollout, quadratic stage/terminal costs (9–11), wheel
limits, distance-to-occupied-set constraints inspired by (14), shifted warm
starts, and application of only the first optimized command. SLSQP performs
nonlinear direct shooting. Collision samples have a speed-based inflation to
cover gaps between samples, and the selected command is checked against the
unchanged environment's conservative interval guard.

The source uses an omniwheel platform, LiDAR/DBSCAN, and three body inputs.
Here those are replaced by the selected differential-drive model, two wheel
inputs, and the known static occupied sets. We do not implement sensing or
claim a certified terminal invariant region. Local optima, infeasibility,
and insufficient horizons remain possible.

### Liu iterative DHOCBF MPC

Grid route, repeated convex QPs, trust regions, linearized condensed physical
predictions, and all levels of the supplied recursion
`psi_next = psi[1:] - (1-gamma)*psi[:-1]`. Every proposed solution is re-evaluated
against the nonlinear dynamics/geometry. Failure is not hidden by accepting a
linearized constraint residual as proof of true feasibility.

The source's acceleration-controlled state and occupancy-grid separating
polytopes are adapted to wheel inputs and exact convex-component distance
barriers. We use hard recursions without the source's per-polytope multiplicative
relaxation. The default order is one: it matches the source case study's order,
and the wheel-speed physical position barrier normally has relative degree one.
Higher configured recursion orders are experimental constraints, not proof of a
higher relative degree for this robot. The inner quadratic programs use SLSQP
with analytic QP gradients; this is not the source's OSQP/ROS implementation.

### Gong HJ reach–avoid admissible controls

A genuine value-function calculation on a three-dimensional (x,y,theta) grid,
with periodic heading. Separate finite-horizon reach and avoid Bellman backups
compute V_r and V_a. A joint reach-avoid variational-inequality backup supplies
an additional compatibility restriction. Online candidate controls must belong
to the separate sampled reach and avoid sets, the joint set, and the physical
interval-safe set. Empty intersections are reported. No RRT/A* tracker is used.

The solver is semi-Lagrangian: exact held wheel dynamics, trilinear value
interpolation, and a finite wheel-control mesh approximate the HJ equations.
It handles one static target and static obstacles without disturbances. The
finite target is an explicitly configured position/heading neighborhood, not
a singleton. The runner automatically uses these tolerances if none are given.
Setting tighter stopping tolerances than the HJ target can prevent completion.
Target grid resolution, finite control mesh, numerical diffusion, and time
budget affect results. Stored zero sublevel sets are **not certified** continuous
under-approximations. The source's R-CLVF/stabilization and multiple time-series
tasks are outside the supplied finite-horizon reach-avoid paradigm.

### Khaledi composite barrier MPC

Stable unnormalized soft-minimum over C1 convex-component barriers,
`-logsumexp(-eta*b)/eta`, a discrete composite decay inequality, and one shared
nonnegative scalar slack with configurable bound/penalty. `max_slack=0` is the
hard-barrier default. Positive slack relaxes barrier decay but physical occupied-
set constraints remain hard. There is one aggregate inequality per predicted
transition, rather than one decay inequality per obstacle/component.

The available publisher abstract/introduction/snippets confirm soft-min and a
single scalar slack; the full article's terminal and slack details were not
available. This implementation follows the **supplied static composite equation**
and documented numerical choices; it is not an exact full-article reproduction.
There is no claim of the original paper's recursive feasibility/stability.

Polygon obstacles are triangulated exactly without filling cavities. Squared
distance to each convex triangle is C1; circle barriers are smooth polynomials.
Their intersection gives exactly the physical clearance set. Soft-min aggregates
**components**, so decomposition count affects conservatism. It has a gap of up
to log(component_count)/eta from the minimum (barriers use squared-meter units).
Increasing eta reduces the gap but can worsen numerical conditioning. Do not
compare it with a per-obstacle soft-min without identifying this adaptation.

## Logging and reproducibility

`experiment.save(directory)` writes:

- `scene.json`: original start state, robot parameters, all obstacle vertices,
  and planning bounds, readable by `Environment.load`.
- `rollout.npz`: accepted physical states/outputs, applied wheels, commands,
  sampled clearance, and online controller timings.
- `report.json`: full config, goal, stopping tolerances/horizon, status/failure,
  diagnostics, attempted inputs (including rejected candidates), map fingerprint,
  and Python/NumPy/SciPy versions. Non-finite empty-map margins are JSON null.

RRT/HJ precomputation time is included separately in `setup_seconds`. Online
solver times include unsuccessful controller calls; thus their count can be
one greater than accepted intervals. Trial time does not include precomputation.
A last fractional interval shorter than dt is omitted. Measured runtimes depend
on hardware and are not repeatability targets; seeded trajectories/configs are.

Statuses distinguish `goal`, `horizon`, `solver_failure`, `planning_failure`,
`empty_admissible_set`, `time_budget_exhausted`, `unsafe_command`,
`collision_guard`, `target_unresolved`, and resource/config/map errors. A failed
method never becomes the baseline follower. The physical guard still rejects
uncertifiable curved intervals; that is distinct from collision in a saved
accepted trajectory. Goal success includes both position and wrapped heading.

## Validation

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s tests/methods -v
```

The new tests check original/vectorized geometry agreement, concave decomposition
free sets, soft-min stability, recursion identities, physical prediction parity,
QP feasibility, RRT endpoint/cost/seed invariants, HJ periodicity/value backups,
empty ACS/memory/target failures, warm-start reset, configs, JSON, and full-pose
empty-map arrival for all six methods. The executed notebook adds an obstructed
circle experiment, all-five-shape short compatibility checks, source notes,
plots, and reproducibility checks. Obstructed-scene failures remain visible;
these are functional demonstrations, not a statistical benchmark or proof.
