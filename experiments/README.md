# Additive MACBO smoothing trial

Open `../notebooks/macbo_smoothing.ipynb` and Run All using the same dependencies as the benchmark (`benchmarks/requirements.txt`). Every new output is written under `results/smoothing/`; approved source files, notebooks, and benchmark artifacts remain unchanged. No experimental setting is promoted automatically.

The notebook tests two existing-hyperparameter configurations on all four maps: an input-change penalty of 1, and a gentler combined configuration with `k_w=.4`, `k_g=.02`, `mu_delta_u=1`, `max_turn_speed=1`, and `heading_gain=3`. All other settings, safety inequalities, physical limits, sampling intervals, goal tolerances, and maps are preserved. The eight candidates save the full standard experiment logs and routes. All applied inputs still come directly from the constrained MACBO QP; no post-QP filter or plot smoothing is used.

`smoothing.py` contains reusable experiment, rate-metric, model-norm, and no-change validation helpers. It is not imported into the approved controller or benchmark.

Sampled input-rate RMS and sampled planar-acceleration RMS include the start/stop transitions and average over active motion, without dilution by a 120 s terminal hold. These are finite-difference proxies on held-input logs, not continuous jerk bounds. Lower attraction/turn gains and stronger input-change regularization trade speed for gentler transitions. Stationary-turn phases and geometric waypoint corners remain; do not claim a continuously rounded trajectory from these parameters alone.

The added `robot_dynamics_norms` figure has two panels: mean vector L2 norm of f and mean matrix Frobenius norm of g, for all seven methods in all four scenarios. Here f is identically zero and `||g||F=sqrt(R²/2+2R²/D²)` is state-independent. Values therefore agree across methods/maps and do not measure controller performance. The norm uses the existing unscaled model coordinates; it mixes the position/angular components of g. A zero-duration failed trial is labelled as an analytical model constant rather than assigned an invented trajectory. Actual velocity `f+g*u` is a different quantity.

Figures use Times New Roman, Seaborn styling, 600 dpi PNG, and vector PDF. Tables and per-scenario/mean/paired-change metrics are pandas CSV exports. The notebook also shows unfiltered trajectory overlays and U + T translational/yaw-rate comparisons.

Checks include goal arrival, independent exact within-interval clearance, original guard acceptance, wheel limits, bounded CLF slack, CLF/barrier residuals, exact U + T replay, figure DPI, and byte-for-byte preservation of existing files. `results/smoothing/protected_files.json` records the pre-experiment hashes. `tests/smoothing/test_smoothing.py` checks the dynamics norm identity and sampled rate definition.

Finder metadata files (`.DS_Store`) are excluded from preservation checks; source, notebook, and actual benchmark artifacts remain protected.

## Subsequent benchmark promotion

The user subsequently approved the gentler configuration. The main benchmark
now uses it. This trial notebook retains its original comparison by loading
`results/benchmarking_before_smoothing` with explicit historical mode, including
the original benchmark manifest and source snapshot. Candidate parameters come
from the archived full baseline config, rather than current benchmark settings.
The original full preservation manifest remains as historical evidence; current
checks cover unchanged environment/controller sources and the original three
method notebooks, allowing the explicitly authorized benchmark update.
The current benchmark includes a separate actual-velocity `f+g*u` figure;
this historical trial's f/g figure still describes only the model fields.
