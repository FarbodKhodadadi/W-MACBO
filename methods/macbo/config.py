"""Validated, immutable MACBO hyperparameters (meters, radians, seconds)."""
from dataclasses import dataclass
import numpy as np
from robot_env.core import positive
from ..common import BaseConfig, integer


@dataclass(frozen=True)
class MACBOConfig(BaseConfig):
    """`hybrid` uses explicit turn/drive CLFs; `strict` uses the supplied V_j.

    Hybrid operation is a documented differential-drive adaptation. It never
    relaxes barrier inequalities or substitutes another avoidance controller.
    """
    dt: float = 0.05
    clf_mode: str = "hybrid"
    grid_spacing: float = 0.2
    refinement_factor: float = 0.5
    max_refinements: int = 3
    grid_max_nodes: int = 20000
    max_expansions: int = 100000
    max_edge_steps: int = 2
    nominal_speed: float = 0.15
    turn_delay: float = 0.3
    route_margin: float = 0.04
    shorten_route: bool = True
    circle_sides: int = 32
    epsilon_a: float = 0.15
    epsilon_w: float = 0.035
    k_w: float = 0.8
    k_g: float = 0.2
    k_c: float = 0.8
    beta: float = 40.0
    ell: float = 0.25
    lambda_memory: float = 2.0
    initial_memory: float = 0.0
    initial_sigma: int = 1
    alpha: float = 3.0
    c: float = 0.03
    c_bar: float = 0.01
    q: float = 0.75
    epsilon_V: float = 1e-6
    mu_v: float = 80.0
    mu_t: float = 0.005
    mu_u: float = 0.00002
    mu_delta_u: float = 0.00001
    mu_delta: float = 100.0
    mu_heading: float = 0.3
    heading_gain: float = 4.0
    max_turn_speed: float = 2.0
    heading_enter: float = 0.06
    heading_exit: float = 0.5
    max_field_heading_offset: float = 0.3
    target_position_tolerance: float = 0.025
    target_heading_tolerance: float = 0.025
    max_iterations: int = 100
    solver_tolerance: float = 1e-10
    feasibility_tolerance: float = 1e-7

    def __post_init__(self):
        super().__post_init__()
        if self.clf_mode not in ("strict", "hybrid"):
            raise ValueError("clf_mode must be 'strict' or 'hybrid'")
        for name in (
            "grid_spacing", "nominal_speed", "epsilon_a", "epsilon_w",
            "k_w", "k_g", "k_c", "beta", "ell", "lambda_memory", "alpha",
            "c", "epsilon_V", "mu_v", "mu_t", "mu_delta", "mu_heading",
            "heading_gain", "max_turn_speed", "heading_enter", "heading_exit",
            "max_field_heading_offset", "target_position_tolerance",
            "target_heading_tolerance", "solver_tolerance", "feasibility_tolerance",
        ):
            positive(getattr(self, name))
        for name in ("turn_delay", "route_margin", "initial_memory", "c_bar", "mu_u", "mu_delta_u"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, lower in (("max_refinements", 0), ("grid_max_nodes", 4),
                            ("max_expansions", 1), ("max_edge_steps", 1),
                            ("circle_sides", 8), ("max_iterations", 1)):
            integer(getattr(self, name), name, lower)
        if not np.isfinite(self.refinement_factor) or not 0 < self.refinement_factor < 1:
            raise ValueError("refinement_factor must be in (0,1)")
        if not np.isfinite(self.q) or not 0 < self.q < 1:
            raise ValueError("q must be in (0,1)")
        if self.c_bar >= self.c:
            raise ValueError("c_bar must be strictly less than c")
        if self.initial_sigma not in (-1, 1):
            raise ValueError("initial_sigma must be -1 or +1")
        if not self.heading_enter < self.heading_exit < np.pi / 2:
            raise ValueError("Require heading_enter < heading_exit < pi/2")
        if self.max_field_heading_offset >= self.heading_exit:
            raise ValueError("max_field_heading_offset must be below heading_exit")
        if not isinstance(self.shorten_route, bool):
            raise ValueError("shorten_route must be bool")
