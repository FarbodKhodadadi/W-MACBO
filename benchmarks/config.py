"""Frozen benchmark design: one parameter set per method across all four maps."""
from copy import deepcopy
import numpy as np

HORIZON = 120.0
POSITION_TOLERANCE = 0.08
HEADING_TOLERANCE = 0.2
SEED = 20261001
LABELS = {
    'kinodynamic_rrt_star': 'Kinodynamic RRT*',
    'clf_cbf_qp': 'CLF–CBF QP',
    'nmpc': 'NMPC',
    'mpc_dhocbf': 'DHOCBF MPC',
    'hj_reach_avoid': 'HJ reach–avoid',
    'composite_mpc': 'Composite MPC',
    'macbo': 'MACBO (hybrid)',
}
PARAMETERS = {
    'kinodynamic_rrt_star': dict(dt=.2, seed=SEED, iterations=600,
        extension_length=.65, neighbor_radius=1., nominal_speed=.19,
        goal_bias=.2, allow_reverse=True),
    'clf_cbf_qp': dict(dt=.05, seed=SEED, alpha=1., gamma=.3, rho=20.),
    'nmpc': dict(dt=.3, seed=SEED, horizon_steps=10, max_iterations=100,
        route_guidance=False, reference_speed=.18, Q=(12.,12.,.8),
        terminal_weights=(25.,25.,2.)),
    'mpc_dhocbf': dict(dt=.3, seed=SEED, horizon_steps=8, max_iterations=100,
        convex_iterations=7, trust_radius=.4, gammas=(.4,),
        reference_speed=.18, grid_spacing=.1, route_margin=.06,
        waypoint_tolerance=.08),
    'hj_reach_avoid': dict(dt=.6, seed=SEED, grid_shape=(61,61,24),
        horizon_steps=200, control_levels=3, target_position_tolerance=.08,
        target_heading_tolerance=.2, memory_limit_mb=512),
    'composite_mpc': dict(dt=.3, seed=SEED, horizon_steps=10,
        max_iterations=100, route_guidance=False, reference_speed=.18,
        eta=200., gamma=.4),
    'macbo': dict(dt=.05, seed=SEED, clf_mode='hybrid', grid_spacing=.1,
        route_margin=.06, grid_max_nodes=16000, max_expansions=150000,
        heading_enter=.005, mu_heading=20., max_field_heading_offset=.02,
        k_w=.4, k_g=.02, mu_delta_u=1., max_turn_speed=1., heading_gain=3.,
        target_position_tolerance=.08, target_heading_tolerance=.2),
}


def parameters_for(name):
    return deepcopy(PARAMETERS[name])
