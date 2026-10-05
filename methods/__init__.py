"""Added literature-method implementations; existing robot_env remains unchanged."""
from .clf_cbf import CLFCBFQP, CLFCBFConfig
from .mpc import NonlinearMPC, NMPCConfig, DiscreteHOCBFMPC, DHOCBFConfig, CompositeBarrierMPC, CompositeConfig
from .rrt_star import KinodynamicRRTStar, RRTStarConfig
from .hj_reach_avoid import HJReachAvoid, HJConfig
from .common import MethodFailure
from .experiment import Experiment, run_method
from .macbo import MACBO, MACBOConfig

METHODS = {
    'kinodynamic_rrt_star': (KinodynamicRRTStar, RRTStarConfig),
    'clf_cbf_qp': (CLFCBFQP, CLFCBFConfig),
    'nmpc': (NonlinearMPC, NMPCConfig),
    'mpc_dhocbf': (DiscreteHOCBFMPC, DHOCBFConfig),
    'hj_reach_avoid': (HJReachAvoid, HJConfig),
    'composite_mpc': (CompositeBarrierMPC, CompositeConfig),
}


def make_method(name, goal, **hyperparameters):
    """Construct a method; misspelled/unknown hyperparameters raise TypeError."""
    if name == 'macbo':
        return MACBO(goal, MACBOConfig(**hyperparameters))
    if name not in METHODS:raise ValueError(f'Unknown method {name!r}; choose {list(METHODS)}')
    controller,config=METHODS[name]
    return controller(goal,config(**hyperparameters))

__all__=['CLFCBFQP','CLFCBFConfig','NonlinearMPC','NMPCConfig','DiscreteHOCBFMPC','DHOCBFConfig',
         'CompositeBarrierMPC','CompositeConfig','KinodynamicRRTStar','RRTStarConfig','HJReachAvoid','HJConfig',
         'MethodFailure','Experiment','run_method','make_method','METHODS','MACBO','MACBOConfig']
