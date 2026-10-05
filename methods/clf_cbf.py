"""Huang et al. static CLF–CBF QP, adapted to disk-center wheel inputs."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from .common import BaseConfig, Method, MethodFailure, pose_error, interval_safe, integer, validate_weights
from robot_env.core import positive


@dataclass(frozen=True)
class CLFCBFConfig(BaseConfig):
    alpha: float = 2.0
    gamma: float = 0.5
    rho: float = 20.0
    W1: tuple = (1.0, 0.05)
    W2: tuple = (0.1, 0.01)
    P: tuple = (1.,0.,0.1, 0.,1.,0.1, 0.1,0.1,0.2)
    max_iterations: int = 100
    solver_tolerance: float = 1e-9
    feasibility_tolerance: float = 1e-7
    def __post_init__(self):
        super().__post_init__()
        for value in [self.alpha,self.gamma,self.rho,self.solver_tolerance,self.feasibility_tolerance]:positive(value)
        validate_weights(self.W1,2,'W1');validate_weights(self.W2,2,'W2')
        P=np.asarray(self.P,dtype=float)
        if P.shape!=(9,):raise ValueError('P must contain nine matrix entries')
        P=P.reshape(3,3)
        if not np.all(np.isfinite(P)) or not np.allclose(P,P.T) or np.linalg.eigvalsh(P).min()<=0:raise ValueError('P must be symmetric positive definite')
        integer(self.max_iterations,'max_iterations')


class CLFCBFQP(Method):
    """Paper equations 11,16,18,24; circles conservatively enclose polygons.

    Huang uses an offset rear-axle point. Here physical center p is the pose
    coordinate, offset equals robot.lookahead, and each radius is increased by
    that offset to make offset-point clearance sufficient for center clearance.
    No route assistance or fallback controller. Polygon cavities are lost only
    for this circular-barrier method. Wrapped heading is piecewise smooth.
    """
    name='clf_cbf_qp'
    def __init__(self,goal,config=None):super().__init__(goal,config or CLFCBFConfig())

    def reset(self):
        super().reset();self.previous=np.zeros(2)

    def prepare(self,env):
        super().prepare(env)
        self.circles=[(o.center.copy(),o.radius if o.radius is not None else float(np.linalg.norm(o.vertices-o.center,axis=1).max())) for o in env.obstacles]

    def problem(self,q,env):
        """Return Q,c,A,lower,upper for z=[omega_R,omega_L,delta]."""
        cfg=self.config;robot=env.robot
        P=np.asarray(cfg.P).reshape(3,3);e=pose_error(q,self.goal)
        V=float(e@P@e);dV=2*e@P@robot.g(q)
        T=np.column_stack([robot.wheels_to_twist(a) for a in np.eye(2)])
        W1,W2=np.diag(cfg.W1),np.diag(cfg.W2)
        Q=np.zeros((3,3));Q[:2,:2]=T.T@(W1+2*W2)@T;Q[2,2]=2*cfg.rho
        c=np.r_[-2*T.T@W2@T@self.previous,0.]
        rows=[np.r_[dV,-1.]];low=[-np.inf];high=[-cfg.gamma*V]
        t=q[2];b=robot.lookahead;z=robot.output(q)
        J=np.array([[1,0,-b*np.sin(t)],[0,1,b*np.cos(t)]])
        for center,radius in self.circles:
            delta=z-center;R=radius+env.safety_radius+b
            barrier=float(delta@delta-R*R)
            derivative=2*delta@J@robot.g(q)
            rows.append(np.r_[derivative,0.]);low.append(-cfg.alpha*barrier);high.append(np.inf)
        return Q,c,np.array(rows),np.array(low),np.array(high)

    def control(self,observation,env):
        q=observation['state'];Q,c,A,lo,hi=self.problem(q,env);cfg=self.config
        initial=np.r_[self.previous,max(0.,float(A[0,:2]@self.previous-hi[0]))]
        bound=env.robot.max_wheel_speed
        res=minimize(lambda z:.5*z@Q@z+c@z,initial,jac=lambda z:Q@z+c,
                     method='SLSQP',bounds=Bounds([-bound,-bound,-np.inf],[bound,bound,np.inf]),
                     constraints=[LinearConstraint(A,lo,hi)],
                     options={'maxiter':cfg.max_iterations,'ftol':cfg.solver_tolerance})
        residual=np.minimum(A@res.x-lo,hi-A@res.x)
        margin=float(np.min(residual))
        self.record(solver_success=bool(res.success),solver_message=str(res.message),iterations=int(res.nit),
                    constraint_margin=margin,clf_slack=float(res.x[2]),objective=float(res.fun),
                    circle_count=len(self.circles))
        if not res.success or not np.all(np.isfinite(res.x)) or margin < -cfg.feasibility_tolerance:
            raise MethodFailure('solver_failure','CLF–CBF QP failed: '+str(res.message))
        if not interval_safe(env,q,res.x[:2],cfg.dt):raise MethodFailure('unsafe_command','QP command fails held-input physical interval check')
        self.previous=res.x[:2].copy();return self.previous.copy()
