"""NMPC, iterative discrete high-order barrier MPC, and composite barrier MPC."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from robot_env.core import positive
from robot_env.planning import astar
from .common import (BaseConfig, Method, MethodFailure, pose_error, wrap, scale_wheels,
                     integer, validate_weights, jacobian, interval_safe)
from .geometry import signed_distances, Barriers, softmin, dhocbf_levels


@dataclass(frozen=True)
class NMPCConfig(BaseConfig):
    horizon_steps: int = 10
    Q: tuple = (15.,15.,0.4)
    R: tuple = (0.01,0.01)
    terminal_weights: tuple = (30.,30.,0.8)
    smoothness_weight: float = 0.01
    max_iterations: int = 80
    solver_tolerance: float = 1e-6
    feasibility_tolerance: float = 1e-6
    safety_margin: float = 0.005
    collision_substeps: int = 2
    route_guidance: bool = False
    grid_spacing: float = 0.15
    route_margin: float = 0.04
    waypoint_tolerance: float = 0.05
    reference_speed: float = 0.12
    enforce_bounds: bool = False
    def __post_init__(self):
        super().__post_init__()
        integer(self.horizon_steps,'horizon_steps',2);integer(self.max_iterations,'max_iterations');integer(self.collision_substeps,'collision_substeps')
        validate_weights(self.Q,3,'Q');validate_weights(self.R,2,'R');validate_weights(self.terminal_weights,3,'terminal_weights')
        for v in [self.solver_tolerance,self.feasibility_tolerance,self.grid_spacing,self.waypoint_tolerance,self.reference_speed]:positive(v)
        for v in [self.smoothness_weight,self.safety_margin,self.route_margin]:
            if not np.isfinite(v) or v<0:raise ValueError('Margins and smoothing weight must be nonnegative finite')
        if not isinstance(self.route_guidance,bool) or not isinstance(self.enforce_bounds,bool):raise ValueError('Flags must be bool')


@dataclass(frozen=True)
class DHOCBFConfig(NMPCConfig):
    route_guidance: bool = True
    gammas: tuple = (0.3,)
    convex_iterations: int = 5
    trust_radius: float = 0.35
    convex_tolerance: float = 1e-3
    def __post_init__(self):
        super().__post_init__()
        if not self.gammas or len(self.gammas)>self.horizon_steps or any(not np.isfinite(g) or not 0<g<=1 for g in self.gammas):raise ValueError('gammas must be in (0,1], with order <= horizon')
        integer(self.convex_iterations,'convex_iterations');positive(self.trust_radius);positive(self.convex_tolerance)


@dataclass(frozen=True)
class CompositeConfig(NMPCConfig):
    eta: float = 80.0
    gamma: float = 0.3
    slack_penalty: float = 10000.0
    max_slack: float = 0.0
    def __post_init__(self):
        super().__post_init__();positive(self.eta);positive(self.slack_penalty)
        if not np.isfinite(self.gamma) or not 0<self.gamma<=1:raise ValueError('gamma must be in (0,1]')
        if not np.isfinite(self.max_slack) or self.max_slack<0:raise ValueError('max_slack must be finite and nonnegative')


class NonlinearMPC(Method):
    """Direct-shooting wheel-speed NMPC using exact held-input physical dynamics.

    Ismael equations 9–14 adapted from 3-input omniwheel/LiDAR to 2-input
    differential drive/known occupied sets. No certified terminal region.
    Route assistance is optional and disabled by default.
    """
    name='nmpc'
    def __init__(self,goal,config=None):super().__init__(goal,config or NMPCConfig())

    def reset(self):
        super().reset();self.warm=None;self.previous=np.zeros(2);self.route=None;self.route_index=1

    def prepare(self,env):
        super().prepare(env);self.barriers=Barriers(env)
        if self.config.route_guidance:
            try:self.route=astar(env,env.state[:2],self.goal[:2],self.config.grid_spacing,self.config.route_margin)
            except (RuntimeError,ValueError) as exc:raise MethodFailure('planning_failure',str(exc)) from exc

    def reference(self,q,env):
        cfg=self.config;N=cfg.horizon_steps
        if self.route is None:return np.tile(self.goal,(N,1))
        while self.route_index<len(self.route)-1 and np.linalg.norm(q[:2]-self.route[self.route_index])<=cfg.waypoint_tolerance:self.route_index+=1
        target=self.route[self.route_index]
        heading=self.goal[2] if self.route_index==len(self.route)-1 and np.linalg.norm(q[:2]-target)<2*cfg.waypoint_tolerance else np.arctan2(target[1]-q[1],target[0]-q[0])
        # Position target stays at active waypoint; heading follows its segment.
        return np.tile(np.r_[target,heading],(N,1))

    def predict(self,q,normalized,env):
        controls=np.asarray(normalized).reshape(-1,2)*env.robot.max_wheel_speed
        states=[q]
        for u in controls:states.append(env.robot.advance(states[-1],u,self.config.dt))
        return np.array(states)

    def initial_guess(self,q,refs,env):
        cfg=self.config;N=cfg.horizon_steps
        if self.warm is not None:return np.vstack((self.warm[1:],self.warm[-1:])).ravel()
        states=q.copy();controls=[]
        for ref in refs:
            delta=ref[:2]-states[:2];dist=np.linalg.norm(delta)
            heading=ref[2] if dist<cfg.waypoint_tolerance else np.arctan2(delta[1],delta[0])
            error=wrap(heading-states[2]);v=min(cfg.reference_speed,dist)*max(0.,np.cos(error))
            if abs(error)>.4:v=0.
            u=scale_wheels(env.robot,[v,np.clip(3*error,-1.5,1.5)])
            controls.append(u/env.robot.max_wheel_speed);states=env.robot.advance(states,u,cfg.dt)
        return np.asarray(controls).ravel()

    def residuals(self,q,y,refs,env):
        cfg=self.config;states=self.predict(q,y[:2*cfg.horizon_steps],env)[1:]
        errors=pose_error(states,refs)
        u=y[:2*cfg.horizon_steps].reshape(-1,2)
        prev=self.previous/env.robot.max_wheel_speed
        return np.r_[(errors*np.sqrt(np.asarray(cfg.Q)*cfg.dt)).ravel(),
                     (errors[-1]*np.sqrt(np.asarray(cfg.terminal_weights)/2)),
                     (u*np.sqrt(np.asarray(cfg.R)*cfg.dt)).ravel(),
                     (np.diff(np.vstack((prev,u)),axis=0)*np.sqrt(cfg.smoothness_weight)).ravel()]

    def obstacle_constraints(self,q,y,env):
        cfg=self.config;N=cfg.horizon_steps
        states=self.predict(q,y[:2*N],env)
        values=[]
        if env.obstacles:
            # Lipschitz SDF sampling with conservative speed-gap inflation;
            # checks the whole curved interval, not just shooting nodes.
            controls=y[:2*N].reshape(-1,2)*env.robot.max_wheel_speed
            for k,u in enumerate(controls):
                v,_=env.robot.wheels_to_twist(u)
                fraction=np.arange(cfg.collision_substeps+1)/cfg.collision_substeps
                points=np.array([states[k] if f==0 else env.robot.advance(states[k],u,cfg.dt*f) for f in fraction])[:,:2]
                inflation=abs(v)*cfg.dt/(2*cfg.collision_substeps)
                values.extend((signed_distances(points,env.obstacles)-env.safety_radius-cfg.safety_margin-inflation).ravel())
        if cfg.enforce_bounds:
            xmin,xmax,ymin,ymax=env.bounds
            values.extend(np.column_stack((states[:,0]-xmin,xmax-states[:,0],states[:,1]-ymin,ymax-states[:,1])).ravel())
        return np.asarray(values)

    def constraints(self,q,y,env):return self.obstacle_constraints(q,y,env)

    def objective(self,q,y,refs,env):
        r=self.residuals(q,y,refs,env);return float(r@r)

    def solve(self,q,guess,refs,env):
        cfg=self.config
        fn=lambda y:self.constraints(q,y,env)
        constraints=[] if fn(guess).size==0 else [{'type':'ineq','fun':fn}]
        result=minimize(lambda y:self.objective(q,y,refs,env),guess,method='SLSQP',bounds=[(-1.,1.)]*len(guess),
                        constraints=constraints,options={'maxiter':cfg.max_iterations,'ftol':cfg.solver_tolerance})
        return result.x,{'solver_success':bool(result.success),'solver_message':str(result.message),'iterations':int(result.nit),'objective':float(result.fun)}

    def control(self,observation,env):
        q=observation['state'];cfg=self.config;refs=self.reference(q,env)
        guess=self.initial_guess(q,refs,env)
        y,info=self.solve(q,guess,refs,env)
        residual=self.constraints(q,y,env)
        margin=float(residual.min()) if residual.size else None
        info.update(constraint_margin=margin,route_guidance=cfg.route_guidance,barrier_components=len(self.barriers.components))
        self.record(**info)
        if not info['solver_success'] or not np.all(np.isfinite(y)) or (margin is not None and margin < -cfg.feasibility_tolerance):
            raise MethodFailure('solver_failure',self.name+': '+info['solver_message'])
        normalized=y[:2*cfg.horizon_steps].reshape(-1,2)
        wheels=normalized[0]*env.robot.max_wheel_speed
        if not interval_safe(env,q,wheels,cfg.dt):raise MethodFailure('unsafe_command','Predicted command fails environment interval guard')
        self.warm=normalized.copy();self.previous=wheels.copy()
        return wheels


class DiscreteHOCBFMPC(NonlinearMPC):
    """Route-guided sequential convex QPs imposing all DHOCBF recursion levels.

    Wheel-input relative degree is normally one (default order=1); higher
    orders are configurable recursion experiments, not a reproduction of Liu's
    acceleration-controlled model. Convex component distances replace occupancy
    polytope extraction. Each QP linearizes condensed dynamics/constraints,
    uses a control trust region, and is rechecked with nonlinear predictions.
    """
    name='mpc_dhocbf'
    def __init__(self,goal,config=None):super().__init__(goal,config or DHOCBFConfig())

    def constraints(self,q,y,env):
        ordinary=self.obstacle_constraints(q,y,env)
        b=self.barriers.values(self.predict(q,y,env)[:,:2])
        levels=dhocbf_levels(b,self.config.gammas)
        return np.r_[ordinary,*[level.ravel() for level in levels]]

    def solve(self,q,guess,refs,env):
        cfg=self.config;nominal=guess.copy();records=[];success=False
        for iteration in range(cfg.convex_iterations):
            r=self.residuals(q,nominal,refs,env)
            J=jacobian(lambda y:self.residuals(q,y,refs,env),nominal)
            c=self.constraints(q,nominal,env)
            A=jacobian(lambda y:self.constraints(q,y,env),nominal) if c.size else np.empty((0,len(nominal)))
            bound=cfg.trust_radius
            lower=np.maximum(-bound,-1-nominal);upper=np.minimum(bound,1-nominal)
            Q=2*J.T@J+np.eye(len(nominal))*1e-8;linear=2*J.T@r
            constraints=[LinearConstraint(A,-c,np.inf)] if c.size else []
            result=minimize(lambda d:.5*d@Q@d+linear@d,np.zeros_like(nominal),jac=lambda d:Q@d+linear,
                            method='SLSQP',bounds=Bounds(lower,upper),constraints=constraints,
                            options={'maxiter':cfg.max_iterations,'ftol':cfg.solver_tolerance})
            records.append({'qp_success':bool(result.success),'qp_iterations':int(result.nit)})
            if not result.success:break
            trial=np.clip(nominal+result.x,-1,1)
            true_c=self.constraints(q,trial,env)
            feasible=not true_c.size or true_c.min()>=-cfg.feasibility_tolerance
            nominal=trial
            if feasible:
                success=True
                if np.linalg.norm(result.x,np.inf)<cfg.convex_tolerance:break
            else:success=False
        # Feasibility is not assumed from a linearized QP alone.
        c=self.constraints(q,nominal,env)
        success=success and (not c.size or c.min()>=-cfg.feasibility_tolerance)
        return nominal,{'solver_success':bool(success),'solver_message':'Sequential convex QPs converged to feasible nonlinear rollout' if success else 'Sequential convex solve failed nonlinear feasibility',
                        'iterations':len(records),'qp_iterations':records,'objective':self.objective(q,nominal,refs,env)}


class CompositeBarrierMPC(NonlinearMPC):
    """Static soft-min composite MPC with one global nonnegative scalar slack.

    Implements the supplied soft-min paradigm; available publisher snippets
    do not suffice to reproduce Khaledi's complete terminal/slack construction.
    Smooth barriers aggregate convex components, not nonsmooth polygon SDFs.
    Physical occupied-set safety remains hard even with positive barrier slack.
    """
    name='composite_mpc'
    def __init__(self,goal,config=None):super().__init__(goal,config or CompositeConfig())

    def constraints(self,q,y,env):
        N=self.config.horizon_steps
        hard=self.obstacle_constraints(q,y,env)
        b=self.barriers.values(self.predict(q,y[:2*N],env)[:,:2])
        if b.shape[1]==0:return hard
        aggregate=softmin(b,self.config.eta)
        slack=y[-1] if y.size>2*N else 0.
        return np.r_[hard,aggregate[0],aggregate[1:]-(1-self.config.gamma)*aggregate[:-1]+slack]

    def objective(self,q,y,refs,env):
        cost=super().objective(q,y,refs,env)
        return cost+self.config.slack_penalty*y[-1]**2

    def solve(self,q,guess,refs,env):
        cfg=self.config;guess=np.r_[guess,0.]
        fn=lambda y:self.constraints(q,y,env)
        constraints=[] if fn(guess).size==0 else [{'type':'ineq','fun':fn}]
        result=minimize(lambda y:self.objective(q,y,refs,env),guess,method='SLSQP',
                        bounds=[(-1.,1.)]*(len(guess)-1)+[(0.,cfg.max_slack)],constraints=constraints,
                        options={'maxiter':cfg.max_iterations,'ftol':cfg.solver_tolerance})
        return result.x,{'solver_success':bool(result.success),'solver_message':str(result.message),'iterations':int(result.nit),
                         'objective':float(result.fun),'composite_slack':float(result.x[-1])}
