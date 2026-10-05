"""Semi-Lagrangian finite-horizon HJ reach/avoid value functions on (x,y,theta)."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates
from robot_env.core import positive
from .common import BaseConfig, Method, MethodFailure, integer, wrap, interval_safe
from .geometry import signed_distances


@dataclass(frozen=True)
class HJConfig(BaseConfig):
    dt: float = 0.25
    horizon_steps: int = 80
    grid_shape: tuple = (31,31,16)
    control_levels: int = 3
    target_position_tolerance: float = 0.12
    target_heading_tolerance: float = 0.4
    value_tolerance: float = 1e-6
    safety_margin: float = 0.01
    memory_limit_mb: float = 256.0
    def __post_init__(self):
        super().__post_init__();integer(self.horizon_steps,'horizon_steps')
        if len(self.grid_shape)!=3:raise ValueError('grid_shape must be (nx,ny,ntheta)')
        for x in self.grid_shape:integer(x,'grid dimension',3)
        integer(self.control_levels,'control_levels',3)
        if self.control_levels%2!=1:raise ValueError('control_levels must be odd to include zero')
        for v in [self.target_position_tolerance,self.target_heading_tolerance,self.value_tolerance,self.memory_limit_mb]:positive(v)
        if not np.isfinite(self.safety_margin) or self.safety_margin<0:raise ValueError('safety_margin must be nonnegative')


class HJReachAvoid(Method):
    """Numerical adaptation of Gong's reach/avoid admissible-set intersection.

    Computes separate reach V_r and avoid V_a, plus a joint reach-avoid VI used
    as an additional restriction to resolve incompatibility. Semi-Lagrangian
    Bellman backups use exact held wheel flows, trilinear interpolation, periodic
    heading, and a finite sampled control set. No R-CLVF stabilizing ACS is
    claimed: this is the literature review's finite-horizon reach-avoid task.
    Interpolation/control discretization does not certify a continuous RAS set.
    """
    name='hj_reach_avoid'
    def __init__(self,goal,config=None):super().__init__(goal,config or HJConfig())

    def reset(self):
        super().reset();self.reach_values=None;self.avoid_values=None;self.joint_values=None

    def _coordinates(self,states):
        q=np.asarray(states).reshape(-1,3)
        nx,ny,nt=self.config.grid_shape;xmin,xmax,ymin,ymax=self.bounds
        return np.vstack(((q[:,0]-xmin)/(xmax-xmin)*(nx-1),
                          (q[:,1]-ymin)/(ymax-ymin)*(ny-1),
                          (wrap(q[:,2])+np.pi)/(2*np.pi)*nt))

    def interpolate(self,field,states):
        # Appending the first heading slice gives interpolation across ±pi.
        periodic=np.concatenate((field,field[:,:,:1]),axis=2)
        coords=self._coordinates(states)
        values=map_coordinates(periodic,coords,order=1,mode='nearest',prefilter=False)
        nx,ny,_=self.config.grid_shape
        outside=(coords[0]<0)|(coords[0]>nx-1)|(coords[1]<0)|(coords[1]>ny-1)
        return np.where(outside,self.big,values)

    def advance_batch(self,states,wheels,robot):
        v,w=robot.wheels_to_twist(wheels);dt=self.config.dt
        result=np.array(states,copy=True)
        travel=v*dt*np.sinc(w*dt/(2*np.pi));mid=states[:,2]+w*dt/2
        result[:,0]+=travel*np.cos(mid);result[:,1]+=travel*np.sin(mid);result[:,2]=wrap(states[:,2]+w*dt)
        return result

    def constraint_field(self,states,env):
        q=np.asarray(states).reshape(-1,3);xmin,xmax,ymin,ymax=self.bounds
        boundary=np.max(np.column_stack((xmin-q[:,0],q[:,0]-xmax,ymin-q[:,1],q[:,1]-ymax)),axis=1)
        distances=signed_distances(q[:,:2],env.obstacles)
        obstacle=np.max(env.safety_radius+self.config.safety_margin-distances,axis=1) if distances.shape[1] else np.full(len(q),-self.big)
        return np.maximum(boundary,obstacle)

    def prepare(self,env):
        super().prepare(env);cfg=self.config;self.bounds=tuple(env.bounds)
        nx,ny,nt=cfg.grid_shape;count=nx*ny*nt
        # Three float32 fields over time plus transitions/interpolation workspace.
        estimate=(3*(cfg.horizon_steps+1)*count*4+cfg.control_levels**2*count*3*8)/1024**2
        if estimate>cfg.memory_limit_mb:raise MethodFailure('resource_limit',f'Estimated HJ arrays {estimate:.1f} MB exceed memory_limit_mb')
        xmin,xmax,ymin,ymax=self.bounds
        self.big=float(max(xmax-xmin,ymax-ymin)*10+10)
        for q in [env.state,self.goal]:
            if not (xmin<=q[0]<=xmax and ymin<=q[1]<=ymax):raise MethodFailure('invalid_endpoint','HJ endpoint outside grid bounds')
        axes=(np.linspace(xmin,xmax,nx),np.linspace(ymin,ymax,ny),np.linspace(-np.pi,np.pi,nt,endpoint=False))
        self.grid_states=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        angle_scale=cfg.target_position_tolerance/cfg.target_heading_tolerance
        self.target=np.maximum(np.linalg.norm(self.grid_states[:,:2]-self.goal[:2],axis=1)-cfg.target_position_tolerance,
                               angle_scale*(np.abs(wrap(self.grid_states[:,2]-self.goal[2]))-cfg.target_heading_tolerance))
        if not np.any(self.target<=0):raise MethodFailure('target_unresolved','No grid node in target; refine grid or increase explicit target tolerances')
        self.constraint=self.constraint_field(self.grid_states,env)
        levels=np.linspace(-env.robot.max_wheel_speed,env.robot.max_wheel_speed,cfg.control_levels)
        self.controls=np.array([(r,l) for r in levels for l in levels])
        self.transitions=[self.advance_batch(self.grid_states,u,env.robot) for u in self.controls]
        self.interval_constraints=[]
        # A Lipschitz speed inflation bounds clearance between half-step samples.
        for u,next_states in zip(self.controls,self.transitions):
            v,w=env.robot.wheels_to_twist(u);dt=cfg.dt
            mid=self.grid_states.copy();travel=v*dt/2*np.sinc(w*dt/(4*np.pi));angle=self.grid_states[:,2]+w*dt/4
            mid[:,0]+=travel*np.cos(angle);mid[:,1]+=travel*np.sin(angle);mid[:,2]+=w*dt/2
            c=np.maximum.reduce([self.constraint,self.constraint_field(mid,env),self.constraint_field(next_states,env)])
            self.interval_constraints.append(c+abs(v)*dt/4)
        shape=(cfg.horizon_steps+1,nx,ny,nt)
        self.reach_values=np.empty(shape,dtype=np.float32);self.avoid_values=np.empty(shape,dtype=np.float32);self.joint_values=np.empty(shape,dtype=np.float32)
        self.reach_values[0]=self.target.reshape(cfg.grid_shape)
        self.avoid_values[0]=self.constraint.reshape(cfg.grid_shape)
        self.joint_values[0]=np.maximum(self.target,self.constraint).reshape(cfg.grid_shape)
        for n in range(1,cfg.horizon_steps+1):
            best_reach=np.full(count,self.big);best_avoid=best_reach.copy();best_joint=best_reach.copy()
            for next_states,c in zip(self.transitions,self.interval_constraints):
                vr=self.interpolate(self.reach_values[n-1],next_states)
                va=self.interpolate(self.avoid_values[n-1],next_states)
                vj=self.interpolate(self.joint_values[n-1],next_states)
                best_reach=np.minimum(best_reach,vr)
                best_avoid=np.minimum(best_avoid,np.maximum(c,va))
                best_joint=np.minimum(best_joint,np.maximum(c,vj))
            self.reach_values[n]=np.minimum(self.target,best_reach).reshape(cfg.grid_shape)
            self.avoid_values[n]=np.maximum(self.constraint,best_avoid).reshape(cfg.grid_shape)
            self.joint_values[n]=np.maximum(self.constraint,np.minimum(self.target,best_joint)).reshape(cfg.grid_shape)
        self.start_time=env.time
        self.record(stage='precomputation',grid_shape=list(cfg.grid_shape),controls=len(self.controls),estimated_memory_mb=estimate,
                    initial_joint_value=float(self.interpolate(self.joint_values[-1],env.state[None,:])[0]))

    def admissible_controls(self,q,remaining,env):
        """Finite control-set intersection, returned with next-state value arrays."""
        if not 1<=remaining<=self.config.horizon_steps:raise ValueError('remaining outside computed horizon')
        states=np.array([env.robot.advance(q,u,self.config.dt) for u in self.controls])
        vr=self.interpolate(self.reach_values[remaining-1],states)
        va=self.interpolate(self.avoid_values[remaining-1],states)
        vj=self.interpolate(self.joint_values[remaining-1],states)
        tol=self.config.value_tolerance
        reachable=vr<=tol;avoidable=va<=tol
        physical=np.array([interval_safe(env,q,u,self.config.dt) for u in self.controls])
        mask=reachable & avoidable & (vj<=tol) & physical
        return mask,states,vr,va,vj

    def control(self,observation,env):
        cfg=self.config;q=observation['state']
        remaining=cfg.horizon_steps-int(round((observation['time']-self.start_time)/cfg.dt))
        if remaining<=0:raise MethodFailure('time_budget_exhausted','HJ precomputed horizon exhausted')
        mask,states,vr,va,vj=self.admissible_controls(q,remaining,env)
        self.record(stage='execution',remaining_steps=remaining,admissible_controls=int(mask.sum()),
                    joint_value=float(self.interpolate(self.joint_values[remaining],q[None,:])[0]))
        if not mask.any():raise MethodFailure('empty_admissible_set','No sampled control preserves discretized reach/avoid sets')
        # Among viable controls, favor lower joint VI value; use goal error to
        # break ties and avoid waiting unnecessarily in the reachable tube.
        angle_scale=cfg.target_position_tolerance/cfg.target_heading_tolerance
        distance=np.linalg.norm(states[:,:2]-self.goal[:2],axis=1)
        heading_error=abs(wrap(states[:,2]-self.goal[2]))
        if np.linalg.norm(q[:2]-self.goal[:2]) <= cfg.target_position_tolerance*1.5:
            # Inside the positional neighborhood, prioritize full-pose arrival
            # among viable controls rather than indefinite value-function waiting.
            score=distance+angle_scale*heading_error+0.01*vj
        else:
            score=vj+0.02*(distance+angle_scale*heading_error)
        score+=1e-8*np.sum(self.controls**2,axis=1)
        return self.controls[int(np.argmin(np.where(mask,score,np.inf)))].copy()

    def save_values(self,path):
        if not self.prepared:raise RuntimeError('Prepare before saving values')
        np.savez_compressed(path,reach=self.reach_values,avoid=self.avoid_values,joint=self.joint_values,
                            bounds=self.bounds,goal=self.goal,dt=self.config.dt,map_signature=self.map_signature)
