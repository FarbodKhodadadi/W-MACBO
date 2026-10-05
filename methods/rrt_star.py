"""Seeded, dynamically feasible RRT* for the existing wheel-speed robot."""
from dataclasses import dataclass
import numpy as np
from robot_env.core import positive
from .common import BaseConfig, Method, MethodFailure, wrap, integer, interval_safe


@dataclass(frozen=True)
class RRTStarConfig(BaseConfig):
    iterations: int = 300
    goal_bias: float = 0.15
    extension_length: float = 0.4
    neighbor_radius: float = 0.7
    heading_weight: float = 0.05
    nominal_speed: float = 0.15
    angular_speed: float = 2.0
    safety_margin: float = 0.01
    tracking_tolerance: float = 1e-5
    allow_reverse: bool = True
    def __post_init__(self):
        super().__post_init__();integer(self.iterations,'iterations')
        for value in [self.extension_length,self.neighbor_radius,self.heading_weight,self.nominal_speed,self.angular_speed,self.tracking_tolerance]:positive(value)
        if not np.isfinite(self.goal_bias) or not 0<=self.goal_bias<=1:raise ValueError('goal_bias must be in [0,1]')
        if not np.isfinite(self.safety_margin) or self.safety_margin<0:raise ValueError('safety_margin must be nonnegative')
        if not isinstance(self.allow_reverse,bool):raise ValueError('allow_reverse must be bool')


@dataclass
class Node:
    state: np.ndarray
    parent: int | None
    cost: float
    controls: np.ndarray


class KinodynamicRRTStar(Method):
    """RRT* parent selection and rewiring with exact rotate/drive/rotate steering.

    Unlike geometric RRT followed by a tracker, every edge stores wheel inputs
    and reaches the requested pose. Rotations and translations obey wheel limits.
    Steering is restricted to these motion primitives; this is a differential-
    drive adaptation, not the source's Dubins steering or an optimal local BVP.
    Asymptotic optimality for this restricted, quantized steering is not claimed.
    """
    name='kinodynamic_rrt_star'
    def __init__(self,goal,config=None):super().__init__(goal,config or RRTStarConfig())

    def reset(self):
        super().reset();self.nodes=[];self.plan_controls=np.empty((0,2));self.plan_states=np.empty((0,3));self.rewires=0

    def metric(self,a,b):
        return np.linalg.norm(a[:2]-b[:2])+self.config.heading_weight*abs(wrap(a[2]-b[2]))

    def steer(self,a,b,env):
        cfg=self.config
        if not env.segment_free(a[:2],b[:2],cfg.safety_margin):return None
        distance=np.linalg.norm(b[:2]-a[:2]);bearing=np.arctan2(b[1]-a[1],b[0]-a[0])
        speed=min(cfg.nominal_speed,env.robot.wheel_radius*env.robot.max_wheel_speed)
        omega=min(cfg.angular_speed,2*env.robot.wheel_radius*env.robot.max_wheel_speed/env.robot.axle_length)
        alternatives=[]
        for reverse in ([False,True] if cfg.allow_reverse else [False]):
            heading=bearing+(np.pi if reverse else 0)
            if distance<1e-12:heading=a[2]
            phases=[(0.,wrap(heading-a[2])),((-1 if reverse else 1)*distance,0.),(0.,wrap(b[2]-heading))]
            controls=[]
            for travel,turn in phases:
                duration=abs(travel)/speed+abs(turn)/omega
                if duration<=1e-12:continue
                steps=max(1,int(np.ceil(duration/cfg.dt-1e-12)))
                v=travel/(steps*cfg.dt);w=turn/(steps*cfg.dt)
                u=env.robot.twist_to_wheels([v,w])
                controls.extend([u.copy() for _ in range(steps)])
            controls=np.asarray(controls).reshape(-1,2)
            alternatives.append((len(controls)*cfg.dt,controls))
        return min(alternatives,key=lambda x:x[0])

    def _ancestors(self,index):
        result=set()
        while index is not None:
            result.add(index);index=self.nodes[index].parent
        return result

    def _update_descendant_costs(self,parent):
        queue=[parent]
        while queue:
            p=queue.pop()
            for i,node in enumerate(self.nodes):
                if node.parent==p:
                    node.cost=self.nodes[p].cost+len(node.controls)*self.config.dt;queue.append(i)

    def prepare(self,env):
        super().prepare(env);cfg=self.config
        xmin,xmax,ymin,ymax=env.bounds
        for q in [env.state,self.goal]:
            if not (xmin<=q[0]<=xmax and ymin<=q[1]<=ymax):raise MethodFailure('invalid_endpoint','RRT endpoints outside sampling bounds')
        self.nodes=[Node(env.state.copy(),None,0.,np.empty((0,2)))];self.start_time=env.time
        for iteration in range(cfg.iterations):
            sample=self.goal.copy() if self.rng.random()<cfg.goal_bias else np.array([self.rng.uniform(xmin,xmax),self.rng.uniform(ymin,ymax),self.rng.uniform(-np.pi,np.pi)])
            distances=[self.metric(node.state,sample) for node in self.nodes];nearest=int(np.argmin(distances))
            a=self.nodes[nearest].state
            delta=sample[:2]-a[:2];distance=np.linalg.norm(delta)
            new=sample.copy()
            if distance>cfg.extension_length:new[:2]=a[:2]+delta*cfg.extension_length/distance
            if not env.is_free(new[:2]) or min(self.metric(n.state,new) for n in self.nodes)<1e-8:continue
            # Radius decreases in pose dimension three, capped at user radius.
            radius=min(cfg.neighbor_radius,cfg.neighbor_radius*2*(np.log(len(self.nodes)+1)/(len(self.nodes)+1))**(1/3))
            near=[i for i,node in enumerate(self.nodes) if self.metric(node.state,new)<=radius]
            if nearest not in near:near.append(nearest)
            best=None
            for i in near:
                edge=self.steer(self.nodes[i].state,new,env)
                if edge is None:continue
                cost=self.nodes[i].cost+edge[0]
                if best is None or cost<best[0]:best=(cost,i,edge[1])
            if best is None:continue
            cost,parent,controls=best;index=len(self.nodes)
            self.nodes.append(Node(new.copy(),parent,cost,controls))
            ancestors=self._ancestors(index)
            for i in near:
                if i in ancestors:continue
                edge=self.steer(new,self.nodes[i].state,env)
                if edge is not None and cost+edge[0]<self.nodes[i].cost-1e-10:
                    self.nodes[i].parent=index;self.nodes[i].controls=edge[1];self.nodes[i].cost=cost+edge[0]
                    self._update_descendant_costs(i);self.rewires+=1
        # Choose cheapest dynamically feasible goal connection after all rewires.
        connections=[]
        for i,node in enumerate(self.nodes):
            edge=self.steer(node.state,self.goal,env)
            if edge is not None:connections.append((node.cost+edge[0],i,edge[1]))
        if not connections:raise MethodFailure('planning_failure',f'RRT* found no goal connection in {cfg.iterations} iterations')
        cost,parent,final=min(connections,key=lambda x:x[0]);chain=[];cur=parent
        while self.nodes[cur].parent is not None:chain.append(self.nodes[cur].controls);cur=self.nodes[cur].parent
        chain.reverse();chain.append(final)
        self.plan_controls=np.vstack(chain) if any(len(e) for e in chain) else np.empty((0,2))
        states=[env.state.copy()]
        for u in self.plan_controls:
            if not interval_safe(env,states[-1],u,cfg.dt):raise MethodFailure('planning_failure','Stored steering edge violates interval clearance')
            states.append(env.robot.advance(states[-1],u,cfg.dt))
        self.plan_states=np.array(states)
        self.record(stage='planning',nodes=len(self.nodes),rewires=self.rewires,planned_duration=cost,seed=cfg.seed)

    def control(self,observation,env):
        index=int(round((observation['time']-self.start_time)/self.config.dt))
        if index>=len(self.plan_controls):return np.zeros(2)
        expected=self.plan_states[index];q=observation['state']
        if np.linalg.norm(q[:2]-expected[:2])>self.config.tracking_tolerance or abs(wrap(q[2]-expected[2]))>self.config.tracking_tolerance:
            raise MethodFailure('tracking_failure','RRT schedule diverged from planned trajectory; reset/replan')
        self.record(stage='execution',edge_sample=index)
        return self.plan_controls[index].copy()
