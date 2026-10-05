"""Planar obstacle geometry and differential-drive research simulator (SI units)."""
from dataclasses import dataclass, field
from typing import Callable
import json
import numpy as np


def vector(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f'Expected {n} finite numbers')
    return a


def positive(value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Dimensions must be finite and positive')
    return float(value)


def point_segment(p, a, b):
    d = b-a
    return a + np.clip(np.dot(p-a, d)/np.dot(d, d), 0, 1)*d


def segment_distance(a, b, c, d):
    """Exact distance between two closed 2-D segments, including intersections."""
    def cross(u, v): return u[0]*v[1]-u[1]*v[0]
    ab, cd = b-a, d-c
    den = cross(ab, cd)
    if abs(den) > 1e-14:
        t, s = cross(c-a, cd)/den, cross(c-a, ab)/den
        if 0 <= t <= 1 and 0 <= s <= 1: return 0.0
    def dist(p, x, y):
        if np.linalg.norm(y-x) == 0: return np.linalg.norm(p-x)
        return np.linalg.norm(p-point_segment(p,x,y))
    return min(dist(a,c,d),dist(b,c,d),dist(c,a,b),dist(d,a,b))


@dataclass
class Obstacle:
    """Circle or simple polygon. h is signed Euclidean boundary distance."""
    name: str
    center: np.ndarray
    radius: float | None = None
    vertices: np.ndarray | None = None

    def signed_distance(self, p):
        p = vector(p, 2)
        if self.radius is not None: return float(np.linalg.norm(p-self.center)-self.radius)
        v = self.vertices
        dist = min(np.linalg.norm(p-point_segment(p,a,b)) for a,b in zip(v,np.roll(v,-1,axis=0)))
        inside = False
        for a,b in zip(v,np.roll(v,-1,axis=0)):
            if (a[1]>p[1]) != (b[1]>p[1]):
                if p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]: inside = not inside
        return float(-dist if inside else dist)

    def distance(self, p):
        """Distance to occupied set; zero inside, as in the manuscript."""
        return max(0.0,self.signed_distance(p))

    def segment_distance(self, a, b):
        a,b = vector(a,2),vector(b,2)
        if self.radius is not None:
            q = a if np.array_equal(a,b) else point_segment(self.center,a,b)
            return max(0.0,float(np.linalg.norm(q-self.center)-self.radius))
        if self.signed_distance(a)<=0 or self.signed_distance(b)<=0: return 0.0
        return min(segment_distance(a,b,c,d) for c,d in zip(self.vertices,np.roll(self.vertices,-1,axis=0)))

    def gradient(self,p,eps=1e-6):
        """Numerical h gradient; polygon corners/medial axes may be nonsmooth."""
        p=vector(p,2)
        return np.array([(self.signed_distance(p+eps*e)-self.signed_distance(p-eps*e))/(2*eps) for e in np.eye(2)])


def shape(kind, center=(0,0), *, angle=0.0, name=None, radius=0.5,
          width=1.0, height=1.0, thickness=0.25, inner_radius=None, points=5):
    """Create a shape, rotate counterclockwise in radians, then translate.

    U opens upward; T has its crossbar at the top. Polygon centers are
    local bounding-box origins (star center is its radial origin).
    """
    center=vector(center,2)
    if not np.isfinite(angle): raise ValueError('angle must be finite')
    kind=kind.lower()
    if kind=='circle': return Obstacle(name or kind,center,positive(radius))
    w,h=positive(width),positive(height)
    if kind=='square': h=w
    if kind in ('square','rectangle'): v=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
    elif kind in ('t','u'):
        t=positive(thickness)
        if t>=min(w/2,h): raise ValueError('thickness must be less than width/2 and height')
        if kind=='t': v=[(-t/2,-h/2),(t/2,-h/2),(t/2,h/2-t),(w/2,h/2-t),(w/2,h/2),(-w/2,h/2),(-w/2,h/2-t),(-t/2,h/2-t)]
        else: v=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(w/2-t,h/2),(w/2-t,-h/2+t),(-w/2+t,-h/2+t),(-w/2+t,h/2),(-w/2,h/2)]
    elif kind=='star':
        radius=positive(radius); inner_radius=positive(inner_radius if inner_radius is not None else radius*0.45)
        if inner_radius>=radius or not isinstance(points,int) or points<3: raise ValueError('Star needs >=3 points and inner_radius < radius')
        theta=np.pi/2+np.arange(2*points)*np.pi/points
        r=np.where(np.arange(2*points)%2==0,radius,inner_radius)
        v=np.column_stack((r*np.cos(theta),r*np.sin(theta)))
    else: raise ValueError('Choose circle, square, rectangle, T, star, or U')
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return Obstacle(name or kind,center,vertices=np.asarray(v)@rot.T+center)


@dataclass(frozen=True)
class Robot:
    """Tiriolo & Lucia (2023), equations 5, 8, 12, 13, 19.

    Inputs ordered [omega_R, omega_L]. Defaults match the Khepera IV
    experiment except footprint and lookahead, which are configurable choices.
    """
    wheel_radius: float = 0.021
    axle_length: float = 0.0884
    max_wheel_speed: float = 10.0
    body_radius: float = 0.07
    lookahead: float = 0.05

    def __post_init__(self):
        for v in self.__dict__.values(): positive(v)

    def wheels_to_twist(self,wheels):
        r,l=vector(wheels,2)
        return np.array([self.wheel_radius*(r+l)/2,self.wheel_radius*(r-l)/self.axle_length])

    def twist_to_wheels(self,twist):
        v,w=vector(twist,2)
        return np.array([(v+self.axle_length*w/2)/self.wheel_radius,(v-self.axle_length*w/2)/self.wheel_radius])

    def output(self,state):
        x,y,t=vector(state,3)
        return np.array([x,y])+self.lookahead*np.array([np.cos(t),np.sin(t)])

    def input_matrix(self,theta):
        c,s=np.cos(theta),np.sin(theta)
        return np.array([[c,s],[-s/self.lookahead,c/self.lookahead]])

    def linear_to_wheels(self,state,u):
        return self.twist_to_wheels(self.input_matrix(vector(state,3)[2])@vector(u,2))

    def H(self,theta):
        """H(theta) u <= 1: exact orientation-dependent linear-input limits."""
        M=np.column_stack([self.twist_to_wheels(self.input_matrix(theta)@e) for e in np.eye(2)])
        return np.vstack((M,-M))/self.max_wheel_speed

    def f(self,state): return np.zeros(3)

    def g(self,state):
        t=vector(state,3)[2]; r=self.wheel_radius; d=self.axle_length
        return np.array([[r*np.cos(t)/2,r*np.cos(t)/2],[r*np.sin(t)/2,r*np.sin(t)/2],[r/d,-r/d]])

    def advance(self,state,wheels,dt,integrator='exact'):
        state=vector(state,3); positive(dt)
        v,w=self.wheels_to_twist(wheels); x,y,t=state
        if integrator=='euler': return state+dt*(self.g(state)@vector(wheels,2))
        if integrator!='exact': raise ValueError('integrator must be exact or euler')
        # Stable exact zero-order-hold unicycle flow, including w=0.
        travel=v*dt*np.sinc(w*dt/(2*np.pi)); mid=t+w*dt/2
        return np.array([x+travel*np.cos(mid),y+travel*np.sin(mid),t+w*dt])


@dataclass
class Environment:
    robot: Robot = field(default_factory=Robot)
    obstacles: list = field(default_factory=list)
    clearance: float = 0.02
    bounds: tuple = (-2,2,-2,2)
    state: np.ndarray = field(default_factory=lambda:np.zeros(3))
    time: float = 0.0

    def __post_init__(self):
        self.state=vector(self.state,3).copy()
        if not np.isfinite(self.clearance) or self.clearance<0: raise ValueError('clearance must be nonnegative')
        b=vector(self.bounds,4)
        if b[0]>=b[1] or b[2]>=b[3]: raise ValueError('Invalid plotting/planning bounds')

    @property
    def safety_radius(self): return self.robot.body_radius+self.clearance

    def add(self,kind,center=(0,0),**kwargs):
        obstacle=shape(kind,center,**kwargs); self.obstacles.append(obstacle); return obstacle

    def margins(self,p): return np.array([o.signed_distance(p)-self.safety_radius for o in self.obstacles])

    def is_free(self,p): return bool(np.all(self.margins(p)>=0))

    def segment_free(self,a,b,extra=0.0):
        if extra<0: raise ValueError('extra margin must be nonnegative')
        return all(o.segment_distance(a,b)>=self.safety_radius+extra for o in self.obstacles)

    def reset(self,state=(0,0,0)):
        state=vector(state,3)
        if not self.is_free(state[:2]): raise ValueError('Initial robot disk violates clearance')
        self.state=state.copy(); self.time=0.0
        return self.observe()

    def observe(self):
        return {'time':self.time,'state':self.state.copy(),'position':self.state[:2].copy(),
                'output':self.robot.output(self.state),'margins':self.margins(self.state[:2])}

    def step(self,command,dt=0.05,mode='wheels',integrator='exact',saturate=True):
        """Apply held input. Reject unsafe entire intervals without changing state.

        A straight chord is checked exactly. For exact curved motion, inflate
        by the arc/chord deviation bound |v*w| dt²/8. Conservative rejection
        is possible. This guard is an execution check, not a CBF controller.
        """
        positive(dt); command=vector(command,2)
        if mode=='wheels': requested=command
        elif mode=='twist': requested=self.robot.twist_to_wheels(command)
        elif mode=='linear': requested=self.robot.linear_to_wheels(self.state,command)
        else: raise ValueError('mode must be wheels, twist, or linear')
        if not saturate and np.any(np.abs(requested)>self.robot.max_wheel_speed+1e-12): raise ValueError('Wheel limits exceeded')
        applied=np.clip(requested,-self.robot.max_wheel_speed,self.robot.max_wheel_speed)
        nxt=self.robot.advance(self.state,applied,dt,integrator)
        v,w=self.robot.wheels_to_twist(applied)
        extra=abs(v*w)*dt*dt/8 if integrator=='exact' else 0.0
        safe=self.segment_free(self.state[:2],nxt[:2],extra)
        info={'accepted':safe,'requested_wheels':requested.copy(),'applied_wheels':applied.copy(),
              'saturated':not np.allclose(requested,applied),'candidate_state':nxt.copy()}
        if safe: self.state=nxt; self.time+=dt
        return self.observe(),info

    def save(self,path):
        data={'robot':self.robot.__dict__,'clearance':self.clearance,'bounds':self.bounds,'state':self.state.tolist(),
              'obstacles':[{'name':o.name,'center':o.center.tolist(),'radius':o.radius,'vertices':None if o.vertices is None else o.vertices.tolist()} for o in self.obstacles]}
        with open(path,'w') as f: json.dump(data,f,indent=2)

    @classmethod
    def load(cls,path):
        with open(path) as f: d=json.load(f)
        obstacles=[Obstacle(o['name'],vector(o['center'],2),o['radius'],None if o['vertices'] is None else np.asarray(o['vertices'])) for o in d.pop('obstacles')]
        return cls(robot=Robot(**d.pop('robot')),obstacles=obstacles,**d)


def simulate(env,controller:Callable,horizon=20.0,dt=0.05,mode='twist',goal=None,tolerance=0.05,integrator='exact'):
    """controller(observation, env) -> length-2 command; logs aligned intervals.

    Rejected candidates are recorded separately; states contain accepted states.
    Controller internal memory should be reset before each experiment.
    """
    positive(horizon); positive(dt); positive(tolerance)
    goal=None if goal is None else vector(goal,3)
    times=[env.time]; states=[env.state.copy()]; outputs=[env.robot.output(env.state)]
    commands=[]; wheels=[]; margins=[env.margins(env.state[:2])]; timings=[]; attempts=[]
    import time
    start=env.time; status='horizon'
    def arrived():
        return goal is not None and np.linalg.norm(env.state[:2]-goal[:2])<=tolerance and abs(np.arctan2(np.sin(env.state[2]-goal[2]),np.cos(env.state[2]-goal[2])))<=tolerance
    while env.time-start < horizon-1e-12:
        if arrived(): status='goal'; break
        tic=time.perf_counter(); command=vector(controller(env.observe(),env),2); elapsed=time.perf_counter()-tic
        obs,info=env.step(command,min(dt,horizon-(env.time-start)),mode,integrator)
        attempts.append(info)
        if not info['accepted']: status='collision_guard'; break
        commands.append(command); wheels.append(info['applied_wheels']);timings.append(elapsed)
        times.append(obs['time']);states.append(obs['state']);outputs.append(obs['output']);margins.append(obs['margins'])
    if arrived(): status='goal'
    return Rollout(np.array(times),np.array(states),np.array(outputs),np.asarray(commands).reshape(-1,2),np.asarray(wheels).reshape(-1,2),np.array(margins),np.array(timings),status,attempts)


@dataclass
class Rollout:
    times: np.ndarray
    states: np.ndarray
    outputs: np.ndarray
    commands: np.ndarray
    wheels: np.ndarray
    margins: np.ndarray
    controller_seconds: np.ndarray
    status: str
    attempts: list

    def metrics(self):
        return {'status':self.status,'elapsed_time':float(self.times[-1]-self.times[0]),
                'path_length':float(np.linalg.norm(np.diff(self.states[:,:2],axis=0),axis=1).sum()),
                'minimum_margin':float(self.margins.min()) if self.margins.size else float('inf'),
                'mean_controller_seconds':float(self.controller_seconds.mean()) if len(self.controller_seconds) else 0.0}

    def save(self,path):
        np.savez_compressed(path,times=self.times,states=self.states,outputs=self.outputs,commands=self.commands,wheels=self.wheels,margins=self.margins,controller_seconds=self.controller_seconds,status=self.status)
