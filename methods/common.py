"""Shared method API, immutable configurations, and audited execution."""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from robot_env.core import vector, positive


class MethodFailure(RuntimeError):
    def __init__(self,status,message):
        super().__init__(message)
        self.status=status


def wrap(angle): return np.arctan2(np.sin(angle),np.cos(angle))


def pose_error(q,goal):
    e=np.asarray(q)-goal
    e=np.array(e,copy=True);e[...,2]=wrap(e[...,2]);return e


def signature(env):
    data={'robot':env.robot.__dict__,'clearance':env.clearance,'bounds':list(env.bounds),
          'obstacles':[(o.name,o.center.tolist(),o.radius,None if o.vertices is None else o.vertices.tolist()) for o in env.obstacles]}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def interval_safe(env,q,wheels,dt):
    nxt=env.robot.advance(q,wheels,dt)
    v,w=env.robot.wheels_to_twist(wheels)
    return env.segment_free(q[:2],nxt[:2],abs(v*w)*dt*dt/8)


def scale_wheels(robot,twist):
    wheels=robot.twist_to_wheels(twist)
    return wheels/max(1.,np.max(np.abs(wheels))/robot.max_wheel_speed)


def jacobian(fn,x,eps=1e-5):
    x=np.asarray(x,dtype=float);y=np.asarray(fn(x));J=np.empty((y.size,x.size))
    for j in range(x.size):
        delta=np.zeros_like(x);delta[j]=eps
        J[:,j]=(np.asarray(fn(x+delta)).ravel()-np.asarray(fn(x-delta)).ravel())/(2*eps)
    return J


def validate_weights(weights,n,name):
    a=np.asarray(weights,dtype=float)
    if a.shape!=(n,) or not np.all(np.isfinite(a)) or np.any(a<=0): raise ValueError(f'{name} requires {n} positive finite diagonal weights')


def integer(value,name,minimum=1):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<minimum: raise ValueError(f'{name} must be integer >= {minimum}')


@dataclass(frozen=True)
class BaseConfig:
    dt: float = 0.1
    seed: int = 0
    def __post_init__(self):
        positive(self.dt);integer(self.seed,'seed',0)


class Method:
    mode='wheels'
    name='method'
    def __init__(self,goal,config):
        self.goal=vector(goal,3).copy();self.config=config
        self.reset()

    def reset(self):
        self.prepared=False;self.diagnostics=[];self.last_info={}
        self.rng=np.random.default_rng(self.config.seed)

    def prepare(self,env):
        if not env.is_free(env.state[:2]) or not env.is_free(self.goal[:2]): raise MethodFailure('invalid_endpoint','Start/goal violates physical disk clearance')
        self.map_signature=signature(env);self.prepared=True

    def check(self,env,dt):
        if not self.prepared:self.prepare(env)
        if signature(env)!=self.map_signature:raise MethodFailure('map_changed','Map or robot changed; reset the method before reuse')
        if not np.isclose(dt,self.config.dt,atol=1e-12,rtol=0):raise ValueError('Execution dt must match method.config.dt')

    def record(self,**info):
        self.last_info=info;self.diagnostics.append(info)

    def __call__(self,observation,env):
        """Compatible with robot_env.simulate(..., mode='wheels', dt=config.dt)."""
        self.check(env,self.config.dt)
        return self.control(observation,env)
