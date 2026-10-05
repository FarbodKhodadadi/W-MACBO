"""Uniform reproducible experiments, logging failures without hiding them."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import json
import time
import platform
import numpy as np
import scipy
from robot_env.core import Rollout, vector, positive
from .common import MethodFailure, pose_error, signature


@dataclass
class Experiment:
    rollout: Rollout
    method: str
    config: dict
    goal: np.ndarray
    setup_seconds: float
    failure: str | None
    diagnostics: list
    map_signature: str
    final_position_error: float
    final_heading_error: float
    scene: dict
    run_options: dict

    def metrics(self):
        values=self.rollout.metrics()
        values.update(method=self.method,setup_seconds=self.setup_seconds,failure=self.failure,
                      final_position_error=self.final_position_error,final_heading_error=self.final_heading_error,
                      attempted_intervals=len(self.rollout.attempts),solver_calls=len(self.rollout.controller_seconds))
        return values

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.rollout.save(directory/'rollout.npz')
        (directory/'scene.json').write_text(json.dumps(self.scene,indent=2))
        report={'metrics':self.metrics(),'config':self.config,'goal':self.goal.tolist(),'map_signature':self.map_signature,
                'diagnostics':self.diagnostics,'run_options':self.run_options,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
                'attempts':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in item.items()} for item in self.rollout.attempts]}
        def clean(v):
            if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [clean(x) for x in v]
            if isinstance(v,np.generic):return clean(v.item())
            if isinstance(v,float) and not np.isfinite(v):return None
            return v
        (directory/'report.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False))


def run_method(environment,method,*,start=None,horizon=30.0,position_tolerance=None,heading_tolerance=None,copy_environment=True):
    """Reset method memory, prepare, then execute using its fixed dt and wheels.

    Environment copying is on by default. Setup and online timings are separate.
    Expected algorithmic failures become statuses; unexpected programming errors
    propagate. No solver failure switches to another controller. Partial last
    intervals are omitted (reported elapsed horizon may be shorter than requested).
    """
    if position_tolerance is None:position_tolerance=getattr(method.config,'target_position_tolerance',0.05)
    if heading_tolerance is None:heading_tolerance=getattr(method.config,'target_heading_tolerance',0.05)
    positive(horizon);positive(position_tolerance);positive(heading_tolerance)
    env=copy.deepcopy(environment) if copy_environment else environment
    env.reset(env.state if start is None else vector(start,3));method.reset()
    scene={'robot':dict(env.robot.__dict__),'clearance':env.clearance,'bounds':list(env.bounds),'state':env.state.tolist(),
           'obstacles':[{'name':o.name,'center':o.center.tolist(),'radius':o.radius,'vertices':None if o.vertices is None else o.vertices.tolist()} for o in env.obstacles]}
    times=[env.time];states=[env.state.copy()];outputs=[env.robot.output(env.state)];margins=[env.margins(env.state[:2])]
    commands=[];wheels=[];seconds=[];attempts=[];failure=None;status='horizon'
    began=time.perf_counter()
    try:method.prepare(env)
    except MethodFailure as exc:status=exc.status;failure=str(exc)
    setup_seconds=time.perf_counter()-began
    if failure is None:
        count=int(np.floor(horizon/method.config.dt+1e-10))
        for _ in range(count):
            error=pose_error(env.state,method.goal)
            if np.linalg.norm(error[:2])<=position_tolerance and abs(error[2])<=heading_tolerance:status='goal';break
            began=time.perf_counter()
            try:command=vector(method(env.observe(),env),2)
            except MethodFailure as exc:
                seconds.append(time.perf_counter()-began);status=exc.status;failure=str(exc);break
            seconds.append(time.perf_counter()-began)
            if np.any(np.abs(command)>env.robot.max_wheel_speed+1e-7):
                status='invalid_command';failure='Method returned wheel speeds outside physical limits';break
            obs,info=env.step(command,method.config.dt,mode='wheels',saturate=False)
            attempts.append(info)
            if not info['accepted']:status='collision_guard';failure='Environment rejected held-input interval';break
            commands.append(command);wheels.append(info['applied_wheels'])
            times.append(obs['time']);states.append(obs['state']);outputs.append(obs['output']);margins.append(obs['margins'])
        error=pose_error(env.state,method.goal)
        if np.linalg.norm(error[:2])<=position_tolerance and abs(error[2])<=heading_tolerance:status='goal'
    error=pose_error(env.state,method.goal)
    run=Rollout(np.array(times),np.array(states),np.array(outputs),np.asarray(commands).reshape(-1,2),np.asarray(wheels).reshape(-1,2),
                np.array(margins),np.asarray(seconds),status,attempts)
    return Experiment(run,method.name,asdict(method.config),method.goal.copy(),setup_seconds,failure,copy.deepcopy(method.diagnostics),
                      signature(env),float(np.linalg.norm(error[:2])),float(abs(error[2])),scene,
                      {'horizon':horizon,'position_tolerance':position_tolerance,'heading_tolerance':heading_tolerance,'integrator':'exact'})
