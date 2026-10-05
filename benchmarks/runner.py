"""Single entry point for all 28 trials, validation, scoring, and persistence."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import scipy, seaborn, matplotlib
from methods import make_method, run_method
from .scenarios import make_scenarios, GOAL
from .config import HORIZON, POSITION_TOLERANCE, HEADING_TOLERANCE, LABELS, parameters_for
from .metrics import score_trial, aggregate_trials


@dataclass
class Trial:
    experiment: object
    waypoints: object = None
    graph_path: object = None


def run_benchmark(output_directory,progress=print):
    output=Path(output_directory);output.mkdir(parents=True,exist_ok=True)
    scenes=make_scenarios(); trials={};rows=[];work=[]
    configurations={}
    for scenario,environment in scenes.items():
        trials[scenario]={}
        for name in LABELS:
            progress(f'{scenario} / {LABELS[name]}: running',flush=True)
            controller=make_method(name,GOAL,**parameters_for(name))
            result=run_method(environment,controller,horizon=HORIZON,
                position_tolerance=POSITION_TOLERANCE,heading_tolerance=HEADING_TOLERANCE)
            directory=output/'trials'/scenario/name
            result.save(directory)
            row,trace,counts=score_trial(environment,result,controller)
            row['scenario']=scenario;counts['scenario']=scenario
            rows.append(row);work.append(counts)
            if not trace.empty:trace.to_csv(directory/'common_objective.csv',index=False)
            configurations[name]=asdict(controller.config)
            route=controller.route if name=='macbo' else None
            trial=Trial(result,None if route is None else controller.waypoints,
                        None if route is None else route.graph_path.copy())
            trials[scenario][name]=trial
            if route is not None:
                np.savez_compressed(directory/'route.npz',waypoints=route.waypoints,
                    graph_path=route.graph_path,graph_cost=route.graph_cost,spacing=route.spacing,
                    expansions=route.expansions)
            # Exact accepted trajectories and all swept guards must remain safe.
            assert row['dense_minimum_margin_m']>=-1e-8,(scenario,name,'unsafe accepted motion')
            assert all(item['accepted'] for item in result.rollout.attempts),(scenario,name,'guard rejection')
            assert np.all(np.abs(result.rollout.wheels)<=environment.robot.max_wheel_speed+1e-7)
            assert np.array_equal(environment.state,[-2.,-2.,0.]),'Runner mutated shared scene'
            if name=='macbo':
                assert result.rollout.status=='goal',(scenario,result.failure)
                if scenario=='u_and_t':
                    assert np.linalg.norm(route.waypoints[1]-GOAL[:2])>np.linalg.norm(route.waypoints[0]-GOAL[:2]),'U route must initially leave away from goal'
                for item in result.diagnostics:
                    if item.get('stage')=='control':
                        assert item['clf_derivative']<=item['required_clf_derivative']+1e-7
                        assert item['minimum_barrier_residual'] is None or item['minimum_barrier_residual']>=-1e-7
            progress(f'  {result.rollout.status}; error={result.final_position_error:.4f} m; accepted={len(result.rollout.wheels)}',flush=True)
            pd.DataFrame(rows).to_csv(output/'per_trial_metrics.csv',index=False)
            # Value tensors are not retained for four maps simultaneously.
            del controller
    frame=pd.DataFrame(rows);summary=aggregate_trials(frame);complexity=pd.DataFrame(work)
    summary.to_csv(output/'mean_metrics.csv');complexity.to_csv(output/'work_proxies.csv',index=False)
    source_root=Path(__file__).resolve().parents[1]
    sources={str(path.relative_to(source_root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('robot_env','methods','benchmarks') for path in (source_root/folder).rglob('*.py')}
    manifest=dict(start=[-2.,-2.,0.],goal=GOAL.tolist(),horizon=HORIZON,
        position_tolerance=POSITION_TOLERANCE,heading_tolerance=HEADING_TOLERANCE,
        configurations=configurations,source_sha256=sources,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                      pandas=pd.__version__,seaborn=seaborn.__version__,matplotlib=matplotlib.__version__),
        scoring='Common goal-referenced final-stage MACBO; known zero-input terminal holds after success; failed horizons are censored; undefined zero-duration means propagate across maps.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return scenes,trials,frame,summary,complexity


def load_benchmark(output_directory, *, historical=False):
    """Load validated trials; historical=True reads an explicitly archived design.

    Historical mode retains saved configurations rather than comparing them
    with current source/settings. Scene identity is still checked. Use only for
    archived comparisons; it is never the default for the active benchmark.
    """
    from robot_env import Environment, Rollout
    from methods.experiment import Experiment
    output=Path(output_directory)
    manifest=json.loads((output/'manifest.json').read_text())
    source_root=Path(__file__).resolve().parents[1]
    for relative,digest in ({} if historical else manifest['source_sha256']).items():
        if hashlib.sha256((source_root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source changed: {relative}; set RECOMPUTE=True')
    scenes=make_scenarios();trials={}
    for scenario in scenes:
        trials[scenario]={}
        for name in LABELS:
            directory=output/'trials'/scenario/name
            report=json.loads((directory/'report.json').read_text())
            expected=asdict(make_method(name,GOAL,**parameters_for(name)).config)
            if not historical and json.dumps(report['config'],sort_keys=True)!=json.dumps(expected,sort_keys=True):
                raise ValueError(f'Config changed: {name}; set RECOMPUTE=True')
            scene=json.loads((directory/'scene.json').read_text())
            restored=Environment.load(directory/'scene.json')
            # Compare the complete scene using the same serialization contract.
            from methods.common import signature
            if signature(restored)!=signature(scenes[scenario]):
                raise ValueError(f'Map changed: {scenario}; set RECOMPUTE=True')
            with np.load(directory/'rollout.npz') as arrays:
                run=Rollout(*(arrays[key].copy() for key in ('times','states','outputs','commands','wheels','margins','controller_seconds')),
                            str(arrays['status']),report['attempts'])
            metrics=report['metrics']
            result=Experiment(run,name,report['config'],np.array(report['goal']),
                metrics['setup_seconds'],metrics['failure'],report['diagnostics'],
                report['map_signature'],metrics['final_position_error'],metrics['final_heading_error'],
                scene,report['run_options'])
            waypoint=graph=None
            if name=='macbo':
                with np.load(directory/'route.npz') as arrays:
                    waypoint=arrays['waypoints'].copy();graph=arrays['graph_path'].copy()
            trials[scenario][name]=Trial(result,waypoint,graph)
    return scenes,trials,pd.read_csv(output/'per_trial_metrics.csv'),pd.read_csv(output/'mean_metrics.csv',index_col='method'),pd.read_csv(output/'work_proxies.csv')
