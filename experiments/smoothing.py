"""Experimental MACBO parameter trials and robot-dynamics norm diagnostics."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from benchmarks.config import parameters_for, HORIZON, POSITION_TOLERANCE, HEADING_TOLERANCE, LABELS
from benchmarks.scenarios import GOAL
from benchmarks.metrics import interval_samples, scoring_intervals
from methods import make_method,run_method

CANDIDATES = {
    'input_penalty_only': {'mu_delta_u':1.},
    'gentler_motion': {'k_w':.4,'k_g':.02,'mu_delta_u':1.,
                      'max_turn_speed':1.,'heading_gain':3.},
}


def assert_protected_files(root,manifest):
    """Verify every pre-existing source, notebook, and benchmark artifact byte."""
    changed=[path for path,digest in manifest.items()
             if not (Path(root)/path).exists() or hashlib.sha256((Path(root)/path).read_bytes()).hexdigest()!=digest]
    if changed:raise AssertionError(f'Protected files changed: {changed}')


def motion_metrics(environment,rollout):
    """Sampled rate/acceleration proxies; not continuous jerk certificates.

    Endpoint zero commands are included consistently. Rates are averaged over
    N+1 sampled transitions, each the same dt as this fixed-step rollout. No
    terminal-horizon padding dilutes the active-motion roughness measure.
    """
    if not len(rollout.wheels):raise ValueError('Motion metrics require an executed interval')
    widths=np.diff(rollout.times);dt=float(widths[0])
    if not np.allclose(widths,dt,atol=1e-10,rtol=0):raise ValueError('Expected fixed step')
    controls=np.vstack((np.zeros(2),rollout.wheels,np.zeros(2)))
    differences=np.diff(controls,axis=0)
    twists=np.array([environment.robot.wheels_to_twist(u) for u in rollout.wheels])
    velocities=np.array([environment.robot.g(q)@u for q,u in zip(rollout.states[:-1],rollout.wheels)])[:,:2]
    dv=np.diff(np.vstack((np.zeros(2),velocities,np.zeros(2))),axis=0)
    return dict(status=rollout.status,elapsed_s=float(rollout.times[-1]-rollout.times[0]),
        input_rate_rms=float(np.sqrt(np.mean(np.sum((differences/dt)**2,axis=1)))),
        planar_acceleration_rms=float(np.sqrt(np.mean(np.sum((dv/dt)**2,axis=1)))),
        input_total_variation=float(np.linalg.norm(differences,axis=1).sum()),
        peak_input_jump=float(np.linalg.norm(differences,axis=1).max()),
        peak_speed=float(np.abs(twists[:,0]).max()),peak_yaw_rate=float(np.abs(twists[:,1]).max()),
        path_length_m=float(np.linalg.norm(np.diff(rollout.states[:,:2],axis=0),axis=1).sum()),
        end_position_error_m=float(np.linalg.norm(rollout.states[-1,:2]-GOAL[:2])),
        end_heading_error_rad=float(abs(np.arctan2(np.sin(rollout.states[-1,2]-GOAL[2]),np.cos(rollout.states[-1,2]-GOAL[2])))))


def validate_trial(environment,experiment,controller):
    run=experiment.rollout;cfg=controller.config
    assert run.status=='goal',experiment.failure
    assert all(item['accepted'] and not item['saturated'] for item in run.attempts)
    assert np.all(np.abs(run.wheels)<=environment.robot.max_wheel_speed+1e-7)
    # Independent exact points between original samples.
    points=interval_samples(environment,run.states[:-1],run.wheels,np.diff(run.times),np.linspace(0,1,21))
    from methods.geometry import signed_distances
    margin=float((signed_distances(points.reshape(-1,2),environment.obstacles)-environment.safety_radius).min())
    assert margin>=-1e-8
    for item in experiment.diagnostics:
        if item.get('stage')=='control':
            assert item['clf_derivative']<=item['required_clf_derivative']+1e-7
            assert -1e-7<=item['clf_slack']<=cfg.c_bar*item['V']**cfg.q+1e-7
            assert item['minimum_barrier_residual'] is None or item['minimum_barrier_residual']>=-1e-7
            assert item['polygon_interval_margin'] is None or item['polygon_interval_margin']>=-1e-10
    return margin


def run_candidates(scenes,baseline_trials,output):
    """Use only existing MACBO hyperparameters; preserve baseline trajectories."""
    output=Path(output);rows=[];results={}
    for scenario,environment in scenes.items():
        baseline=baseline_trials[scenario]['macbo'].experiment
        rows.append(dict(scenario=scenario,variant='approved_benchmark',
                         **motion_metrics(environment,baseline.rollout),
                         minimum_margin_m=float(baseline.rollout.margins.min())))
        results[scenario]={}
        for variant,overrides in CANDIDATES.items():
            parameters=dict(baseline.config);parameters.update(overrides)
            controller=make_method('macbo',GOAL,**parameters)
            trial=run_method(environment,controller,horizon=HORIZON,
                position_tolerance=POSITION_TOLERANCE,heading_tolerance=HEADING_TOLERANCE)
            margin=validate_trial(environment,trial,controller)
            trial.save(output/'trials'/scenario/variant)
            np.savez_compressed(output/'trials'/scenario/variant/'route.npz',
                                graph_path=controller.route.graph_path,waypoints=controller.waypoints)
            rows.append(dict(scenario=scenario,variant=variant,**motion_metrics(environment,trial.rollout),minimum_margin_m=margin))
            results[scenario][variant]=trial
            print(scenario,variant,trial.rollout.status,f'{trial.rollout.times[-1]:.2f} s',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'motion_metrics.csv',index=False)
    (output/'candidate_overrides.json').write_text(json.dumps(CANDIDATES,indent=2))
    return results,frame


def dynamics_norm_table(scenes,trials):
    """Time-weighted vector L2 / matrix Frobenius norms of the model f and g.

    Their norms are analytically state-independent here. Initial failed runs
    are labelled analytical rather than assigned a fictitious trajectory.
    """
    rows=[]
    for scenario,environment in scenes.items():
        robot=environment.robot
        analytical_g=np.sqrt(robot.wheel_radius**2/2+2*robot.wheel_radius**2/robot.axle_length**2)
        for name in LABELS:
            trial=trials[scenario][name].experiment
            states,_,widths,_=scoring_intervals(trial)
            if len(widths):
                f=np.array([np.linalg.norm(robot.f(q)) for q in states])
                g=np.array([np.linalg.norm(robot.g(q),ord='fro') for q in states])
                mean_f=float(np.average(f,weights=widths));mean_g=float(np.average(g,weights=widths))
                source='time-weighted recorded states / declared terminal hold'
            else:
                mean_f=float(np.linalg.norm(robot.f(trial.rollout.states[0])))
                mean_g=float(np.linalg.norm(robot.g(trial.rollout.states[0]),ord='fro'))
                source='analytical constant; no executed interval'
            np.testing.assert_allclose(mean_f,0,atol=1e-14)
            np.testing.assert_allclose(mean_g,analytical_g,atol=1e-14)
            rows.append(dict(scenario=scenario,method=name,label=LABELS[name],status=trial.rollout.status,
                             mean_f_norm=mean_f,mean_g_norm=mean_g,source=source))
    return pd.DataFrame(rows)
