"""Time-weighted, common-objective scoring without solving a replacement QP."""
from dataclasses import replace
import numpy as np
import pandas as pd
from methods import MACBO, MACBOConfig
from methods.macbo.geometry import PolygonGeometry
from methods.macbo.planner import Route
from methods.geometry import signed_distances
from methods.common import pose_error
from .config import HORIZON, LABELS, PARAMETERS
from .scenarios import GOAL


def interval_samples(environment, states, wheels, widths, fractions):
    """Independent exact held-twist positions, vectorized over control intervals."""
    right, left = wheels.T
    speed = environment.robot.wheel_radius*(right+left)/2
    yaw = environment.robot.wheel_radius*(right-left)/environment.robot.axle_length
    t = widths[:,None]*np.asarray(fractions)[None,:]
    travel = speed[:,None]*t*np.sinc(yaw[:,None]*t/(2*np.pi))
    angle = states[:,2,None]+yaw[:,None]*t/2
    return states[:,None,:2]+travel[:,:,None]*np.stack((np.cos(angle),np.sin(angle)),axis=-1)


def scoring_intervals(experiment, horizon=HORIZON):
    """Known zero-input terminal hold only for successful runs; never pad failures."""
    run=experiment.rollout
    states=run.states[:-1].copy(); controls=run.wheels.copy()
    widths=np.diff(run.times)
    held=False
    if run.status=='goal' and run.times[-1]-run.times[0] < horizon-1e-9:
        states=np.vstack((states,run.states[-1]))
        controls=np.vstack((controls,np.zeros(2)))
        widths=np.r_[widths,horizon-(run.times[-1]-run.times[0])]
        held=True
    return states,controls,widths,held


def common_objective_trace(environment, states, controls, widths):
    """Supplied final-stage MACBO objective, applied uniformly to recorded inputs.

    Shared w_j = goal, full-pose V, no hybrid yaw term. The virtual memory is
    replayed from zero for each method using its actual states. Minimal required
    nonnegative CLF slack is scored even if above its bound; violations are
    separately reported. This scores a candidate, it does not assert feasibility.
    Left-endpoint quadrature uses the recorded command intervals. Long zero-input
    terminal holds are subdivided to <= 0.6 s for field/memory evaluation.
    """
    config=replace(MACBOConfig(**PARAMETERS['macbo']),clf_mode='strict')
    evaluator=MACBO(GOAL,config)
    evaluator.geometry=PolygonGeometry(environment,config.circle_sides)
    evaluator.route=Route(np.array([states[0,:2],GOAL[:2]]),np.array([states[0,:2],GOAL[:2]]),0.,config.grid_spacing,0,0)
    evaluator.waypoint_index=1
    records=[]; elapsed=0.
    for state,control,width in zip(states,controls,widths):
        count=max(1,int(np.ceil(width/.6-1e-12)))
        step=width/count
        for sub in range(count):
            q=environment.robot.advance(state,control,sub*step) if sub else state
            field=evaluator.field(q,environment)
            velocity=environment.robot.g(q)@control
            error=pose_error(q,GOAL); V=float(.5*error@error)
            derivative=float(error@velocity)
            slack=max(0.,derivative+config.c*V**config.q)
            transformed=(1-config.q)*(V+config.epsilon_V)**(-config.q)*derivative
            tracking=.5*config.mu_v*np.sum((velocity[:2]-field.desired)**2)
            wheel=.5*config.mu_u*np.sum(control**2)
            smooth=.5*config.mu_delta_u*np.sum((control-evaluator.previous)**2)
            relaxation=.5*config.mu_delta*slack**2
            value=float(tracking+config.mu_t*transformed+wheel+smooth+relaxation)
            records.append(dict(time=elapsed,weight=step,objective=value,V=V,
                clf_derivative=derivative,required_slack=slack,
                slack_cap=config.c_bar*V**config.q,
                above_slack_cap=slack>config.c_bar*V**config.q+1e-7,
                memory=evaluator.memory,control_norm=float(np.linalg.norm(control))))
            decay=np.exp(-config.lambda_memory*step)
            evaluator.memory=evaluator.memory*decay+field.memory_forcing*(-np.expm1(-config.lambda_memory*step))/config.lambda_memory
            evaluator.previous=control.copy();elapsed+=step
    return pd.DataFrame(records)


def optimizer_iterations(diagnostics):
    values=[]
    for item in diagnostics:
        if 'qp_iterations' in item:
            values.append(sum(part['qp_iterations'] for part in item['qp_iterations']))
        elif 'iterations' in item: values.append(item['iterations'])
    return float(np.mean(values)) if values else np.nan


def score_trial(environment,experiment,controller):
    run=experiment.rollout
    states,controls,widths,held=scoring_intervals(experiment)
    duration=float(widths.sum())
    trace=common_objective_trace(environment,states,controls,widths) if duration else pd.DataFrame()
    if duration:
        mean_control=float(np.dot(widths,np.linalg.norm(controls,axis=1))/duration)
        nodes,weights=np.polynomial.legendre.leggauss(5)
        points=interval_samples(environment,states,controls,widths,(nodes+1)/2)
        mean_position=float(np.dot(widths,np.linalg.norm(points,axis=-1)@(weights/2))/duration)
        mean_distance=float(np.dot(widths,np.linalg.norm(points-GOAL[:2],axis=-1)@(weights/2))/duration)
        objective=float(np.dot(trace.weight,trace.objective)/duration)
        violation=float(np.dot(trace.weight,trace.above_slack_cap)/duration)
    else:
        mean_control=mean_position=mean_distance=objective=violation=np.nan
    if len(run.wheels):
        points=interval_samples(environment,run.states[:-1],run.wheels,np.diff(run.times),np.linspace(0,1,21))
        dense_margin=float((signed_distances(points.reshape(-1,2),environment.obstacles)-environment.safety_radius).min())
    else:dense_margin=float(run.margins.min())
    timing=run.controller_seconds
    metrics=experiment.metrics()
    row=dict(method=experiment.method,label=LABELS[experiment.method],status=run.status,
        success=int(run.status=='goal'),end_position_error_m=experiment.final_position_error,
        end_heading_error_rad=experiment.final_heading_error,
        mean_control_norm_rad_s=mean_control,mean_trajectory_norm_m=mean_position,
        mean_goal_distance_m=mean_distance,mean_macbo_objective=objective,
        path_length_m=metrics['path_length'],elapsed_s=metrics['elapsed_time'],
        evaluated_time_s=duration,horizon_coverage=duration/HORIZON,terminal_hold=held,
        mean_online_ms=float(timing.mean()*1000) if len(timing) else np.nan,
        p95_online_ms=float(np.percentile(timing,95)*1000) if len(timing) else np.nan,
        setup_s=experiment.setup_seconds,controller_wall_s=float(experiment.setup_seconds+timing.sum()),
        mean_optimizer_iterations=optimizer_iterations(experiment.diagnostics),
        accepted_intervals=len(run.wheels),attempted_intervals=len(run.attempts),
        rejected_intervals=sum(not item['accepted'] for item in run.attempts),
        dense_minimum_margin_m=dense_margin,objective_slack_violation_fraction=violation,
        failure=experiment.failure)
    # Work proxies are intentionally method-specific, not a shared FLOP count.
    work=dict(method=experiment.method,dt=controller.config.dt,
              optimizer_variables=(2*getattr(controller.config,'horizon_steps',0)+int(experiment.method=='composite_mpc')) if experiment.method in ('nmpc','mpc_dhocbf','composite_mpc') else 3 if experiment.method in ('macbo','clf_cbf_qp') else np.nan,
              astar_expansions=controller.route.expansions if experiment.method=='macbo' and controller.route else np.nan,
              rrt_nodes=len(controller.nodes) if experiment.method=='kinodynamic_rrt_star' else np.nan,
              rrt_rewires=controller.rewires if experiment.method=='kinodynamic_rrt_star' else np.nan,
              hj_grid_nodes=int(np.prod(controller.config.grid_shape)) if experiment.method=='hj_reach_avoid' else np.nan,
              hj_scalar_backups=3*controller.config.horizon_steps*controller.config.control_levels**2*int(np.prod(controller.config.grid_shape)) if experiment.method=='hj_reach_avoid' else np.nan,
              hj_value_storage_mib=sum(getattr(controller,key).nbytes for key in ('reach_values','avoid_values','joint_values'))/1024**2 if experiment.method=='hj_reach_avoid' and controller.joint_values is not None else np.nan)
    return row,trace,work


def aggregate_trials(frame):
    """Equal map weights. Undefined zero-duration means propagate, never disappear."""
    columns=['end_position_error_m','end_heading_error_rad','mean_control_norm_rad_s',
        'mean_trajectory_norm_m','mean_macbo_objective','path_length_m','elapsed_s',
        'mean_online_ms','setup_s','controller_wall_s','mean_optimizer_iterations',
        'horizon_coverage','dense_minimum_margin_m']
    rows=[]
    for name in LABELS:
        group=frame.loc[frame.method==name]
        if len(group)!=4:raise ValueError('Exactly four scenarios are required per method')
        values=group[columns].agg(lambda col:col.mean(skipna=False)).to_dict()
        rows.append(dict(method=name,label=LABELS[name],scenarios=4,successes=int(group.success.sum()),
                         defined_control_scenarios=int(group.mean_control_norm_rad_s.notna().sum()),**values))
    return pd.DataFrame(rows).set_index('method')
