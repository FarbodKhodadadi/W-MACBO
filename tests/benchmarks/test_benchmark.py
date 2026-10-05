"""Benchmark contracts: maps, time weighting, objective, failure accounting."""
import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pandas as pd
from robot_env import Environment, Rollout
from methods import MACBOConfig
from methods.common import pose_error
from benchmarks.scenarios import make_scenarios, START, GOAL
from benchmarks.config import PARAMETERS, LABELS
from benchmarks.metrics import scoring_intervals, common_objective_trace, aggregate_trials, interval_samples


class BenchmarkTests(unittest.TestCase):
    def test_four_fixed_shared_maps(self):
        scenes=make_scenarios()
        self.assertEqual(len(scenes),4)
        for env in scenes.values():
            np.testing.assert_array_equal(env.state,START)
            self.assertTrue(env.is_free(GOAL[:2]))
        self.assertFalse(scenes['u_and_t'].segment_free(START[:2],GOAL[:2]))
        self.assertTrue(scenes['square_passage'].segment_free(START[:2],GOAL[:2]))
        u=scenes['u_and_t'].obstacles[0]
        opening=np.array([-1.,-1.])/np.sqrt(2)
        self.assertTrue(scenes['u_and_t'].segment_free(START[:2],START[:2]+.9*opening))
        self.assertEqual(u.radius,None)

    def test_only_success_is_extended_with_zero_input(self):
        run=Rollout(np.array([0.,1.]),np.array([START,START]),np.zeros((2,2)),
                    np.array([[1.,1.]]),np.array([[1.,1.]]),np.ones((2,1)),np.zeros(1),'solver_failure',[])
        exp=SimpleNamespace(rollout=run)
        _,u,w,held=scoring_intervals(exp,horizon=3)
        self.assertFalse(held);self.assertEqual(w.sum(),1)
        run.status='goal'
        _,u,w,held=scoring_intervals(exp,horizon=3)
        self.assertTrue(held);self.assertEqual(w.sum(),3)
        np.testing.assert_array_equal(u[-1],[0,0])

    def test_independent_exact_motion_matches_core(self):
        env=Environment();q=np.array([[1.,2.,.3],[-1.,0.,-.7]])
        wheels=np.array([[3.,2.],[-2.,-2.]])
        widths=np.array([.2,.5]);fractions=np.array([0.,.25,1.])
        p=interval_samples(env,q,wheels,widths,fractions)
        for i in range(2):
            for j,fraction in enumerate(fractions):
                expected=q[i,:2] if fraction==0 else env.robot.advance(q[i],wheels[i],widths[i]*fraction)[:2]
                np.testing.assert_allclose(p[i,j],expected,atol=1e-14)

    def test_common_objective_matches_every_supplied_term(self):
        env=Environment();q=np.array([[1.,1.,.4]]);u=np.array([[2.,1.]])
        trace=common_objective_trace(env,q,u,np.array([.1]))
        cfg=replace(MACBOConfig(**PARAMETERS['macbo']),clf_mode='strict')
        e=pose_error(q[0],GOAL);V=.5*e@e;velocity=env.robot.g(q[0])@u[0]
        derivative=e@velocity;vd=(cfg.k_w+cfg.k_g)*(GOAL[:2]-q[0,:2])
        delta=max(0.,derivative+cfg.c*V**cfg.q)
        value=(.5*cfg.mu_v*np.sum((velocity[:2]-vd)**2)
               +cfg.mu_t*(1-cfg.q)*(V+cfg.epsilon_V)**(-cfg.q)*derivative
               +.5*(cfg.mu_u+cfg.mu_delta_u)*np.sum(u[0]**2)
               +.5*cfg.mu_delta*delta**2)
        self.assertAlmostEqual(trace.objective.iloc[0],value,places=11)
        self.assertEqual(trace.required_slack.iloc[0],delta)

    def test_undefined_trial_is_not_silently_dropped_from_four_map_mean(self):
        cols=['end_position_error_m','end_heading_error_rad','mean_control_norm_rad_s',
        'mean_trajectory_norm_m','mean_macbo_objective','path_length_m','elapsed_s',
        'mean_online_ms','setup_s','controller_wall_s','mean_optimizer_iterations',
        'horizon_coverage','dense_minimum_margin_m']
        rows=[]
        for name in LABELS:
            for i in range(4):
                row=dict(method=name,success=1,**{c:float(i) for c in cols})
                if name=='clf_cbf_qp' and i==0:row['mean_control_norm_rad_s']=np.nan
                rows.append(row)
        summary=aggregate_trials(pd.DataFrame(rows))
        self.assertTrue(np.isnan(summary.loc['clf_cbf_qp','mean_control_norm_rad_s']))
        self.assertEqual(summary.loc['clf_cbf_qp','defined_control_scenarios'],3)
        self.assertEqual(summary.loc['macbo','mean_control_norm_rad_s'],1.5)

if __name__=='__main__':unittest.main()
