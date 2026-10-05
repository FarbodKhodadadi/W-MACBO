"""Physical derivatives, censored means, and turn-resolved GIF reconstruction."""
import unittest
from types import SimpleNamespace
import numpy as np
from robot_env import Environment, Rollout
from benchmarks.config import LABELS
from benchmarks.dynamics import executed_dynamics_table
from benchmarks.plotting import motion_frame_times, pose_at


def rollout(env,u,dt,status='goal'):
    q=env.state.copy();end=env.robot.advance(q,u,dt)
    return Rollout(np.array([0.,dt]),np.array([q,end]),np.zeros((2,2)),
        np.array([u]),np.array([u]),np.ones((2,1)),np.zeros(1),status,[])


class DynamicsAnimationTests(unittest.TestCase):
    def test_actual_derivative_matches_exact_motion(self):
        env=Environment();q=np.array([.3,-.4,.9]);u=np.array([3.,-1.])
        derivative=env.robot.f(q)+env.robot.g(q)@u
        h=1e-5
        central=(env.robot.advance(q,u,h)-env.robot.advance(q,-u,h))/(2*h)
        np.testing.assert_allclose(central,derivative,atol=1e-10)
        self.assertGreater(np.linalg.norm(derivative),0)

    def test_dynamics_means_use_actual_controls_and_success_hold(self):
        env=Environment();u=[2.,1.];run=rollout(env,u,2.)
        exp=SimpleNamespace(rollout=run)
        trials={'map':{name:SimpleNamespace(experiment=exp) for name in LABELS}}
        frame=executed_dynamics_table({'map':env},trials)
        norm=np.linalg.norm(env.robot.g(run.states[0])@u)
        np.testing.assert_allclose(frame.mean_actual_dynamics_norm,norm*2/120)
        run.status='solver_failure'
        frame=executed_dynamics_table({'map':env},trials)
        np.testing.assert_allclose(frame.mean_actual_dynamics_norm,norm)
        self.assertTrue((frame.mean_f_norm==0).all())

    def test_fast_turn_is_resolved_with_exact_poses(self):
        env=Environment();run=rollout(env,[10.,-10.],.6)
        times=motion_frame_times(env,run)
        poses=np.array([pose_at(env,run,t) for t in times])
        self.assertEqual(times[0],0);self.assertEqual(times[-1],.6)
        self.assertTrue((np.diff(times)<=.4+1e-12).all())
        self.assertTrue((np.abs(np.diff(poses[:,2]))<=.25+1e-12).all())
        np.testing.assert_allclose(poses[-1],run.states[-1],atol=1e-14)
        np.testing.assert_allclose(poses[:,:2],np.tile(run.states[0,:2],(len(times),1)))

    def test_zero_motion_failure_is_undefined(self):
        env=Environment();run=rollout(env,[0.,0.],1.)
        run.times=run.times[:1];run.states=run.states[:1];run.wheels=np.empty((0,2))
        run.status='solver_failure';exp=SimpleNamespace(rollout=run)
        trials={'map':{name:SimpleNamespace(experiment=exp) for name in LABELS}}
        frame=executed_dynamics_table({'map':env},trials)
        self.assertTrue(frame.mean_actual_dynamics_norm.isna().all())
        self.assertTrue((frame.mean_g_norm>0).all())
        np.testing.assert_array_equal(motion_frame_times(env,run),[0.])

if __name__=='__main__':unittest.main()
