import unittest
from types import SimpleNamespace
import numpy as np
from robot_env import Robot,Environment,Rollout
from experiments.smoothing import motion_metrics

class SmoothingMetricsTests(unittest.TestCase):
    def test_analytical_dynamics_norms(self):
        robot=Robot()
        expected=np.sqrt(robot.wheel_radius**2/2+2*robot.wheel_radius**2/robot.axle_length**2)
        for theta in np.linspace(-20,20,101):
            self.assertEqual(np.linalg.norm(robot.f([1,2,theta])),0)
            self.assertAlmostEqual(np.linalg.norm(robot.g([1,2,theta]),ord='fro'),expected,places=14)

    def test_rate_metric_includes_start_and_stop(self):
        env=Environment();wheels=np.array([[1.,1.],[2.,2.]])
        run=Rollout(np.array([0.,.1,.2]),np.zeros((3,3)),np.zeros((3,2)),wheels,wheels,np.ones((3,1)),np.zeros(2),'goal',[])
        values=motion_metrics(env,run)
        jumps=np.array([[1,1],[1,1],[-2,-2]])
        self.assertAlmostEqual(values['input_rate_rms'],np.sqrt(np.sum(jumps*jumps)/3)/.1)
        self.assertAlmostEqual(values['input_total_variation'],4*np.sqrt(2))

if __name__=='__main__':unittest.main()
