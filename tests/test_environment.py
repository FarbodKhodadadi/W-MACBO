import unittest
import tempfile
from pathlib import Path
import numpy as np
from robot_env import *

class EnvironmentTests(unittest.TestCase):
    def test_shapes_and_concavity(self):
        for kind in ['circle','square','T','star','U']:
            o=shape(kind,(2,3));self.assertGreater(o.signed_distance((20,20)),0)
        u=shape('U',width=2,height=2,thickness=.25)
        self.assertGreater(u.signed_distance((0,.5)),0)
        self.assertLess(u.signed_distance((.9,.5)),0)
        self.assertLess(shape('T').signed_distance((0,0)),0)
    def test_transforms_and_H(self):
        r=Robot()
        for theta in np.linspace(-3,3,13):
            wheels=np.array([2.,-3.]);twist=r.wheels_to_twist(wheels)
            np.testing.assert_allclose(r.twist_to_wheels(twist),wheels)
            u=np.linalg.solve(r.input_matrix(theta),twist)
            np.testing.assert_allclose(r.linear_to_wheels([0,0,theta],u),wheels)
            self.assertTrue(np.all(r.H(theta)@u<=1))
            q=np.array([0.,0.,theta]);eps=1e-7
            derivative=(r.output(r.advance(q,wheels,eps))-r.output(q))/eps
            np.testing.assert_allclose(derivative,u,atol=1e-8)
    def test_exact_and_euler(self):
        r=Robot();np.testing.assert_allclose(r.advance([0,0,0],[5,5],1),[.105,0,0])
        np.testing.assert_allclose(r.advance([0,0,0],[5,-5],1)[:2],[0,0])
        np.testing.assert_allclose(r.advance([0,0,0],[5,5],1,'euler'),[.105,0,0],atol=1e-14)
    def test_interval_guard(self):
        e=Environment(robot=Robot(wheel_radius=1,body_radius=.05),clearance=0)
        e.add('circle',(0,0),radius=.1);e.reset((-1,0,0))
        _,info=e.step([2,2],dt=1)
        self.assertFalse(info['accepted']);self.assertEqual(e.time,0)
        np.testing.assert_allclose(e.state,[-1,0,0])
    def test_segments(self):
        e=Environment();e.add('square',width=1)
        self.assertFalse(e.segment_free([-1,0],[1,0]))
        self.assertTrue(e.segment_free([-1,1],[1,1]))
    def test_roundtrip_and_rollout(self):
        e=Environment();e.add('star',(1,1),radius=.3);e.reset((-1,-1,0))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scene.json';e.save(p);f=Environment.load(p)
            np.testing.assert_allclose(e.obstacles[0].vertices,f.obstacles[0].vertices)
        run=simulate(e,lambda obs,env:[.1,0],horizon=.23,dt=.1)
        self.assertEqual(run.states.shape[0],run.wheels.shape[0]+1)
        self.assertAlmostEqual(run.times[-1],.23)
    def test_route_and_goal(self):
        e=Environment(bounds=(-1,1,-1,1));e.add('square',width=.4);e.reset((-.8,0,0))
        route=astar(e,e.state[:2],[.8,0],spacing=.15)
        self.assertTrue(all(e.segment_free(a,b) for a,b in zip(route[:-1],route[1:])))
        run=simulate(e,WaypointFollower(route),horizon=40,goal=[.8,0,0])
        self.assertEqual(run.status,'goal');self.assertGreaterEqual(run.metrics()['minimum_margin'],0)

if __name__=='__main__':unittest.main()
