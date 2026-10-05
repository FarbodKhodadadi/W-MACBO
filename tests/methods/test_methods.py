"""Method invariants, numerical equations, compatibility, and deterministic trials."""
import copy
import tempfile
import unittest
from dataclasses import asdict
import numpy as np
from robot_env import Environment, Robot, shape
from methods import *
from methods.geometry import signed_distances, Barriers, softmin, dhocbf_levels, triangulate
from methods.common import pose_error, signature


def scene():
    e=Environment(bounds=(-.5,.5,-.5,.5));e.reset((-.3,0,0));return e


class GeometryTests(unittest.TestCase):
    def test_vectorized_matches_existing(self):
        rng=np.random.default_rng(9);points=rng.uniform(-1.5,1.5,(70,2))
        for kind in ['circle','square','T','star','U']:
            o=shape(kind,(.2,-.1),angle=.3)
            np.testing.assert_allclose(signed_distances(points,[o])[:,0],[o.signed_distance(p) for p in points],atol=1e-12)

    def test_component_free_set_preserves_concavity(self):
        rng=np.random.default_rng(8)
        for kind in ['T','star','U','square']:
            e=Environment();e.add(kind)
            barriers=Barriers(e);points=rng.uniform(-1,1,(100,2))
            np.testing.assert_array_equal(np.all(barriers.values(points)>=0,axis=1),np.all(signed_distances(points,e.obstacles)>=e.safety_radius,axis=1))
        e=Environment();e.add('U',width=1,height=1,thickness=.15)
        self.assertTrue(np.all(Barriers(e).values([[0,.3]])>=0))

    def test_softmin_conservative_stable(self):
        values=np.array([[1000.,1001.],[-1000.,-999.]])
        actual=softmin(values,80.)
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(actual<=values.min(axis=1)+1e-12))
        np.testing.assert_allclose(softmin([[2.,2.]],10.),2-np.log(2)/10)

    def test_high_order_recursion(self):
        b=np.array([[1.],[.9],[.85],[.8]])
        levels=dhocbf_levels(b,(.2,.3))
        np.testing.assert_allclose(levels[1],b[1:]-.8*b[:-1])
        np.testing.assert_allclose(levels[2],levels[1][1:]-.7*levels[1][:-1])


class MethodTests(unittest.TestCase):
    def test_configuration_validation(self):
        for cls in [RRTStarConfig,CLFCBFConfig,NMPCConfig,DHOCBFConfig,HJConfig,CompositeConfig]:
            with self.assertRaises(ValueError):cls(dt=0)
        with self.assertRaises(ValueError):HJConfig(control_levels=4)
        with self.assertRaises(ValueError):DHOCBFConfig(gammas=(0,))
        with self.assertRaises(ValueError):CLFCBFConfig(P=(0,)*9)
        with self.assertRaises(ValueError):CompositeConfig(max_slack=-1)
        with self.assertRaises(TypeError):make_method('nmpc',[0,0,0],wrong_option=1)

    def test_model_prediction_matches_environment(self):
        e=scene();m=NonlinearMPC([.3,0,0],NMPCConfig(dt=.2,horizon_steps=3));m.prepare(e)
        u=np.array([[.2,.2],[.1,-.1],[.3,.2]])
        predicted=m.predict(e.state,u.ravel(),e)
        for i,command in enumerate(u):
            e.step(command*e.robot.max_wheel_speed,.2,mode='wheels')
            np.testing.assert_allclose(e.state,predicted[i+1])

    def test_qp_residual_and_wheel_limits(self):
        e=scene();e.add('circle',(0,.3),radius=.03)
        m=CLFCBFQP([.3,0,0]);m.prepare(e)
        command=m(e.observe(),e)
        Q,c,A,lo,hi=m.problem(e.state,e)
        z=np.r_[command,m.last_info['clf_slack']]
        self.assertTrue(np.all(A@z>=lo-1e-7));self.assertTrue(np.all(A@z<=hi+1e-7))
        self.assertTrue(np.all(abs(command)<=e.robot.max_wheel_speed))

    def test_rrt_edges_rewire_costs_and_seed(self):
        e=scene();e.add('circle',radius=.04)
        cfg=RRTStarConfig(dt=.2,iterations=60,seed=12,extension_length=.2)
        a=KinodynamicRRTStar([.3,0,0],cfg);b=KinodynamicRRTStar([.3,0,0],cfg)
        a.prepare(e);b.prepare(e)
        np.testing.assert_array_equal(a.plan_controls,b.plan_controls)
        for node in a.nodes[1:]:
            parent=a.nodes[node.parent];q=parent.state.copy()
            for u in node.controls:q=e.robot.advance(q,u,cfg.dt)
            self.assertLess(np.linalg.norm(pose_error(q,node.state)),1e-10)
            self.assertAlmostEqual(node.cost,parent.cost+len(node.controls)*cfg.dt)
        self.assertLess(np.linalg.norm(pose_error(a.plan_states[-1],a.goal)),1e-10)

    def test_hj_value_update_periodicity_and_target(self):
        e=scene();m=HJReachAvoid([.3,0,0],HJConfig(dt=.2,grid_shape=(15,15,12),horizon_steps=4,target_position_tolerance=.1,target_heading_tolerance=.6))
        m.prepare(e)
        q=np.array([[0.,0.,-np.pi],[0.,0.,np.pi]])
        np.testing.assert_allclose(m.interpolate(m.reach_values[-1],q)[0],m.interpolate(m.reach_values[-1],q)[1])
        self.assertTrue(np.all(m.reach_values[1:]<=m.reach_values[:-1]+1e-6))
        np.testing.assert_allclose(m.joint_values[0].ravel(),np.maximum(m.target,m.constraint),atol=1e-7)
        self.assertTrue(np.all(m.avoid_values[-1].ravel()>=m.constraint-1e-6))
        self.assertLess(m.interpolate(m.reach_values[0],m.goal[None,:])[0],0)

    def test_hj_memory_and_unresolved_target_failure(self):
        e=scene();m=HJReachAvoid([.3,0,0],HJConfig(memory_limit_mb=.001))
        r=run_method(e,m,horizon=1)
        self.assertEqual(r.rollout.status,'resource_limit')
        m=HJReachAvoid([.31,.013,.15],HJConfig(grid_shape=(5,5,4),target_position_tolerance=.001,target_heading_tolerance=.001))
        self.assertEqual(run_method(e,m,horizon=1).rollout.status,'target_unresolved')

    def test_mpc_constraints_and_composite_scalar(self):
        e=scene();e.add('circle',(0,.3),radius=.03)
        for name in ['nmpc','mpc_dhocbf','composite_mpc']:
            m=make_method(name,[.3,0,0],dt=.2,horizon_steps=4,max_iterations=60)
            m.prepare(e);u=m(e.observe(),e)
            self.assertGreaterEqual(m.last_info['constraint_margin'],-m.config.feasibility_tolerance)
            self.assertTrue(np.all(abs(u)<=e.robot.max_wheel_speed+1e-9))
            if name=='composite_mpc':self.assertEqual(m.last_info['composite_slack'],0)

    def test_high_order_mpc_constraints_are_present(self):
        e=scene();e.add('circle',(0,.35),radius=.03)
        m=DiscreteHOCBFMPC([.3,0,0],DHOCBFConfig(horizon_steps=4,gammas=(.4,.4)))
        m.prepare(e);u=np.zeros(8)
        c=m.constraints(e.state,u,e)
        hard=m.obstacle_constraints(e.state,u,e)
        # five b0, four b1, three b2 entries for one circle.
        self.assertEqual(c.size-hard.size,12)

    def test_all_methods_full_pose_empty_scene(self):
        for name in METHODS:
            with self.subTest(method=name):
                e=scene();kw={'dt':.2}
                if name=='kinodynamic_rrt_star':kw.update(iterations=30)
                if name=='hj_reach_avoid':kw.update(grid_shape=(21,21,16),horizon_steps=60,target_position_tolerance=.08,target_heading_tolerance=.3)
                if name in ['nmpc','mpc_dhocbf','composite_mpc']:kw.update(horizon_steps=6)
                m=make_method(name,[.3,0,0],**kw)
                before=signature(e);start=e.state.copy()
                result=run_method(e,m,horizon=15,position_tolerance=.08,heading_tolerance=.3)
                self.assertEqual(result.rollout.status,'goal',result.metrics())
                self.assertLessEqual(result.final_position_error,.08)
                self.assertLessEqual(result.final_heading_error,.3)
                np.testing.assert_array_equal(e.state,start);self.assertEqual(signature(e),before)
                self.assertEqual(len(result.rollout.states),len(result.rollout.wheels)+1)

    def test_map_change_detected(self):
        e=scene();m=NonlinearMPC([.3,0,0]);m.prepare(e);e.add('circle',(1,1))
        with self.assertRaises(MethodFailure) as caught:m(e.observe(),e)
        self.assertEqual(caught.exception.status,'map_changed')

    def test_reset_repeatability(self):
        e=scene();m=NonlinearMPC([.3,0,0],NMPCConfig(dt=.2,horizon_steps=4))
        a=run_method(e,m,horizon=1);b=run_method(e,m,horizon=1)
        np.testing.assert_allclose(a.rollout.states,b.rollout.states,atol=1e-12)

    def test_no_route_is_reported(self):
        e=scene();e.add('square',(0,0),width=.3)
        # A wall spanning the planner's full box blocks grid and straight routes.
        e.add('rectangle',(0,0),width=.1,height=2.)
        m=DiscreteHOCBFMPC([.3,0,0])
        result=run_method(e,m,horizon=1)
        self.assertEqual(result.rollout.status,'planning_failure')
        self.assertEqual(len(result.rollout.commands),0)

    def test_empty_acs_reported_without_fallback(self):
        e=scene();m=HJReachAvoid([.3,0,0],HJConfig(dt=.2,horizon_steps=1,grid_shape=(15,15,12),target_position_tolerance=.08,target_heading_tolerance=.4))
        result=run_method(e,m,horizon=.2)
        self.assertEqual(result.rollout.status,'empty_admissible_set')
        self.assertEqual(len(result.rollout.commands),0)

    def test_saved_report_is_strict_json(self):
        import json
        e=scene();m=NonlinearMPC([.3,0,0],NMPCConfig(dt=.2,horizon_steps=3))
        result=run_method(e,m,horizon=.4)
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            from pathlib import Path
            text=(Path(directory)/'report.json').read_text()
            self.assertNotIn('Infinity',text);self.assertNotIn('NaN',text)
            report=json.loads(text);self.assertEqual(report['config']['seed'],0)
            self.assertIn('versions',report)


if __name__=='__main__':unittest.main()
