"""Equation checks, infeasibility detection, and independent motion validation."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from robot_env import Environment, shape, simulate
from methods import MACBO, MACBOConfig, MethodFailure, make_method, run_method
from methods.common import pose_error
from methods.geometry import signed_distances
from methods.macbo.geometry import PolygonGeometry, polygon_segment_distance
from methods.macbo.planner import _search, graph_path_cost, sparse_route


def obstacle_scene(kind, angle=0.0):
    environment = Environment(bounds=(-1, 1, -1, 1))
    environment.add(kind, radius=0.22, width=0.45, height=0.55, thickness=0.12, angle=angle)
    environment.reset((-0.8, -0.1, 0.0))
    return environment


def dense_interval_positions(environment, rollout, dt, samples=21):
    """Independent exact held-twist evaluation, including every interval endpoint."""
    if not len(rollout.wheels):
        return rollout.states[:, :2]
    right, left = rollout.wheels.T
    robot = environment.robot
    speed = robot.wheel_radius * (right + left) / 2
    yaw = robot.wheel_radius * (right - left) / robot.axle_length
    times = np.linspace(0, dt, samples)[None, :]
    travel = speed[:, None] * times * np.sinc(yaw[:, None] * times / (2 * np.pi))
    mid = rollout.states[:-1, 2, None] + yaw[:, None] * times / 2
    x = rollout.states[:-1, 0, None] + travel * np.cos(mid)
    y = rollout.states[:-1, 1, None] + travel * np.sin(mid)
    return np.stack((x, y), axis=-1).reshape(-1, 2)


class GeometryTests(unittest.TestCase):
    def test_outer_circle_contains_actual_circle(self):
        environment = obstacle_scene("circle")
        geometry = PolygonGeometry(environment, 32)
        angles = np.linspace(-np.pi, np.pi, 501)
        points = 0.22 * np.column_stack((np.cos(angles), np.sin(angles)))
        self.assertTrue(np.all(signed_distances(points, geometry.obstacles) <= 1e-12))
        self.assertTrue(all(len(group) == 1 for group in geometry.groups))

    def test_projection_gradient(self):
        geometry = PolygonGeometry(obstacle_scene("square"), 16)
        point = np.array([0.6, 0.4])
        _, barriers, gradients = geometry.component_data(point)
        eps = 1e-6
        numerical = np.column_stack([
            (geometry.component_data(point + eps * axis)[1] - geometry.component_data(point - eps * axis)[1]) / (2 * eps)
            for axis in np.eye(2)
        ])
        np.testing.assert_allclose(gradients, numerical, atol=1e-9)
        component = geometry.components[0]
        np.testing.assert_allclose(component.projection([0, 0]), [0, 0])
        self.assertGreater(barriers[0], 0)

    def test_vectorized_segment_distance_matches_core(self):
        random = np.random.default_rng(19)
        for kind in ("square", "T", "star", "U"):
            obstacle = shape(kind, (0.1, -0.2), angle=0.4)
            for _ in range(30):
                start, end = random.uniform(-1, 1, (2, 2))
                self.assertAlmostEqual(polygon_segment_distance(start, end, obstacle.vertices), obstacle.segment_distance(start, end), places=11)
            self.assertEqual(polygon_segment_distance(obstacle.vertices[0], obstacle.vertices[1], obstacle.vertices), 0.0)

    def test_active_set_distinct_primaries_and_extra_monitoring(self):
        environment = Environment()
        for center in ((0.3, 0), (2, 0), (1, 0.3), (-0.3, 0)):
            environment.add("square", center, width=0.1)
        geometry = PolygonGeometry(environment, 16)
        obstacles, components = geometry.active_set(np.array([0., 0.]), np.array([2., 0.]), 0.2)
        self.assertEqual(obstacles, [0, 1, 2, 3])
        self.assertEqual(components, [0, 1, 2, 3])

    def test_cavities_not_convexified(self):
        environment = Environment()
        environment.add("U", width=0.8, height=0.8, thickness=0.15)
        geometry = PolygonGeometry(environment, 32)
        self.assertGreater(len(geometry.components), 1)
        self.assertTrue(geometry.segment_free([0, 0], [0, 0.9], extra=0.04))


class PlannerTests(unittest.TestCase):
    def test_astar_matches_dijkstra_with_turn_penalty(self):
        environment = obstacle_scene("square")
        geometry = PolygonGeometry(environment, 16)
        config = MACBOConfig(grid_spacing=0.4, turn_delay=1.1, shorten_route=False)
        start = environment.state[:2]
        goal = np.array([0.8, 0.3])
        astar = _search(geometry, start, goal, environment.bounds, config, 0.4)
        dijkstra = _search(geometry, start, goal, environment.bounds, config, 0.4, use_heuristic=False)
        self.assertIsNotNone(astar)
        self.assertAlmostEqual(astar[1], dijkstra[1], places=12)
        self.assertAlmostEqual(astar[1], graph_path_cost(astar[0], config.nominal_speed, config.turn_delay))

    def test_route_refinement_and_safe_shortening(self):
        environment = Environment(bounds=(-1, 1, -1, 1))
        environment.add("circle", (-0.8, -0.8), radius=0.2)
        environment.reset((-0.7, 0.5, 0))
        geometry = PolygonGeometry(environment, 32)
        config = MACBOConfig(grid_spacing=3., max_refinements=2)
        route = sparse_route(geometry, environment.state[:2], np.array([0.7, 0.5]), environment.bounds, config)
        self.assertGreater(route.refinements, 0)
        self.assertTrue(all(geometry.segment_free(a, b, config.route_margin) for a, b in zip(route.waypoints[:-1], route.waypoints[1:])))

    def test_blocked_route_reports_failure(self):
        environment = Environment(bounds=(-1, 1, -1, 1))
        environment.add("rectangle", width=0.15, height=3.)
        environment.reset((-0.7, 0, 0))
        result = run_method(environment, MACBO([0.7, 0, 0], MACBOConfig(max_refinements=1)), horizon=1)
        self.assertEqual(result.rollout.status, "planning_failure")
        self.assertEqual(len(result.rollout.wheels), 0)

    def test_grid_resource_bound(self):
        environment = Environment(bounds=(-1, 1, -1, 1))
        result = run_method(environment, MACBO([0.3, 0, 0], MACBOConfig(grid_spacing=0.001, grid_max_nodes=100)), horizon=1)
        self.assertEqual(result.rollout.status, "resource_limit")


class ControllerTests(unittest.TestCase):
    def test_config_validation(self):
        for kwargs in ({"c_bar": .04}, {"q": 1}, {"clf_mode": "unknown"}, {"mu_delta": 0},
                       {"grid_spacing": 0}, {"refinement_factor": 1}, {"initial_memory": -1}, {"circle_sides": 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MACBOConfig(**kwargs)
        self.assertIsInstance(make_method("macbo", [0.2, 0, 0]), MACBO)

    def test_strict_sideways_infeasibility_is_not_hidden(self):
        environment = Environment(bounds=(-0.5, 0.5, -0.5, 0.5))
        controller = MACBO([0, 0.3, 0], MACBOConfig(clf_mode="strict"))
        result = run_method(environment, controller, horizon=10)
        self.assertEqual(result.rollout.status, "qp_infeasible")
        self.assertEqual(result.rollout.times[-1], 0)
        self.assertEqual(len(result.rollout.wheels), 0)

    def test_strict_qp_objective_and_lie_derivative(self):
        environment = Environment(bounds=(-0.5, 0.5, -0.5, 0.5))
        controller = MACBO([0.3, 0, 0], MACBOConfig(clf_mode="strict"))
        controller.prepare(environment)
        state = environment.state.copy()
        field = controller.field(state, environment)
        program = controller.quadratic_program(state, environment, field, "strict", None)
        normalized = np.array([0.3, 0.2])
        slack = 0.0001
        controls = normalized * environment.robot.max_wheel_speed
        derivative = float(pose_error(state, controller.goal) @ environment.robot.g(state) @ controls)
        self.assertAlmostEqual(program.dV @ normalized, derivative)
        cfg = controller.config
        progress = (1-cfg.q) * (program.V+cfg.epsilon_V)**(-cfg.q)
        def original(z):
            u = z[:2] * environment.robot.max_wheel_speed
            velocity = environment.robot.g(state)[:2] @ u
            return (cfg.mu_v/2 * np.sum((velocity-field.desired)**2) + cfg.mu_t*progress*(program.dV @ z[:2])
                    + cfg.mu_u/2 * (u @ u) + cfg.mu_delta_u/2 * np.sum((u-controller.previous)**2) + cfg.mu_delta/2*z[2]**2)
        z = np.r_[normalized, slack]
        reconstructed = lambda value: .5*value @ program.hessian @ value + program.linear @ value
        self.assertAlmostEqual(original(z)-original(np.zeros(3)), reconstructed(z)-reconstructed(np.zeros(3)), places=11)

    def test_memory_field_matches_equations_and_nonnegative_update(self):
        environment = obstacle_scene("circle")
        controller = MACBO([0.8, 0.15, 0.7], MACBOConfig(initial_memory=0.2))
        controller.prepare(environment)
        state = environment.state.copy()
        field = controller.field(state, environment)
        index = field.nearest_component
        projections, barriers, gradients = controller.geometry.component_data(state[:2])
        cfg = controller.config
        distance = np.linalg.norm(controller.waypoints[controller.waypoint_index]-state[:2])
        rho = -np.expm1(-distance/cfg.ell)
        s = np.exp(-cfg.beta*max(0, barriers[index]))
        e = max(0, -field.normal @ field.waypoint_attraction)
        self.assertAlmostEqual(field.memory_forcing, rho*s*e)
        np.testing.assert_allclose(field.circulation, cfg.k_c*controller.memory*rho*s*field.tangent)
        before = controller.memory
        command = controller(environment.observe(), environment)
        expected = before*np.exp(-cfg.lambda_memory*cfg.dt) + field.memory_forcing*(-np.expm1(-cfg.lambda_memory*cfg.dt))/cfg.lambda_memory
        self.assertAlmostEqual(controller.memory, expected)
        self.assertGreaterEqual(controller.memory, 0)

    def test_no_obstacle_memory_decay_and_sigma_tie(self):
        environment = Environment(bounds=(-1, 1, -1, 1))
        controller = MACBO([0.4, 0, 0], MACBOConfig(initial_memory=0.3, initial_sigma=-1))
        controller.prepare(environment)
        field = controller.field(environment.state, environment)
        np.testing.assert_array_equal(field.circulation, [0, 0])
        controller(environment.observe(), environment)
        self.assertAlmostEqual(controller.memory, 0.3*np.exp(-controller.config.lambda_memory*controller.config.dt))
        self.assertEqual(controller.sigma, -1)

    def test_actual_qp_feasibility_and_rate_bound(self):
        environment = obstacle_scene("circle")
        controller = MACBO([0.8, 0.15, 0.7])
        result = run_method(environment, controller, horizon=30)
        self.assertEqual(result.rollout.status, "goal", result.metrics())
        for info in result.diagnostics:
            if info.get("stage") != "control":
                continue
            self.assertGreaterEqual(info["constraint_margin"], -controller.config.feasibility_tolerance)
            self.assertLessEqual(info["clf_slack"], controller.config.c_bar*info["V"]**controller.config.q + 1e-7)
            self.assertLessEqual(info["clf_derivative"], info["required_clf_derivative"] + 1e-7)
            self.assertGreaterEqual(info["memory_after"], 0)
            self.assertGreaterEqual(info["minimum_barrier_residual"], -1e-7)
            self.assertGreaterEqual(info["polygon_interval_margin"], -1e-12)

    def test_reset_reproduces_exact_trajectory(self):
        environment = obstacle_scene("U")
        controller = MACBO([0.8, 0.15, 0.7])
        first = run_method(environment, controller, horizon=30)
        second = run_method(environment, controller, horizon=30)
        np.testing.assert_array_equal(first.rollout.states, second.rollout.states)
        np.testing.assert_array_equal(first.rollout.wheels, second.rollout.wheels)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_original_simulate_interface(self):
        environment = Environment(bounds=(-0.5, 0.5, -0.5, 0.5))
        controller = MACBO([0.3, 0, 0])
        result = simulate(environment, controller, mode="wheels", dt=controller.config.dt,
                          horizon=10, goal=controller.goal, tolerance=controller.config.target_position_tolerance)
        self.assertEqual(result.status, "goal")

    def test_repeated_unexecuted_control_call_is_rejected(self):
        environment = Environment(bounds=(-1, 1, -1, 1))
        controller = MACBO([0.4, 0, 0])
        controller(environment.observe(), environment)
        with self.assertRaises(MethodFailure) as caught:
            controller(environment.observe(), environment)
        self.assertEqual(caught.exception.status, "execution_mismatch")

    def test_saved_scene_and_configuration_replay(self):
        environment = obstacle_scene("star")
        controller = MACBO([0.8, 0.15, 0.7])
        result = run_method(environment, controller, horizon=30)
        with tempfile.TemporaryDirectory() as directory:
            result.save(directory)
            report = json.loads((Path(directory)/"report.json").read_text())
            restored = Environment.load(Path(directory)/"scene.json")
            replay = run_method(restored, make_method("macbo", report["goal"], **report["config"]), horizon=30)
            np.testing.assert_array_equal(result.rollout.states, replay.rollout.states)


class NavigationRegressionTests(unittest.TestCase):
    def assert_navigation(self, environment, goal, config=None, horizon=60):
        controller = MACBO(goal, config)
        result = run_method(environment, controller, horizon=horizon)
        self.assertEqual(result.rollout.status, "goal", result.metrics())
        self.assertLessEqual(result.final_position_error, controller.config.target_position_tolerance)
        self.assertLessEqual(result.final_heading_error, controller.config.target_heading_tolerance)
        self.assertTrue(all(attempt["accepted"] and not attempt["saturated"] for attempt in result.rollout.attempts))
        self.assertLessEqual(np.max(np.abs(result.rollout.wheels)), environment.robot.max_wheel_speed + 1e-9)
        points = dense_interval_positions(environment, result.rollout, controller.config.dt)
        margins = signed_distances(points, environment.obstacles) - environment.safety_radius
        self.assertTrue(np.all(margins >= -1e-10))
        return result

    def test_all_five_obstacles_and_rotations(self):
        for kind, angle in (("circle", 0), ("square", 0.5), ("T", 0.4), ("star", 0.3), ("U", -0.4)):
            with self.subTest(shape=kind):
                self.assert_navigation(obstacle_scene(kind, angle), [0.8, 0.15, 0.7])

    def test_mixed_map_and_bad_initial_heading(self):
        environment = Environment(bounds=(-1, 1, -1, 1))
        for kind, center in (("circle", (-0.4, 0.4)), ("square", (0.4, 0.4)),
                             ("T", (-0.4, -0.4)), ("star", (0.4, -0.4)), ("U", (0, 0))):
            environment.add(kind, center, radius=0.16, width=0.3, height=0.35, thickness=0.08)
        environment.reset((-0.85, -0.8, 2))
        self.assert_navigation(environment, [0.85, 0.8, -1])

    def test_u_cavity_exit(self):
        environment = Environment(bounds=(-1.2, 1.2, -1.2, 1.2))
        environment.add("U", width=0.8, height=0.8, thickness=0.15)
        environment.reset((0, 0, np.pi))
        self.assert_navigation(environment, [0, 0.9, 0.3])

    def test_sideways_reversed_and_heading_only_goals(self):
        for initial, goal in (([0, 0, 0], [0, 0.3, 0]),
                              ([0, 0, np.pi], [0.3, 0, 0]),
                              ([0, 0, 0], [0, 0, 1.2])):
            with self.subTest(goal=goal):
                environment = Environment(bounds=(-1, 1, -1, 1))
                environment.reset(initial)
                self.assert_navigation(environment, goal)

    def test_deterministic_varied_obstacle_trials(self):
        random = np.random.default_rng(20261001)
        for index, kind in enumerate(("T", "U", "star", "square", "circle", "U")):
            with self.subTest(index=index, shape=kind):
                environment = Environment(bounds=(-1, 1, -1, 1))
                environment.add(kind, random.uniform(-0.1, 0.1, 2), radius=0.18, width=0.38,
                                height=0.45, thickness=0.1, angle=random.uniform(-1, 1))
                environment.reset((-0.8, random.uniform(-0.35, -0.15), random.uniform(-np.pi, np.pi)))
                goal = [0.8, random.uniform(0.1, 0.35), random.uniform(-np.pi, np.pi)]
                self.assert_navigation(environment, goal)


if __name__ == "__main__":
    unittest.main()
