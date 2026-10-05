"""Memory-augmented circulation fields and constrained differential-drive control."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from ..common import Method, MethodFailure, interval_safe, pose_error, wrap
from .config import MACBOConfig
from .geometry import PolygonGeometry
from .planner import sparse_route


@dataclass(frozen=True)
class Field:
    desired: np.ndarray
    waypoint_attraction: np.ndarray
    goal_attraction: np.ndarray
    circulation: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    memory_forcing: float
    visibility: bool
    nearest_component: int | None
    active_obstacles: list
    active_components: list


@dataclass(frozen=True)
class QP:
    """Decision [normalized right wheel, normalized left wheel, CLF slack]."""
    hessian: np.ndarray
    linear: np.ndarray
    matrix: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    V: float
    dV: np.ndarray
    barrier_values: np.ndarray
    barrier_derivatives: np.ndarray
    phase: str
    heading_target: float | None


class MACBO(Method):
    """Callable wheel-input MACBO, with strict and explicit hybrid CLF modes.

    Strict mode follows the attached optimization's intermediate-position and
    final-full-pose CLFs. Hybrid mode adds stationary turning CLFs and a yaw
    tracking objective to make waypoint execution usable on this nonholonomic
    robot. Both preserve hard convex-component barriers and bounded CLF slack.
    No infeasible solve switches to another controller or unbounded relaxation.
    """
    name = "macbo"

    def __init__(self, goal, config=None):
        super().__init__(goal, config or MACBOConfig())

    def reset(self):
        super().reset()
        self.memory = self.config.initial_memory
        self.sigma = self.config.initial_sigma
        self.previous = np.zeros(2)
        self.waypoint_index = 1
        self.phase = "turn"
        self.route = None
        self.stage_entries = []
        self._last_stage = None
        self._expected_time = None

    def prepare(self, environment):
        super().prepare(environment)
        self.geometry = PolygonGeometry(environment, self.config.circle_sides)
        self.route = sparse_route(self.geometry, environment.state[:2], self.goal[:2],
                                  environment.bounds, self.config)
        self._expected_time = environment.time
        self.record(stage="planning", clf_mode=self.config.clf_mode,
                    graph_cost=self.route.graph_cost, spacing=self.route.spacing,
                    refinements=self.route.refinements, expansions=self.route.expansions,
                    waypoint_count=len(self.route.waypoints),
                    convex_components=len(self.geometry.components))

    @property
    def waypoints(self):
        if self.route is None:
            raise RuntimeError("Prepare MACBO before reading its waypoints")
        return self.route.waypoints.copy()

    def _switch_waypoints(self, position):
        last = len(self.route.waypoints) - 1
        while self.waypoint_index < last:
            target = self.route.waypoints[self.waypoint_index]
            if np.linalg.norm(position - target) > self.config.epsilon_w:
                break
            self.waypoint_index += 1
            self.phase = "turn"

    def field(self, state, environment):
        """Compute the attached field at the current memory without advancing it."""
        config = self.config
        position = state[:2]
        waypoint = self.route.waypoints[self.waypoint_index]
        waypoint_attraction = config.k_w * (waypoint - position)
        visible = self.geometry.segment_free(position, self.goal[:2])
        goal_attraction = config.k_g * (self.goal[:2] - position) if visible else np.zeros(2)
        # Sampled implementation activates early by at most one-step translation.
        monitoring = config.epsilon_a
        if config.clf_mode == "hybrid":
            monitoring += environment.robot.wheel_radius * environment.robot.max_wheel_speed * config.dt
        obstacles, components = self.geometry.active_set(position, waypoint, monitoring)
        normal, tangent, circulation = np.zeros(2), np.zeros(2), np.zeros(2)
        forcing = 0.0
        nearest = None
        if components:
            projections, barriers, gradients = self.geometry.component_data(position)
            distances = np.linalg.norm(position - projections, axis=1)
            nearest = min(components, key=lambda i: (distances[i], i))
            if distances[nearest] <= 1e-12:
                raise MethodFailure("invalid_state", "Robot center is inside an active outer-polygon component")
            normal = (position - projections[nearest]) / distances[nearest]
            counterclockwise = np.array([-normal[1], normal[0]])
            preference = float(counterclockwise @ (waypoint - position))
            if abs(preference) > 1e-12:
                self.sigma = 1 if preference > 0 else -1
            tangent = self.sigma * counterclockwise
            localization = np.exp(-config.beta * max(0.0, barriers[nearest]))
            conflict = max(0.0, -float(normal @ waypoint_attraction))
            attenuation = -np.expm1(-np.linalg.norm(waypoint - position) / config.ell)
            forcing = float(attenuation * localization * conflict)
            circulation = config.k_c * self.memory * attenuation * localization * tangent
        return Field(waypoint_attraction + goal_attraction + circulation,
                     waypoint_attraction, goal_attraction, circulation,
                     normal, tangent, forcing, visible, nearest, obstacles, components)

    def _select_phase(self, state, desired_field):
        config = self.config
        if config.clf_mode == "strict":
            self.phase = "strict"
            return self.phase, None
        waypoint = self.route.waypoints[self.waypoint_index]
        final = self.waypoint_index == len(self.route.waypoints) - 1
        if final and np.linalg.norm(state[:2] - self.goal[:2]) <= config.target_position_tolerance:
            self.phase = "terminal_turn"
            return self.phase, self.goal[2]
        bearing = np.arctan2(waypoint[1] - state[1], waypoint[0] - state[0])
        error = abs(wrap(bearing - state[2]))
        if self.phase != "drive":
            self.phase = "drive" if error <= config.heading_enter else "turn"
        elif error > config.heading_exit:
            self.phase = "turn"
        if self.phase == "turn":
            return self.phase, bearing
        if np.linalg.norm(desired_field) < 1e-12:
            return self.phase, bearing
        desired_bearing = np.arctan2(desired_field[1], desired_field[0])
        offset = np.clip(wrap(desired_bearing - bearing),
                         -config.max_field_heading_offset, config.max_field_heading_offset)
        return self.phase, bearing + offset

    def quadratic_program(self, state, environment, field, phase, heading_target):
        """Assemble the actual constrained objective and exact Lie derivatives."""
        config = self.config
        robot = environment.robot
        G = robot.g(state) * robot.max_wheel_speed
        Gp = G[:2]
        waypoint = self.route.waypoints[self.waypoint_index]
        if phase in ("turn", "terminal_turn"):
            error = wrap(state[2] - heading_target)
            V = float(0.5 * error**2)
            dV = error * G[2]
        elif phase == "strict" and self.waypoint_index == len(self.route.waypoints) - 1:
            error = pose_error(state, self.goal)
            V = float(0.5 * error @ error)
            dV = error @ G
        else:
            error = state[:2] - waypoint
            V = float(0.5 * error @ error)
            dV = error @ Gp
        progress = (1 - config.q) * (V + config.epsilon_V)**(-config.q)
        wheel_scale2 = robot.max_wheel_speed**2
        Q = np.zeros((3, 3))
        Q[:2, :2] = config.mu_v * Gp.T @ Gp + (config.mu_u + config.mu_delta_u) * wheel_scale2 * np.eye(2)
        Q[2, 2] = config.mu_delta
        linear = np.r_[-config.mu_v * Gp.T @ field.desired +
                       config.mu_t * progress * dV -
                       config.mu_delta_u * robot.max_wheel_speed * self.previous, 0.0]
        if phase != "strict":
            desired_yaw = np.clip(-config.heading_gain * wrap(state[2] - heading_target),
                                  -config.max_turn_speed, config.max_turn_speed)
            Q[:2, :2] += config.mu_heading * np.outer(G[2], G[2])
            linear[:2] -= config.mu_heading * desired_yaw * G[2]
        _, all_barriers, gradients = self.geometry.component_data(state[:2])
        selected = field.active_components
        barriers = all_barriers[selected]
        derivatives = gradients[selected] @ Gp
        rows = [np.r_[row, 0.0] for row in derivatives]
        lower = (-config.alpha * barriers).tolist()
        upper = [np.inf] * len(rows)
        rows.append(np.r_[dV, -1.0])
        lower.append(-np.inf)
        upper.append(-config.c * V**config.q)
        if phase in ("turn", "terminal_turn"):
            # Stationary turn: physical center and disk remain fixed.
            rows.append(np.r_[Gp[0] if abs(np.cos(state[2])) >= abs(np.sin(state[2])) else Gp[1], 0.0])
            lower.append(0.0)
            upper.append(0.0)
        return QP(Q, linear, np.array(rows), np.array(lower), np.array(upper),
                  np.array([-1.0, -1.0, 0.0]),
                  np.array([1.0, 1.0, config.c_bar * V**config.q]),
                  V, dV, barriers, derivatives, phase, heading_target)

    def _solve(self, program):
        config = self.config
        equality = np.isfinite(program.lower) & np.isfinite(program.upper) & (program.lower == program.upper)
        constraints = []
        for selected in (equality, ~equality):
            if np.any(selected):
                constraints.append(LinearConstraint(program.matrix[selected], program.lower[selected], program.upper[selected]))
        initial = np.r_[self.previous / self._wheel_limit, program.bounds_upper[2]]
        result = minimize(lambda z: 0.5 * z @ program.hessian @ z + program.linear @ z,
                          initial, jac=lambda z: program.hessian @ z + program.linear,
                          method="SLSQP", constraints=constraints,
                          bounds=Bounds(program.bounds_lower, program.bounds_upper),
                          options={"maxiter": config.max_iterations, "ftol": config.solver_tolerance})
        residuals = np.r_[program.matrix @ result.x - program.lower,
                          program.upper - program.matrix @ result.x,
                          result.x - program.bounds_lower, program.bounds_upper - result.x]
        margin = float(np.min(residuals))
        if not result.success or not np.all(np.isfinite(result.x)) or margin < -config.feasibility_tolerance:
            finite_upper = np.isfinite(program.upper) & ~equality
            finite_lower = np.isfinite(program.lower) & ~equality
            A = np.vstack((program.matrix[finite_upper], -program.matrix[finite_lower]))
            b = np.r_[program.upper[finite_upper], -program.lower[finite_lower]]
            feasible = linprog(np.zeros(3), A_ub=A if len(A) else None, b_ub=b if len(b) else None,
                               A_eq=program.matrix[equality] if equality.any() else None,
                               b_eq=program.upper[equality] if equality.any() else None,
                               bounds=list(zip(program.bounds_lower, program.bounds_upper)), method="highs")
            status = "qp_infeasible" if feasible.status == 2 else "solver_failure"
            return None, {"solver_success": False, "status": status,
                          "solver_message": str(result.message), "constraint_margin": margin,
                          "feasibility_message": str(feasible.message), "iterations": int(result.nit)}
        return result.x, {"solver_success": True, "solver_message": str(result.message),
                          "constraint_margin": margin, "iterations": int(result.nit)}

    def control(self, observation, environment):
        config = self.config
        state = observation["state"]
        if not np.isclose(observation["time"], self._expected_time, atol=1e-9, rtol=0):
            raise MethodFailure("execution_mismatch", "MACBO requires one call per accepted config.dt interval; reset after rejected/external steps")
        error = pose_error(state, self.goal)
        if np.linalg.norm(error[:2]) <= config.target_position_tolerance and abs(error[2]) <= config.target_heading_tolerance:
            self.memory *= np.exp(-config.lambda_memory * config.dt)
            self.previous = np.zeros(2)
            self._expected_time = observation["time"] + config.dt
            self.record(stage="complete", memory=self.memory)
            return np.zeros(2)
        self._switch_waypoints(state[:2])
        field = self.field(state, environment)
        phase, heading = self._select_phase(state, field.desired)
        program = self.quadratic_program(state, environment, field, phase, heading)
        self._wheel_limit = environment.robot.max_wheel_speed
        decision, info = self._solve(program)
        info.update(stage="control", clf_mode=config.clf_mode, phase=phase,
                    waypoint_index=self.waypoint_index, V=program.V,
                    active_obstacles=field.active_obstacles,
                    active_components=field.active_components,
                    memory_before=self.memory, memory_forcing=field.memory_forcing,
                    sigma=self.sigma, visibility=field.visibility,
                    desired_field=field.desired.tolist(), circulation=field.circulation.tolist())
        if decision is None:
            self.record(**info)
            raise MethodFailure(info["status"], "MACBO bounded-relaxation QP: " + info["solver_message"])
        wheels = decision[:2] * environment.robot.max_wheel_speed
        polygon_margin = self.geometry.interval_margin(environment, state, wheels, config.dt)
        if polygon_margin < -1e-12 or not interval_safe(environment, state, wheels, config.dt):
            info.update(polygon_interval_margin=polygon_margin, status="unsafe_command")
            self.record(**info)
            raise MethodFailure("unsafe_command", "MACBO candidate fails full held-input interval clearance; reduce dt or revise parameters")
        decay = np.exp(-config.lambda_memory * config.dt)
        next_memory = self.memory * decay + field.memory_forcing * (-np.expm1(-config.lambda_memory * config.dt)) / config.lambda_memory
        clf_derivative = float(program.dV @ decision[:2])
        info.update(memory_after=float(next_memory), clf_slack=float(decision[2]),
                    clf_derivative=clf_derivative,
                    required_clf_derivative=-(config.c - config.c_bar) * program.V**config.q,
                    minimum_barrier_residual=float(np.min(program.barrier_derivatives @ decision[:2] + config.alpha * program.barrier_values)) if len(program.barrier_values) else None,
                    polygon_interval_margin=None if not np.isfinite(polygon_margin) else float(polygon_margin),
                    wheels=wheels.tolist())
        stage = (self.waypoint_index, phase)
        if stage != self._last_stage:
            self.stage_entries.append({"time": observation["time"], "waypoint_index": self.waypoint_index,
                                       "phase": phase, "entry_V": program.V})
            self._last_stage = stage
        self.record(**info)
        self.memory = float(next_memory)
        self.previous = wheels.copy()
        self._expected_time = observation["time"] + config.dt
        return wheels
