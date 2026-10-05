"""Outer polygons, unique convex projections, and manuscript active-set rules."""
from dataclasses import dataclass
import copy
import numpy as np
from robot_env.core import Environment, Obstacle, vector
from ..geometry import triangulate, signed_distances


@dataclass(frozen=True)
class Component:
    obstacle_index: int
    local_index: int
    vertices: np.ndarray

    def projection(self, position):
        """Euclidean projection on a closed convex counterclockwise polygon."""
        p = vector(position, 2)
        a = self.vertices
        directions = np.roll(a, -1, axis=0) - a
        offset = p - a
        crosses = directions[:, 0] * offset[:, 1] - directions[:, 1] * offset[:, 0]
        if np.all(crosses >= -1e-14):
            return p.copy()
        fraction = np.clip(np.sum(offset * directions, axis=1) /
                           np.sum(directions * directions, axis=1), 0, 1)
        candidates = a + fraction[:, None] * directions
        distances2 = np.sum((candidates - p)**2, axis=1)
        return candidates[int(np.argmin(distances2))].copy()


def is_convex(vertices):
    edge = np.roll(vertices, -1, axis=0) - vertices
    following = np.roll(edge, -1, axis=0)
    cross = edge[:, 0] * following[:, 1] - edge[:, 1] * following[:, 0]
    return bool(np.all(cross >= -1e-13))


class PolygonGeometry:
    """Preserve polygons; replace circles by regular circumscribed polygons."""
    def __init__(self, environment, circle_sides):
        self.radius = environment.safety_radius
        self.obstacles = []
        self.components = []
        self.groups = []
        for i, obstacle in enumerate(environment.obstacles):
            if obstacle.radius is not None:
                angles = 2 * np.pi * np.arange(circle_sides) / circle_sides
                outer_radius = obstacle.radius / np.cos(np.pi / circle_sides)
                vertices = obstacle.center + outer_radius * np.column_stack((np.cos(angles), np.sin(angles)))
                polygon = Obstacle(obstacle.name, obstacle.center.copy(), vertices=vertices)
            else:
                polygon = copy.deepcopy(obstacle)
            self.obstacles.append(polygon)
            v = polygon.vertices
            area = np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1))
            if area < 0:
                v = v[::-1].copy()
                polygon.vertices = v
            pieces = [v] if is_convex(v) else triangulate(v)
            group = []
            for local_index, vertices in enumerate(pieces):
                group.append(len(self.components))
                self.components.append(Component(i, local_index, vertices.copy()))
            self.groups.append(group)
        self.planning_environment = Environment(
            robot=environment.robot, obstacles=self.obstacles,
            clearance=environment.clearance, bounds=environment.bounds,
            state=environment.state.copy(),
        )

    def obstacle_distances(self, position):
        if not self.obstacles:
            return np.empty(0)
        return np.maximum(0.0, signed_distances(np.asarray(position)[None, :], self.obstacles)[0])

    def component_data(self, position):
        if not self.components:
            return np.empty((0, 2)), np.empty(0), np.empty((0, 2))
        projections = np.array([c.projection(position) for c in self.components])
        displacement = np.asarray(position) - projections
        distances = np.linalg.norm(displacement, axis=1)
        return projections, distances**2 - self.radius**2, 2 * displacement

    def active_set(self, position, waypoint, epsilon_a):
        """Ordered distinct robot/waypoint/segment primaries, then monitoring."""
        count = len(self.obstacles)
        if count == 0:
            return [], []
        robot_distances = self.obstacle_distances(position)
        waypoint_distances = self.obstacle_distances(waypoint)
        segment_distances = np.array([polygon_segment_distance(position, waypoint, o.vertices) for o in self.obstacles])
        remaining = list(range(count))
        primary = []
        for distances in (robot_distances, waypoint_distances, segment_distances):
            if not remaining:
                break
            index = min(remaining, key=lambda i: (distances[i], i))
            primary.append(index)
            remaining.remove(index)
        monitored = np.flatnonzero(robot_distances <= self.radius + epsilon_a).tolist()
        active_obstacles = sorted(set(primary + monitored))
        projections, barriers, gradients = self.component_data(position)
        distances = np.linalg.norm(np.asarray(position) - projections, axis=1)
        active_components = []
        for i in active_obstacles:
            group = self.groups[i]
            nearest = min(group, key=lambda j: (distances[j], self.components[j].local_index))
            active_components.append(nearest)
            active_components.extend(j for j in group if distances[j] <= self.radius + epsilon_a)
        return active_obstacles, sorted(set(active_components))

    def segment_free(self, start, end, extra=0.0):
        return all(polygon_segment_distance(start, end, o.vertices) >= self.radius + extra
                   for o in self.obstacles)

    def interval_margin(self, environment, state, wheels, dt):
        """Conservative lower bound for clearance over the entire held arc."""
        if not self.obstacles:
            return float("inf")
        candidate = environment.robot.advance(state, wheels, dt)
        speed, turn_rate = environment.robot.wheels_to_twist(wheels)
        deviation = abs(speed * turn_rate) * dt**2 / 8
        return min(polygon_segment_distance(state[:2], candidate[:2], o.vertices) for o in self.obstacles) - self.radius - deviation


def polygon_segment_distance(start, end, vertices):
    """Exact occupied-polygon/segment distance with vectorized edge operations."""
    p, q = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    a = np.asarray(vertices, dtype=float)
    b = np.roll(a, -1, axis=0)
    edges = b - a
    edge_lengths2 = np.sum(edges**2, axis=1)

    def inside(point):
        crosses_y = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
        ratio = np.divide(point[1] - a[:, 1], edges[:, 1],
                          out=np.zeros(len(a)), where=edges[:, 1] != 0)
        return bool(np.count_nonzero(crosses_y & (point[0] < a[:, 0] + ratio * edges[:, 0])) % 2)

    if inside(p) or inside(q):
        return 0.0
    direction = q - p
    length2 = float(direction @ direction)
    offset = a - p
    denominators = direction[0] * edges[:, 1] - direction[1] * edges[:, 0]
    nonparallel = np.abs(denominators) > 1e-14
    t = np.divide(offset[:, 0] * edges[:, 1] - offset[:, 1] * edges[:, 0],
                  denominators, out=np.full(len(a), np.inf), where=nonparallel)
    s = np.divide(offset[:, 0] * direction[1] - offset[:, 1] * direction[0],
                  denominators, out=np.full(len(a), np.inf), where=nonparallel)
    if np.any(nonparallel & (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)):
        return 0.0
    distances2 = []
    for point in (p, q):
        fraction = np.clip(np.sum((point - a) * edges, axis=1) / edge_lengths2, 0, 1)
        distances2.append(np.min(np.sum((point - a - fraction[:, None] * edges)**2, axis=1)))
    if length2 > 0:
        for endpoints in (a, b):
            fraction = np.clip((endpoints - p) @ direction / length2, 0, 1)
            distances2.append(np.min(np.sum((endpoints - p - fraction[:, None] * direction)**2, axis=1)))
    return float(np.sqrt(min(distances2)))
