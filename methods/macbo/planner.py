"""Sparse axis-aligned A* with incoming directions and the manuscript cost."""
from dataclasses import dataclass
import heapq
import itertools
import numpy as np
from ..common import MethodFailure
from ..geometry import signed_distances


@dataclass(frozen=True)
class Route:
    waypoints: np.ndarray
    graph_path: np.ndarray
    graph_cost: float
    spacing: float
    refinements: int
    expansions: int


def direction_of(displacement):
    length = np.linalg.norm(displacement)
    if length < 1e-12:
        return None
    return tuple(np.round(displacement / length, 12))


def graph_path_cost(path, nominal_speed, turn_delay):
    """Travel L1 time plus changes of oriented edge direction."""
    cost = 0.0
    previous = None
    for a, b in zip(path[:-1], path[1:]):
        direction = direction_of(b - a)
        cost += np.linalg.norm(b - a, ord=1) / nominal_speed
        if direction is not None:
            if previous is not None and previous != direction:
                cost += turn_delay
            previous = direction
    return float(cost)


def _search(geometry, start, goal, bounds, config, spacing, use_heuristic=True):
    xmin, xmax, ymin, ymax = bounds
    xs = np.arange(xmin, xmax + 1e-12, spacing)
    ys = np.arange(ymin, ymax + 1e-12, spacing)
    if len(xs) * len(ys) > config.grid_max_nodes:
        raise MethodFailure("resource_limit", "MACBO grid exceeds grid_max_nodes")
    coordinates = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    distances = signed_distances(coordinates, geometry.obstacles)
    free = np.all(distances >= geometry.radius + config.route_margin, axis=1)
    indices = np.stack(np.meshgrid(np.arange(len(xs)), np.arange(len(ys)), indexing="ij"), axis=-1).reshape(-1, 2)
    nodes = {tuple(ij): p for ij, p in zip(indices[free], coordinates[free])}
    nodes["start"], nodes["goal"] = start, goal
    grid_keys = [key for key in nodes if isinstance(key, tuple)]
    visible_start = [key for key in grid_keys if geometry.segment_free(start, nodes[key], config.route_margin)]
    visible_goal = {key for key in grid_keys if geometry.segment_free(nodes[key], goal, config.route_margin)}
    if not visible_start or not visible_goal:
        return None
    edge_cache = {}

    def neighbors(node):
        if node == "start":
            return visible_start
        if node == "goal":
            return []
        i, j = node
        candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for k in range(1, config.max_edge_steps + 1):
                neighbor = (i + k * dx, j + k * dy)
                if neighbor not in nodes:
                    continue
                pair = tuple(sorted((node, neighbor)))
                if pair not in edge_cache:
                    edge_cache[pair] = geometry.segment_free(nodes[node], nodes[neighbor], config.route_margin)
                if edge_cache[pair]:
                    candidates.append(neighbor)
        if node in visible_goal:
            candidates.append("goal")
        return candidates

    sequence = itertools.count()
    initial = ("start", None)
    costs = {initial: 0.0}
    parents = {}
    initial_heuristic = np.linalg.norm(start - goal, ord=1) / config.nominal_speed if use_heuristic else 0.0
    queue = [(initial_heuristic, next(sequence), initial)]
    closed = set()
    while queue:
        _, _, state = heapq.heappop(queue)
        if state in closed:
            continue
        node, incoming = state
        if node == "goal":
            path = [goal]
            cursor = state
            while cursor != initial:
                cursor = parents[cursor]
                path.append(nodes[cursor[0]])
            path = np.array(path[::-1])
            return path, costs[state], len(closed)
        closed.add(state)
        if len(closed) > config.max_expansions:
            raise MethodFailure("resource_limit", "MACBO A* exceeded max_expansions")
        for neighbor in neighbors(node):
            displacement = nodes[neighbor] - nodes[node]
            outgoing = direction_of(displacement)
            next_direction = incoming if outgoing is None else outgoing
            turn = config.turn_delay if incoming is not None and outgoing is not None and outgoing != incoming else 0.0
            tentative = costs[state] + np.linalg.norm(displacement, ord=1) / config.nominal_speed + turn
            next_state = (neighbor, next_direction)
            if tentative < costs.get(next_state, float("inf")):
                costs[next_state] = tentative
                parents[next_state] = state
                heuristic = np.linalg.norm(nodes[neighbor] - goal, ord=1) / config.nominal_speed if use_heuristic else 0.0
                heapq.heappush(queue, (tentative + heuristic, next(sequence), next_state))
    return None


def sparse_route(geometry, start, goal, bounds, config):
    """Refine only after graph failure; shortening requires full segment clearance."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    xmin, xmax, ymin, ymax = bounds
    for point in (start, goal):
        if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
            raise MethodFailure("invalid_endpoint", "MACBO endpoint outside planning bounds")
        if not geometry.segment_free(point, point):
            raise MethodFailure("invalid_endpoint", "Endpoint violates outer-polygon disk clearance")
        if not geometry.segment_free(point, point, config.route_margin):
            raise MethodFailure("planning_failure", "Endpoint has less than route_margin; reduce the explicit planning margin")
    if np.linalg.norm(start - goal) < 1e-12:
        return Route(np.array([start, goal]), np.array([start, goal]), 0.0, config.grid_spacing, 0, 0)
    for refinement in range(config.max_refinements + 1):
        spacing = config.grid_spacing * config.refinement_factor**refinement
        result = _search(geometry, start, goal, bounds, config, spacing)
        if result is None:
            continue
        path, cost, expansions = result
        waypoints = [path[0]]
        index = 0
        while index < len(path) - 1:
            candidate = len(path) - 1 if config.shorten_route else index + 1
            while candidate > index + 1 and not geometry.segment_free(path[index], path[candidate], config.route_margin):
                candidate -= 1
            if np.linalg.norm(path[candidate] - waypoints[-1]) > 1e-12:
                waypoints.append(path[candidate])
            index = candidate
        if len(waypoints) == 1:
            waypoints.append(goal)
        return Route(np.array(waypoints), path, cost, spacing, refinement, expansions)
    raise MethodFailure("planning_failure", "No sparse-grid route after all configured refinements")
