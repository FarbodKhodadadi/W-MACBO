"""Four fixed, shared maps; no controller-dependent obstacle placement."""
import numpy as np
from robot_env import Environment

START = np.array([-2., -2., 0.])
GOAL = np.array([2., 2., np.pi / 2])
BOUNDS = (-3., 3., -3., 3.)
TITLES = {
    'three_circles': 'Three unequal circles',
    'u_and_t': 'Escape from U, then avoid T',
    'central_star': 'Large central star',
    'square_passage': 'Narrow passage between squares',
}


def make_scenarios():
    scenes = {key: Environment(bounds=BOUNDS) for key in TITLES}
    for center, radius in [((-1., -1.), .25), ((0., 0.), .38), ((1., 1.), .29)]:
        scenes['three_circles'].add('circle', center, radius=radius)
    # U's opening faces southwest, away from the goal. Exit requires backing
    # away from goal in position space. A* searches both signed axis directions.
    scenes['u_and_t'].add('U', (-2., -2.), width=1.1, height=1.1,
                          thickness=.18, angle=3*np.pi/4)
    scenes['u_and_t'].add('T', (1.15, 1.15), width=1.0, height=1.05,
                          thickness=.22, angle=-np.pi/4)
    scenes['central_star'].add('star', (0., 0.), radius=.95,
                              inner_radius=.43, points=5, angle=np.pi/10)
    # Two rotated squares straddle the diagonal. Their boundary gap is .36 m:
    # twice the physical radius .14 m, plus the .04 m clearance, fit inside.
    normal = np.array([-1., 1.]) / np.sqrt(2)
    for sign in (-1, 1):
        scenes['square_passage'].add('square', sign*.63*normal,
                                    width=.9, angle=np.pi/4)
    for env in scenes.values():
        env.reset(START)
        if not env.is_free(START[:2]) or not env.is_free(GOAL[:2]):
            raise ValueError('Benchmark has an occupied endpoint')
    circles = scenes['three_circles'].obstacles
    gaps = [np.linalg.norm(a.center-b.center)-a.radius-b.radius
            for a,b in zip(circles[:-1],circles[1:])]
    assert min(gaps) > 2*scenes['three_circles'].safety_radius
    return scenes
