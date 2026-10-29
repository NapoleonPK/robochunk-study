import numpy as np

from robochunk.baselines import fixed_boundaries, local_minima_score, threshold_for_rate, velocity_boundaries
from robochunk.data_real import gripper_events
from robochunk.metrics import boundary_prf, match_boundaries, true_boundaries


def test_true_boundaries():
    assert true_boundaries([0, 0, 1, 1, 2, 0]).tolist() == [2, 4, 5]


def test_matching_one_to_one_nearest_first():
    # two predictions near one true boundary: only one may match
    assert match_boundaries([9, 11], [10], 3) == 1
    # nearest first: 12 should take 13, leaving 9 for 8
    assert match_boundaries([9, 12], [8, 13], 3) == 2
    assert match_boundaries([0, 20], [10], 3) == 0


def test_prf_ignores_step0():
    p, r, f = boundary_prf([0, 10, 30], [10, 20], 3)
    assert (p, r) == (0.5, 0.5) and abs(f - 0.5) < 1e-9
    assert boundary_prf([0], [5], 3) == (0.0, 0.0, 0.0)


def test_fixed_and_rate():
    assert np.flatnonzero(fixed_boundaries(25, 10)).tolist() == [0, 10, 20]
    rng = np.random.default_rng(0)
    speeds = [np.abs(rng.normal(size=200)) for _ in range(20)]
    b, _ = velocity_boundaries(speeds, 10)
    rate = sum(x.sum() for x in b) / 4000
    assert abs(rate - 0.1) < 0.01


def test_velocity_rule_is_causal():
    s = np.array([5, 4, 1, 4, 5, 3, 2, 6.0])
    sc = local_minima_score(s)
    # minimum at index 2 is known at index 3; minimum at 6 known at 7
    assert np.isfinite(sc).nonzero()[0].tolist() == [3, 7]


def test_gripper_hysteresis():
    g = np.array([0, 0.5, 0.7, 0.6, 0.66, 0.5, 0.3, 0.4, 0.2, 0.9])
    assert gripper_events(g).tolist() == [2, 6, 9]
