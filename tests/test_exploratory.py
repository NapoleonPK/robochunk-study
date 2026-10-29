import numpy as np

from robochunk.exploratory import chance_f1, merge_bursts, phase_density


def test_merge_bursts_keeps_first():
    assert merge_bursts([0, 1, 5, 6, 7, 10, 12, 13]).tolist() == [0, 5, 10, 13]


def test_phase_density():
    ph = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    d = phase_density([np.array([0, 3, 4, 6])], [ph])
    assert d["approach"] == 0.0 and d["descend"] == 1.0 and abs(d["close"] - 1 / 3) < 1e-12


def test_chance_keeps_count_and_is_deterministic():
    ph = np.repeat(np.arange(7), 10)
    b = [np.array([0, 10, 20, 30, 40, 50, 60])]
    a = chance_f1(b, [ph], 3, n=50, seed=1)
    assert a == chance_f1(b, [ph], 3, n=50, seed=1) and 0 < a[0] < 1
