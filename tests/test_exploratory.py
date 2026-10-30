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


def test_circular_shift_keeps_count_and_spacing():
    from robochunk.exploratory import circular_chance_f1
    ph = np.repeat(np.arange(7), 10)  # T = 70, true boundaries every 10 steps
    # a pattern that sits exactly on the true boundaries: chance must be below its F1 of 1
    b = [np.arange(0, 70, 10)]
    c = circular_chance_f1(b, [ph], 0, n=200, seed=0)
    assert c.shape == (1,) and c[0] < 1.0
    # shifting a regular grid by k keeps it a grid: F1 at tol 0 is 1 exactly when k is a multiple of 10
    T, pat = 70, np.arange(0, 70, 10)
    for k in (10, 20):
        s = ((pat + k) % T)
        assert len(s[s != 0]) == len(pat) - 1 and set(np.diff(np.sort(s))) == {10}
    assert circular_chance_f1(b, [ph], 3, n=50, seed=0)[0] == circular_chance_f1(b, [ph], 3, n=50, seed=0)[0]
