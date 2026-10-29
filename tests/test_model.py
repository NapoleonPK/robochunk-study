import numpy as np
import pytest
import torch

from robochunk.data import pad_batch
from robochunk.model import Proxy, RoboChunk, gaussian_nll

torch.manual_seed(0)


def _model(encoder="gru", refractory=0):
    m = RoboChunk(n_ch=11, d0=32, d1=64, main_layers=2, heads=2, encoder=encoder, refractory=refractory)
    # random router weights so boundaries actually depend on the input
    with torch.no_grad():
        m.wq.weight.normal_()
        m.wk.weight.normal_()
    return m.eval()


@pytest.mark.parametrize("encoder", ["gru", "conv"])
@pytest.mark.parametrize("rule", ["learned", "fixed", "refractory"])
def test_causality(encoder, rule):
    m = _model(encoder, refractory=3 if rule == "refractory" else 0)
    T = 60
    x = torch.randn(1, T, 11)
    mask = torch.ones(1, T, dtype=torch.bool)
    b = torch.zeros(1, T, dtype=torch.int8)
    b[:, ::7] = 1
    bnd = b if rule == "fixed" else None
    with torch.no_grad():
        ref = m(x, mask, bnd)
        for t in [0, 10, 31, 58]:
            x2 = x.clone()
            x2[:, t + 1:] = torch.randn(1, T - t - 1, 11) * 5
            out = m(x2, mask, bnd)
            for k in ["mu", "logvar", "p"]:
                assert torch.allclose(ref[k][:, :t + 1], out[k][:, :t + 1], atol=1e-5), (k, t)
            assert torch.equal(ref["b"][:, :t + 1], out["b"][:, :t + 1])


def test_proxy_causality():
    m = Proxy(11, 16).eval()
    x = torch.randn(1, 40, 11)
    with torch.no_grad():
        a = m(x)["mu"]
        x[:, 21:] += 3
        b = m(x)["mu"]
    assert torch.allclose(a[:, :21], b[:, :21], atol=1e-6)


@pytest.mark.parametrize("rule", ["learned", "fixed", "flat"])
def test_batch_equals_alone(rule):
    """Episodes of very different length and chunk count give the same per-episode loss batched or alone."""
    m = _model()
    rng = np.random.default_rng(0)
    lens = [12, 140, 57, 90]
    seqs = [rng.normal(size=(L, 11)).astype(np.float32) for L in lens]
    bnds = None
    if rule == "fixed":
        bnds = [(np.arange(L) % (3 + i) == 0).astype(np.int8) for i, L in enumerate(lens)]
    if rule == "flat":
        bnds = [np.ones(L, np.int8) for L in lens]

    def per_ep_loss(xs, bs):
        x, mask, b = pad_batch(xs, bs)
        with torch.no_grad():
            nll, valid = gaussian_nll(m(x, mask, b), x, mask)
        return [(nll[i][valid[i]]).sum().item() for i in range(len(xs))]

    batched = per_ep_loss(seqs, bnds)
    alone = [per_ep_loss([s], None if bnds is None else [bnds[i]])[0] for i, s in enumerate(seqs)]
    np.testing.assert_allclose(batched, alone, rtol=1e-4, atol=1e-3)


def test_fixed_rule_is_identity_dechunk():
    """With p = b, STE factor is 1 and EMA keeps the latest chunk unchanged."""
    m = _model()
    x = torch.randn(2, 30, 11)
    mask = torch.ones(2, 30, dtype=torch.bool)
    b = torch.zeros(2, 30, dtype=torch.int8)
    b[:, ::5] = 1
    out = m(x, mask, b)
    assert out["b"].sum().item() == 12
    assert torch.equal(out["p"], out["b"].float())


def test_refractory_gaps():
    m = _model(refractory=3)
    with torch.no_grad():
        out = m(torch.randn(4, 80, 11), torch.ones(4, 80, dtype=torch.bool))
    for row in out["b"]:
        idx = row.nonzero().flatten()
        assert (idx.diff() > 3).all()
