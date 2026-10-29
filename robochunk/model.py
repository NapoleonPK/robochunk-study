"""RoboChunk: a 1-stage H-Net (Hwang, Wang & Gu 2025) ported to continuous trajectories.

encoder (causal, per step) -> router -> downsample -> main Transformer on chunks
-> dechunk (EMA + causal upsample + STE) + residual -> decoder (causal) -> Gaussian head.
Every component is causal; the output at step t predicts x(t+1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0


class ConvBlock(nn.Module):
    """Causal depthwise conv (kernel 4) + gated MLP: a cheap stand-in for a Mamba layer."""

    def __init__(self, d, k=4):
        super().__init__()
        self.k = k
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, groups=d)
        self.w1, self.w3 = nn.Linear(d, 2 * d), nn.Linear(d, 2 * d)
        self.w2 = nn.Linear(2 * d, d)

    def forward(self, x):
        h = self.n1(x).transpose(1, 2)
        h = self.conv(F.pad(h, (self.k - 1, 0))).transpose(1, 2)
        x = x + h
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class SeqStack(nn.Module):
    """Causal per-step stack: 'gru' (2 layers) or 'conv' (4 ConvBlocks)."""

    def __init__(self, d, kind="gru", layers=None):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(d)  # H-Net puts a norm at the end of every encoder/decoder/main network
        if kind == "gru":
            self.net = nn.GRU(d, d, num_layers=layers or 2, batch_first=True)
        elif kind == "conv":
            self.net = nn.Sequential(*[ConvBlock(d) for _ in range(layers or 4)])
        else:
            raise ValueError(kind)

    def forward(self, x):
        return self.norm(self.net(x)[0] if self.kind == "gru" else self.net(x))


class MainNet(nn.Module):
    """Causal pre-norm Transformer over chunks with learned chunk-index positions."""

    def __init__(self, d_in, d=256, layers=4, heads=4, max_len=512):
        super().__init__()
        self.inp, self.out = nn.Linear(d_in, d), nn.Linear(d, d_in)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, z):
        K = z.shape[1]
        h = self.inp(z) + self.pos(torch.arange(K, device=z.device))
        mask = torch.triu(torch.full((K, K), float("-inf"), device=z.device), 1)
        return self.out(self.norm(self.tf(h, mask=mask, is_causal=True)))


def ste(c):
    """Forward value 1, gradient of c (straight-through)."""
    return c + (1 - c).detach()


def ratio_loss(b, p, mask, N):
    """H-Net ratio loss: N/(N-1) * ((N-1) F G + (1-F)(1-G)), F = boundary fraction, G = mean p."""
    m = mask.float()
    Fr = (b.float() * m).sum() / m.sum()
    G = (p * m).sum() / m.sum()
    return N / (N - 1) * ((N - 1) * Fr * G + (1 - Fr) * (1 - G))


class RoboChunk(nn.Module):
    def __init__(self, n_ch, d0=128, d1=256, main_layers=4, heads=4, encoder="gru", router_norm=False,
                 identity_init=True, max_chunks=512):
        super().__init__()
        self.n_ch = n_ch
        self.inp = nn.Linear(n_ch, d0)
        self.encoder = SeqStack(d0, encoder)
        self.router_norm = nn.LayerNorm(d0) if router_norm else nn.Identity()
        self.wq, self.wk = nn.Linear(d0, d0, bias=False), nn.Linear(d0, d0, bias=False)
        if identity_init:
            with torch.no_grad():
                self.wq.weight.copy_(torch.eye(d0))
                self.wk.weight.copy_(torch.eye(d0))
        self.main = MainNet(d0, d1, main_layers, heads, max_chunks)
        self.residual = nn.Linear(d0, d0)
        with torch.no_grad():  # H-Net initialises the residual projection close to 0
            self.residual.weight.mul_(1e-3)
            self.residual.bias.zero_()
        self.decoder = SeqStack(d0, "gru")
        self.head = nn.Linear(d0, 2 * n_ch)

    def route(self, xh):
        r = self.router_norm(xh)
        q, k = self.wq(r), self.wk(r)
        cos = F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)
        p = (0.5 * (1 - cos)).clamp(0, 1)
        return torch.cat([torch.ones_like(p[:, :1]), p], 1)

    def forward(self, x, mask, boundaries=None):
        """x: (B,T,C) normalised; mask: (B,T) bool valid steps; boundaries: (B,T) 0/1 for a fixed
        rule, or None to use the learned router. Returns dict(mu, logvar, p, b)."""
        B, T, _ = x.shape
        xh = self.encoder(self.inp(x))
        if boundaries is None:
            p = self.route(xh)
            b = (p >= 0.5) & mask
        else:
            b = boundaries.bool() & mask
            p = b.float()
        b = b.clone()
        b[:, 0] = True
        # downsample: keep boundary steps
        idx = b.long().cumsum(1) - 1                     # chunk index of each step
        K = int(b.sum(1).max())
        bi, ti = b.nonzero(as_tuple=True)
        chunks = xh.new_zeros(B, K, xh.shape[-1])
        chunks[bi, idx[bi, ti]] = xh[bi, ti]
        P = xh.new_zeros(B, K)
        P[bi, idx[bi, ti]] = p[bi, ti]
        zh = self.main(chunks)
        # dechunk: EMA over chunks (identity when every P is 1, i.e. for fixed rules)
        if boundaries is not None:
            zbar = zh
        else:
            zs, prev = [], torch.zeros_like(zh[:, 0])
            for j in range(K):
                prev = P[:, j:j + 1] * zh[:, j] + (1 - P[:, j:j + 1]) * prev
                zs.append(prev)
            zbar = torch.stack(zs, 1)
        up = torch.gather(zbar, 1, idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, zbar.shape[-1]))
        c = torch.where(b, p, 1 - p)
        up = up * ste(c).unsqueeze(-1)
        h = self.decoder(up + self.residual(xh))
        mu, logvar = self.head(h).chunk(2, -1)
        return dict(mu=mu, logvar=logvar.clamp(LOGVAR_MIN, LOGVAR_MAX), p=p, b=b)


class Proxy(nn.Module):
    """Small causal next-step model for the entropy (BLT-like) baseline."""

    def __init__(self, n_ch, d=64):
        super().__init__()
        self.inp = nn.Linear(n_ch, d)
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True)
        self.head = nn.Linear(d, 2 * n_ch)

    def forward(self, x, mask=None, boundaries=None):
        mu, logvar = self.head(self.gru(self.inp(x))[0]).chunk(2, -1)
        return dict(mu=mu, logvar=logvar.clamp(LOGVAR_MIN, LOGVAR_MAX))


def gaussian_nll(out, x, mask):
    """Per-step NLL in nats of x(t+1) given output at t, summed over channels.
    Returns (B, T-1) values and the (B, T-1) validity mask."""
    mu, lv = out["mu"][:, :-1], out["logvar"][:, :-1]
    y = x[:, 1:]
    nll = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv) + math.log(2 * math.pi)).sum(-1)
    valid = mask[:, 1:] & mask[:, :-1]
    return nll, valid


def step_surprise(model, x, mask):
    """NLL of x(t) given x(<t), aligned to step t (step 0 gets +inf). Used by the entropy baseline."""
    nll, valid = gaussian_nll(model(x, mask), x, mask)
    s = torch.full(x.shape[:2], float("inf"), device=x.device)
    s[:, 1:] = torch.where(valid, nll, torch.full_like(nll, float("inf")))
    return s
