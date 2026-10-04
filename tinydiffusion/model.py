"""The denoiser: a residual MLP conditioned on the timestep through sinusoidal embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding of integer timesteps -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class ResBlock(nn.Module):
    def __init__(self, width: int, temb: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.temb = nn.Linear(temb, width)
        self.fc2 = nn.Linear(width, width)
        nn.init.zeros_(self.fc2.weight)  # each block starts as identity -> stable early training
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        z = self.fc1(self.norm(h)) + self.temb(te)
        return h + self.fc2(nn.functional.silu(z))


class Denoiser(nn.Module):
    def __init__(self, dim: int = 2, width: int = 128, depth: int = 4, temb: int = 64):
        super().__init__()
        self.config = {"dim": dim, "width": width, "depth": depth, "temb": temb}
        self.temb_dim = temb
        self.t_mlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Linear(dim, width)
        self.blocks = nn.ModuleList(ResBlock(width, temb) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_mlp(timestep_embedding(t, self.temb_dim))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, te)
        return self.out(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
