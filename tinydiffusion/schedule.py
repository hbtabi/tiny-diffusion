"""The forward (noising) process and the two reverse samplers.

Notation follows Ho et al. 2020 (DDPM) and Song et al. 2021 (DDIM):

    q(x_t | x_0) = N( sqrt(abar_t) * x_0,  (1 - abar_t) * I )
    x_t          = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,     eps ~ N(0, I)

The network learns eps_theta(x_t, t) ~ eps.
"""

from __future__ import annotations

import math

import torch


class NoiseSchedule:
    def __init__(self, T: int = 200, kind: str = "cosine", beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T, self.kind = T, kind
        if kind == "linear":
            betas = torch.linspace(beta_start, beta_end * 1000 / T, T, dtype=torch.float64)
        elif kind == "cosine":  # Nichol & Dhariwal 2021: smoother, less wasted signal at the end
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            betas = torch.clamp(1 - f[1:] / f[:-1], 1e-6, 0.999)
        else:
            raise ValueError("kind must be 'linear' or 'cosine'")
        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        self.betas = betas.float()
        self.alphas = alphas.float()
        self.abar = abar.float()
        self.abar_prev = abar_prev.float()
        # variance of q(x_{t-1} | x_t, x_0) - the "beta tilde" choice for sigma_t^2
        self.posterior_var = (betas * (1 - abar_prev) / (1 - abar)).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.abar[t].unsqueeze(-1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def predict_x0(self, xt: torch.Tensor, t: int, eps: torch.Tensor, clip: float | None = 4.0) -> torch.Tensor:
        """Invert q_sample. At large t, sqrt(abar_t) is tiny and small eps errors explode, so we clip
        the estimate to a box around the data (a standard trick; the data lives well inside [-4, 4])."""
        a = self.abar[t]
        x0 = (xt - (1 - a).sqrt() * eps) / a.sqrt()
        return x0.clamp(-clip, clip) if clip else x0

    def ddpm_step(self, xt: torch.Tensor, t: int, eps: torch.Tensor, generator: torch.Generator | None = None, clip: float | None = 4.0) -> torch.Tensor:
        """One ancestral sampling step x_t -> x_{t-1}, using the posterior q(x_{t-1} | x_t, x0_hat)."""
        beta, alpha, abar, abar_prev = self.betas[t], self.alphas[t], self.abar[t], self.abar_prev[t]
        x0 = self.predict_x0(xt, t, eps, clip)
        mean = (beta * abar_prev.sqrt() / (1 - abar)) * x0 + ((1 - abar_prev) * alpha.sqrt() / (1 - abar)) * xt
        if t == 0:
            return mean
        z = torch.randn(xt.shape, generator=generator)
        return mean + self.posterior_var[t].sqrt() * z

    def ddim_step(self, xt: torch.Tensor, t: int, t_prev: int, eps: torch.Tensor, clip: float | None = 4.0) -> torch.Tensor:
        """Deterministic DDIM (eta=0) jump from t to t_prev (t_prev = -1 means 'clean')."""
        a = self.abar[t]
        x0 = self.predict_x0(xt, t, eps, clip)
        eps = (xt - a.sqrt() * x0) / (1 - a).sqrt()  # keep eps consistent with the clipped x0
        a_prev = self.abar[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
