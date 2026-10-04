"""Sample-quality metrics for 2D point clouds.

* **MMD** (RBF kernel, multiple bandwidths): 0 means the distributions match.
* **Chamfer**: mean nearest-neighbour distance, both directions (penalises missing modes *and* stray points).

Both are reported next to the same numbers for pure Gaussian noise, so you can see how far training moved.
"""

from __future__ import annotations

import numpy as np


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)


def mmd(x: np.ndarray, y: np.ndarray, bandwidths=(0.05, 0.2, 0.5, 1.0)) -> float:
    dxx, dyy, dxy = _sqdist(x, x), _sqdist(y, y), _sqdist(x, y)
    total = 0.0
    for h in bandwidths:
        k = lambda d, h=h: np.exp(-d / (2 * h * h))  # noqa: E731
        total += k(dxx).mean() + k(dyy).mean() - 2 * k(dxy).mean()
    return float(max(total, 0.0))


def chamfer(x: np.ndarray, y: np.ndarray) -> float:
    d = np.sqrt(_sqdist(x, y))
    return float(d.min(1).mean() + d.min(0).mean())


def evaluate(samples: np.ndarray, reference: np.ndarray, seed: int = 0) -> dict[str, float]:
    noise = np.random.default_rng(seed).normal(size=samples.shape).astype(np.float32)
    return {"mmd": mmd(samples, reference), "mmd_noise_baseline": mmd(noise, reference),
            "chamfer": chamfer(samples, reference), "chamfer_noise_baseline": chamfer(noise, reference)}
