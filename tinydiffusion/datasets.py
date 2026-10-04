"""2D toy datasets. Each returns an (n, 2) float32 array, roughly zero-mean with unit-ish scale."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def moons(n: int, rng: np.random.Generator) -> np.ndarray:
    """Two interleaving half circles."""
    k = n // 2
    t1, t2 = rng.uniform(0, math.pi, k), rng.uniform(0, math.pi, n - k)
    a = np.stack([np.cos(t1), np.sin(t1)], 1)
    b = np.stack([1 - np.cos(t2), 0.5 - np.sin(t2)], 1)
    x = np.concatenate([a, b]) + rng.normal(0, 0.06, (n, 2))
    return (x - [0.5, 0.25]) * 1.6


def spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    """A single noisy spiral arm."""
    t = np.sqrt(rng.uniform(0, 1, n)) * 3 * math.pi
    x = np.stack([t * np.cos(t), t * np.sin(t)], 1) / (3 * math.pi) * 2.2
    return x + rng.normal(0, 0.05, (n, 2))


def rings(n: int, rng: np.random.Generator) -> np.ndarray:
    """Three concentric circles."""
    r = rng.choice([0.6, 1.3, 2.0], n)
    t = rng.uniform(0, 2 * math.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, 0.05, (n, 2))


def checkerboard(n: int, rng: np.random.Generator) -> np.ndarray:
    """Alternating squares on a 4x4 board (sharp edges are hard!)."""
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    keep = (np.floor(x) + np.floor(y)) % 2 == 0
    pts = np.stack([x[keep], y[keep]], 1)
    while len(pts) < n:
        pts = np.concatenate([pts, checkerboard(n - len(pts), rng)])
    return pts[:n]


def gaussians(n: int, rng: np.random.Generator) -> np.ndarray:
    """Eight Gaussian blobs on a circle (tests mode coverage)."""
    centers = np.array([(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]) * 2
    return centers[rng.integers(0, 8, n)] + rng.normal(0, 0.12, (n, 2))


def heart(n: int, rng: np.random.Generator) -> np.ndarray:
    """A parametric heart curve."""
    t = rng.uniform(0, 2 * math.pi, n)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.stack([x, y], 1) / 8.5 + rng.normal(0, 0.04, (n, 2))


def text(n: int, rng: np.random.Generator, word: str = "AI") -> np.ndarray:
    """Points sampled uniformly from the pixels of a rendered word."""
    try:
        font = ImageFont.load_default(size=64)
    except TypeError:  # very old Pillow: small bitmap font
        font = ImageFont.load_default()
    img = Image.new("L", (64 * len(word), 96), 0)
    ImageDraw.Draw(img).text((4, 4), word, fill=255, font=font)
    ys, xs = np.nonzero(np.asarray(img) > 128)
    idx = rng.integers(0, len(xs), n)
    pts = np.stack([xs[idx], -ys[idx]], 1).astype(np.float64) + rng.uniform(0, 1, (n, 2))
    pts -= pts.mean(0)
    return pts / pts.std() * 0.9


DATASETS = {"moons": moons, "spiral": spiral, "rings": rings, "checkerboard": checkerboard, "gaussians": gaussians, "heart": heart, "text": text}


def sample_dataset(name: str, n: int, seed: int | None = None) -> np.ndarray:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name](n, np.random.default_rng(seed)).astype(np.float32)
