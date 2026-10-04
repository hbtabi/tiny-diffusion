"""Matplotlib helpers: denoising GIF, before/after grids and loss curves."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

BG, FG, ACCENT = "#0d1117", "#e6edf3", "#ff7b72"


def _style(ax, lim=3.0):
    ax.set_facecolor(BG)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#30363d")


def _fig_to_img(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=fig.dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def denoise_gif(traj: list[tuple[int, np.ndarray]], path: str | Path, title: str = "", T: int = 200, fps: int = 12) -> Path:
    frames = []
    for t, x in traj:
        fig, ax = plt.subplots(figsize=(4, 4.3), dpi=80, facecolor=BG)
        _style(ax, 3.2)
        color = plt.cm.plasma(0.2 + 0.8 * (1 - t / T))
        ax.scatter(x[:, 0], x[:, 1], s=2, color=color, alpha=0.7, linewidths=0)
        ax.set_title(f"{title}  t = {t:3d}", color=FG, fontsize=11, family="monospace")
        fig.tight_layout()
        frames.append(_fig_to_img(fig))
    frames += [frames[-1]] * fps  # hold the final frame for a second
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def snapshot_strip(traj: list[tuple[int, np.ndarray]], path: str | Path, title: str = "", k: int = 6) -> Path:
    idx = np.linspace(0, len(traj) - 1, k).round().astype(int)
    fig, axes = plt.subplots(1, k, figsize=(2.3 * k, 2.6), dpi=100, facecolor=BG)
    for ax, i in zip(axes, idx, strict=True):
        t, x = traj[i]
        _style(ax, 3.2)
        ax.scatter(x[:, 0], x[:, 1], s=1.2, color=plt.cm.plasma(0.2 + 0.8 * (1 - t / traj[0][0])), alpha=0.7, linewidths=0)
        ax.set_title(f"t = {t}", color=FG, fontsize=10, family="monospace")
    if title:
        fig.suptitle(title, color=FG, fontsize=12)
    fig.tight_layout()
    path = Path(path)
    _fig_to_img(fig).save(path, optimize=True)
    return path


def comparison_grid(rows: list[tuple[str, np.ndarray, np.ndarray, dict]], path: str | Path) -> Path:
    """rows: (name, real points, generated points, metrics)."""
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), dpi=100, facecolor=BG)
    axes = np.atleast_2d(axes).reshape(2, n)
    for j, (name, real, gen, m) in enumerate(rows):
        for i, (pts, color, label) in enumerate([(real, "#79c0ff", "data"), (gen, ACCENT, "generated")]):
            ax = axes[i, j]
            _style(ax, 3.0)
            ax.scatter(pts[:, 0], pts[:, 1], s=1.0, color=color, alpha=0.6, linewidths=0)
            if i == 0:
                ax.set_title(name, color=FG, fontsize=12)
            else:
                ax.set_xlabel(f"MMD {m['mmd']:.4f}\n(noise {m['mmd_noise_baseline']:.3f})", color="#8b949e", fontsize=9)
            if j == 0:
                ax.set_ylabel(label, color=FG, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    _fig_to_img(fig).save(path, optimize=True)
    return path


def loss_plot(losses: list[float], path: str | Path, title: str = "training loss") -> Path:
    w = max(1, len(losses) // 100)
    smooth = np.convolve(losses, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100, facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(losses, color="#30363d", lw=0.6)
    ax.plot(np.arange(len(smooth)) + w // 2, smooth, color=ACCENT, lw=1.6)
    ax.set_title(title, color=FG)
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values():
        s.set_color("#30363d")
    fig.tight_layout()
    path = Path(path)
    _fig_to_img(fig).save(path, optimize=True)
    return path
