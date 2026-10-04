"""Training loop, samplers and checkpoint I/O."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .datasets import sample_dataset
from .model import Denoiser
from .schedule import NoiseSchedule


@dataclass
class TrainConfig:
    dataset: str = "moons"
    steps: int = 4000
    batch_size: int = 512
    lr: float = 2e-3
    T: int = 200
    schedule: str = "cosine"
    width: int = 128
    depth: int = 4
    ema_decay: float = 0.995
    pool_size: int = 50_000
    seed: int = 0
    log_every: int = 250


@dataclass
class TrainResult:
    model: Denoiser
    schedule: NoiseSchedule
    config: TrainConfig
    losses: list[float] = field(default_factory=list)
    seconds: float = 0.0


def train(cfg: TrainConfig, log=print) -> TrainResult:
    torch.manual_seed(cfg.seed)
    data = torch.from_numpy(sample_dataset(cfg.dataset, cfg.pool_size, seed=cfg.seed))
    sched = NoiseSchedule(cfg.T, cfg.schedule)
    model = Denoiser(width=cfg.width, depth=cfg.depth)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    lr_at = lambda s: cfg.lr * min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / cfg.steps))  # noqa: E731 warmup + cosine
    losses, running, t0 = [], None, time.perf_counter()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x0 = data[torch.randint(0, len(data), (cfg.batch_size,))]
        t = torch.randint(0, cfg.T, (cfg.batch_size,))
        noise = torch.randn_like(x0)
        loss = torch.nn.functional.mse_loss(model(sched.q_sample(x0, t, noise), t), noise)  # the whole DDPM objective
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.lerp_(pm, 1 - cfg.ema_decay)
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        losses.append(loss.item())
        if log and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            log(f"step {step:5d}/{cfg.steps}  loss {running:.4f}  lr {lr_at(step):.2e}")
    return TrainResult(ema.eval(), sched, cfg, losses, time.perf_counter() - t0)


@torch.no_grad()
def sample_ddpm(model: Denoiser, sched: NoiseSchedule, n: int, seed: int = 0, keep_every: int = 0) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 2), generator=g)
    traj = [(sched.T, x.clone())] if keep_every else []
    for t in reversed(range(sched.T)):
        eps = model(x, torch.full((n,), t))
        x = sched.ddpm_step(x, t, eps, generator=g)
        if keep_every and (t % keep_every == 0):
            traj.append((t, x.clone()))
    return x, traj


@torch.no_grad()
def sample_ddim(model: Denoiser, sched: NoiseSchedule, n: int, steps: int = 25, seed: int = 0, keep_every: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 2), generator=g)
    ts = torch.linspace(sched.T - 1, 0, steps).round().long().tolist()
    traj = [(sched.T, x.clone())] if keep_every else []
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        x = sched.ddim_step(x, t, t_prev, model(x, torch.full((n,), t)))
        if keep_every and (i % keep_every == 0 or t_prev == -1):
            traj.append((max(t_prev, 0), x.clone()))
    return x, traj


def save_checkpoint(res: TrainResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # store weights in half precision: these toy models lose nothing measurable and files stay tiny
    state = {k: v.half() for k, v in res.model.state_dict().items()}
    torch.save({"state_dict": state, "model_config": res.model.config, "train_config": asdict(res.config)}, path)
    return path


def load_checkpoint(path: str | Path) -> tuple[Denoiser, NoiseSchedule, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)  # weights_only: no arbitrary pickle execution
    model = Denoiser(**ck["model_config"])
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    cfg = ck["train_config"]
    return model.eval(), NoiseSchedule(cfg["T"], cfg["schedule"]), cfg
