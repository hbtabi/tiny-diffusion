"""tiny-diffusion CLI: datasets | train | sample | eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .datasets import DATASETS, sample_dataset
from .diffusion import TrainConfig, load_checkpoint, sample_ddim, sample_ddpm, save_checkpoint, train
from .metrics import evaluate
from .viz import comparison_grid, denoise_gif, loss_plot, snapshot_strip

BUNDLED = Path(__file__).resolve().parent.parent / "checkpoints"


def _resolve_ckpt(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    if (BUNDLED / f"{arg}.pt").exists():
        return BUNDLED / f"{arg}.pt"
    raise SystemExit(f"checkpoint not found: {arg} (bundled: {', '.join(sorted(x.stem for x in BUNDLED.glob('*.pt'))) or 'none'})")


def _sample(model, sched, args, keep_every=0):
    if args.sampler == "ddim":
        return sample_ddim(model, sched, args.n, steps=args.ddim_steps, seed=args.seed, keep_every=keep_every)
    return sample_ddpm(model, sched, args.n, seed=args.seed, keep_every=keep_every)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tiny-diffusion", description="A small denoising diffusion model from scratch (PyTorch, CPU)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="list datasets")
    t = sub.add_parser("train", help="train a model on a 2D toy dataset")
    t.add_argument("--dataset", default="moons", choices=sorted(DATASETS))
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--T", type=int, default=200, help="diffusion timesteps")
    t.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    t.add_argument("--width", type=int, default=128)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None, help="checkpoint path (default runs/<dataset>.pt)")
    t.add_argument("--loss-plot", default=None)
    for name in ("sample", "eval"):
        p = sub.add_parser(name, help="generate points / measure sample quality")
        p.add_argument("--ckpt", default="moons", help="path or bundled name (moons, spiral, text...)")
        p.add_argument("--n", type=int, default=2000)
        p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
        p.add_argument("--ddim-steps", type=int, default=25)
        p.add_argument("--seed", type=int, default=0)
        if name == "sample":
            p.add_argument("--gif", help="write the denoising process as an animated GIF")
            p.add_argument("--png", help="write a snapshot strip of the reverse process")
            p.add_argument("--npy", help="save the generated points")
    args = ap.parse_args(argv)
    # These networks are tiny: one thread avoids OpenMP spin/oversubscription overhead and is usually fastest.
    torch.set_num_threads(1 if args.cmd != "train" else max(1, min(4, torch.get_num_threads())))

    if args.cmd == "datasets":
        for name in sorted(DATASETS):
            print(f"{name:<13} {(DATASETS[name].__doc__ or '').strip()}")
        return 0

    if args.cmd == "train":
        cfg = TrainConfig(dataset=args.dataset, steps=args.steps, batch_size=args.batch_size, lr=args.lr, T=args.T, schedule=args.schedule,
                          width=args.width, depth=args.depth, seed=args.seed)
        print(f"training on '{cfg.dataset}' for {cfg.steps} steps (T={cfg.T}, {cfg.schedule} schedule)")
        res = train(cfg)
        out = save_checkpoint(res, args.out or f"runs/{cfg.dataset}.pt")
        print(f"done in {res.seconds:.1f}s, {res.model.n_params():,} parameters -> {out} ({out.stat().st_size / 1024:.0f} KB)")
        if args.loss_plot:
            loss_plot(res.losses, args.loss_plot, f"loss: {cfg.dataset}")
        return 0

    model, sched, cfg = load_checkpoint(_resolve_ckpt(args.ckpt))
    if args.cmd == "eval":
        x, _ = _sample(model, sched, args)
        ref = sample_dataset(cfg["dataset"], args.n, seed=12345)  # fresh data the model never saw
        m = evaluate(x.numpy(), ref, seed=args.seed)
        print(json.dumps({"dataset": cfg["dataset"], "sampler": args.sampler, **{k: round(v, 5) for k, v in m.items()}}, indent=2))
        return 0

    keep = 0
    if args.gif or args.png:
        keep = max(1, sched.T // 50) if args.sampler == "ddpm" else 1
    x, traj = _sample(model, sched, args, keep_every=keep)
    traj = [(t, v.numpy()) for t, v in traj]
    if args.gif:
        print("gif ->", denoise_gif(traj, args.gif, title=cfg["dataset"], T=sched.T))
    if args.png:
        print("png ->", snapshot_strip(traj, args.png, title=f"reverse diffusion: {cfg['dataset']} ({args.sampler})"))
    if args.npy:
        np.save(args.npy, x.numpy())
        print("npy ->", args.npy)
    if not (args.gif or args.png or args.npy):
        np.savetxt(sys.stdout, x.numpy()[:20], fmt="%.4f", header="first 20 generated points (x y)")
    return 0


def comparison(names: list[str], out: str, n: int = 2000) -> None:
    """Used by scripts/make_figures.py: data vs samples for each bundled checkpoint."""
    torch.set_num_threads(1)
    rows = []
    for name in names:
        model, sched, cfg = load_checkpoint(BUNDLED / f"{name}.pt")
        x, _ = sample_ddpm(model, sched, n, seed=0)
        ref = sample_dataset(cfg["dataset"], n, seed=12345)
        rows.append((name, ref, x.numpy(), evaluate(x.numpy(), ref)))
    comparison_grid(rows, out)


if __name__ == "__main__":
    raise SystemExit(main())
