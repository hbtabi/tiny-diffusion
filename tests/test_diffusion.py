import numpy as np
import pytest
import torch

from tinydiffusion.datasets import DATASETS, sample_dataset
from tinydiffusion.diffusion import TrainConfig, load_checkpoint, sample_ddim, sample_ddpm, save_checkpoint, train
from tinydiffusion.metrics import chamfer, evaluate, mmd
from tinydiffusion.model import Denoiser, timestep_embedding
from tinydiffusion.schedule import NoiseSchedule


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_datasets_shape_and_scale(name):
    x = sample_dataset(name, 1000, seed=0)
    assert x.shape == (1000, 2) and x.dtype == np.float32
    assert np.abs(x).max() < 4 and 0.3 < x.std() < 2.0
    assert np.allclose(x, sample_dataset(name, 1000, seed=0))  # seeded


def test_unknown_dataset():
    with pytest.raises(ValueError):
        sample_dataset("donut", 10)


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_properties(kind):
    s = NoiseSchedule(100, kind)
    assert torch.all(s.betas > 0) and torch.all(s.betas < 1)
    assert torch.all(s.abar[1:] < s.abar[:-1])  # signal monotonically destroyed
    assert s.abar[0] > 0.99 and s.abar[-1] < 0.05  # starts clean, ends ~pure noise
    assert torch.all(s.posterior_var >= 0)


def test_q_sample_statistics():
    s = NoiseSchedule(100)
    x0 = torch.full((20000, 2), 1.5)
    t = torch.full((20000,), 50)
    xt = s.q_sample(x0, t)
    a = s.abar[50]
    assert torch.allclose(xt.mean(0), torch.full((2,), 1.5) * a.sqrt(), atol=0.03)
    assert torch.allclose(xt.var(0), (1 - a).expand(2), atol=0.03)


def test_perfect_eps_recovers_x0():
    """With the true noise, predict_x0 and a full DDIM jump reconstruct the data exactly."""
    s = NoiseSchedule(100)
    x0 = torch.randn(64, 2)
    eps = torch.randn(64, 2)
    xt = s.q_sample(x0, torch.full((64,), 60), eps)
    assert torch.allclose(s.predict_x0(xt, 60, eps, clip=None), x0, atol=1e-4)
    assert torch.allclose(s.ddim_step(xt, 60, -1, eps, clip=None), x0, atol=1e-4)


def test_timestep_embedding_and_model_shapes():
    e = timestep_embedding(torch.tensor([0, 5, 199]), 64)
    assert e.shape == (3, 64) and torch.allclose(e[0, :32], torch.zeros(32))
    m = Denoiser()
    assert m(torch.randn(7, 2), torch.randint(0, 200, (7,))).shape == (7, 2)


def test_metrics_sanity():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(500, 2)), rng.normal(size=(500, 2))
    far = b + 5
    assert mmd(a, b) < mmd(a, far) and chamfer(a, b) < chamfer(a, far)


def test_short_training_beats_noise(tmp_path):
    res = train(TrainConfig(dataset="gaussians", steps=600, batch_size=256, T=100, pool_size=5000, log_every=10_000), log=None)
    assert np.mean(res.losses[-50:]) < np.mean(res.losses[:50])
    x, traj = sample_ddpm(res.model, res.schedule, 800, seed=1, keep_every=25)
    assert len(traj) == 5 and traj[0][0] == 100 and traj[-1][0] == 0
    m = evaluate(x.numpy(), sample_dataset("gaussians", 800, seed=99))
    assert m["mmd"] < 0.5 * m["mmd_noise_baseline"], m
    # checkpoint round-trip is (fp16-)faithful and loads with weights_only=True
    path = save_checkpoint(res, tmp_path / "g.pt")
    model, sched, cfg = load_checkpoint(path)
    assert cfg["dataset"] == "gaussians" and sched.T == 100
    xd, _ = sample_ddim(model, sched, 200, steps=10)
    assert torch.isfinite(xd).all() and xd.abs().max() < 6
