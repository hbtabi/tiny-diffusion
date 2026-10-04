from pathlib import Path

import pytest
from PIL import Image

from tinydiffusion.cli import BUNDLED, main


def test_datasets_listing(capsys):
    assert main(["datasets"]) == 0
    assert "moons" in capsys.readouterr().out


@pytest.mark.skipif(not (BUNDLED / "moons.pt").exists(), reason="bundled checkpoint missing")
def test_sample_gif_png_from_bundled(tmp_path, capsys):
    gif, png = tmp_path / "d.gif", tmp_path / "d.png"
    assert main(["sample", "--ckpt", "moons", "--n", "300", "--gif", str(gif), "--png", str(png)]) == 0
    im = Image.open(gif)
    assert im.is_animated and im.n_frames > 10
    assert Image.open(png).size[0] > 600


@pytest.mark.skipif(not (BUNDLED / "moons.pt").exists(), reason="bundled checkpoint missing")
def test_eval_bundled_quality(capsys):
    import json

    assert main(["eval", "--ckpt", "moons", "--n", "1000"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["mmd"] < 0.1 * m["mmd_noise_baseline"]


def test_train_cli_tiny(tmp_path):
    out = tmp_path / "m.pt"
    assert main(["train", "--dataset", "rings", "--steps", "50", "--T", "50", "--out", str(out), "--loss-plot", str(tmp_path / "l.png")]) == 0
    assert out.exists() and Path(tmp_path / "l.png").exists()


def test_missing_ckpt():
    with pytest.raises(SystemExit):
        main(["sample", "--ckpt", "nope"])
