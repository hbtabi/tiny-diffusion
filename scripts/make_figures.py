"""Regenerate the README figures from the bundled checkpoints: python scripts/make_figures.py"""

from tinydiffusion.cli import comparison, main

if __name__ == "__main__":
    comparison(["moons", "spiral", "gaussians", "checkerboard", "text"], "docs/samples_grid.png")
    main(["sample", "--ckpt", "text", "--n", "2500", "--gif", "docs/denoise_text.gif", "--png", "docs/reverse_text.png"])
    main(["sample", "--ckpt", "spiral", "--n", "2500", "--gif", "docs/denoise_spiral.gif"])
    main(["sample", "--ckpt", "moons", "--n", "2500", "--sampler", "ddim", "--ddim-steps", "25", "--png", "docs/reverse_moons_ddim.png"])
