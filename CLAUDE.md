# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ChromGP is a research codebase for inferring 3D chromatin structure from Hi-C-style contact / distance data using deep Gaussian Processes built on top of the sibling **GPzoo** package (`../GPzoo`, importable as `gpzoo`). The repo itself is small library code + a large set of experimental Jupyter notebooks; almost all driver code lives in the notebooks.

## Install & dependencies

- Editable install: `pip install -e .` from the repo root. `setup.py` declares no dependencies — they come transitively from GPzoo, which must also be installed (typically `pip install -e .` in `../GPzoo`).
- Runtime imports rely on `torch`, `numpy`, `matplotlib`, `plotly`, `tqdm`, and `gpzoo.{gp,kernels,utilities}`. There is no requirements.txt and no test/lint/build configuration.
- Notebooks expect data files under `notebooks/`: `hg38.fa` / `hg38.fa.fai` (reference genome), `hg38_gc_cov_15kb.tsv`, and `.mcool` files reachable via the `mcools/` symlink (points to `/gladstone/engelhardt/lab/lchumpitaz/mcools`).

## Architecture

The `chromgp/` package is intentionally thin — it provides a wrapper that turns any GP from GPzoo into a generative model over pairwise distances/contacts, plus training and visualization helpers. The notebooks supply the GP, kernel, optimizer, and data.

### `chromgp/models.py` — the core wrapper

`ChromGP(nn.Module)` composes:
- `gp`: a GPzoo GP (e.g. `SVGP`, `WSVGP`, `WVNNGP`, `MGGP_WSVGP`) that returns variational posterior `qZ`, inducing posterior `qU`, and prior `pU` for a batch of inputs `X`.
- `kernel`: a *second* kernel applied in latent (3D) space (typically `batched_RBF` from `gpzoo.kernels`), used to build the predicted covariance over observations.
- `noise`: a softplus-positive scalar diagonal noise added to the predicted covariance.

The forward pass:
1. Samples `F ~ qZ` with `E` Monte-Carlo samples, transposes to `(E, N, D)`.
2. Calls `process_F(X, F) → Z` (identity by default; subclasses override).
3. Builds `Kzz = kernel(Z, Z) + jitter*I + noise²*I` (jitter via `gpzoo.utilities.add_jitter`).
4. Returns `pY = MultivariateNormal(0, Kzz)` along with `qZ, qU, pU`.

`IntegratedForceGP(ChromGP)` is the key subclass: it interprets the GP output as a force field `F(x)` and overrides `process_F` to trapezoidally integrate `F` over the (sorted) input coordinate to produce 3D positions `Z(x)`, then unsorts back to the input order. Use this when the underlying GP models forces rather than positions directly.

### `chromgp/utilities.py` — the training loop contract

Both `train` (full-batch) and `train_batched` (mini-batch with multinomial subsampling) implement the **same ELBO**:

```
y_norm = y - y.mean(dim=1, keepdim=True)        # mean-center each row of the distance/contact matrix
L1 = pY.log_prob(y_norm).sum()                  # (× N/batch_size in the batched version)
L2 = KL(qU || pU).sum()
loss = -(L1 - L2)
```

Targets `y` are expected to be `(M, N)` pairwise-distance-style observations (one row per simulation/replicate); inputs `X` are `(N,)` or `(N, 1)` 1D positions along a chromosome / polymer. Both functions also collect `qZ.mean.T` snapshots per step into `Zs` for animating the trajectory, which is the standard "reconstruction quality over training" diagnostic used throughout the notebooks.

When changing the model contract, the forward signature `(pY, qZ, qU, pU)` and the row-mean centering of `y` must be preserved or `train`/`train_batched` will silently break.

### `chromgp/simulations.py` — synthetic data

Generates ground-truth 3D shapes (`make_helix`, plus cylinder/spiral/sponge variants in notebooks), produces noisy replicate point clouds with `generate_simulations`, converts distance matrices to contact maps via a Poisson distance-decay model in `compute_contacts`, and renders side-by-side 3D-trajectory + reconstructed-distance + true-distance MP4 animations via `create_animation`.

### Notebooks

The `notebooks/` directory is the actual experimental surface — each notebook plugs a different GPzoo model (`SVGP`, `WSVGP`, `WVNNGP`, `VNNGP`, `MGGP_WSVGP`) and kernel (`batched_RBF`, `BatchedBrownianKernel`, `batched_MGGP_RBF`) into `ChromGP` or `IntegratedForceGP` and runs it on either a synthetic shape or a real `.mcool` Hi-C contact map. There is no shared notebook utility module beyond `chromgp/` itself; expect duplicated boilerplate across notebooks.
