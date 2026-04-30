# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ChromGP is a research codebase for inferring 3D chromatin structure from Hi-C-style contact / distance data using deep Gaussian Processes built on top of the sibling **GPzoo** package (`../GPzoo`, importable as `gpzoo`). The repo itself is small library code + a large set of experimental Jupyter notebooks; almost all driver code lives in the notebooks.

## Install & dependencies

Dependencies live in `pyproject.toml` (Python 3.14+, GPU PyTorch via the cu128 wheel index). `gpzoo` and `bioframe` are intentionally **not** listed there — they are sibling repos installed editable so live edits propagate.

```bash
conda create -n chromgp python=3.14 -y
conda activate chromgp

# Editable installs from sibling clones — torch comes in via this repo's pyproject
pip install -e ../GPzoo
pip install -e ../bioframe
pip install -e .                  # this repo (jupyter / ipykernel come in as core deps)

python -m ipykernel install --user --name chromgp --display-name "Python (chromgp)"
```

The same `python=3.14` + GPU `torch` combo is what the sibling `factorization` conda env uses, so it is known to work on this machine. The default PyPI torch wheel on Linux x86_64 already ships with CUDA — no extra `--index-url` step is needed.

Notebooks expect data files via symlinks into `/gladstone/engelhardt/lab/lchumpitaz/hi-c/`: `notebooks/hg38.fa` and `notebooks/hg38.fa.fai` (reference genome) and the repo-root `mcools/` symlink (`.mcool` Hi-C files). `notebooks/hg38_gc_cov_15kb.tsv` lives directly in the repo.

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

### ChromHMM data sources

Three ChromHMM models are available in `/gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/`. All three are for **GM12878** — they differ in which consortium produced them, which histone marks were used for training, and how Polycomb is resolved.

| File | Consortium | States | Polycomb states | Polycomb on chr9 | Source |
|------|-----------|--------|----------------|-------------------|--------|
| `ENCFF140VIG.bed` | ENCODE (Weng Lab) | 15 | ReprPC, Biv | 2.5M bp (1.8%) | ENCODE portal |
| `ENCFF338RIC.bed` | ENCODE (Weng Lab) | 18 | ReprPC, ReprPCWk, TssBiv, EnhBiv | — | ENCODE portal |
| `E116_15_coreMarks_hg38lift_mnemonics.bed.gz` | NIH Roadmap (E116) | 15 | TssBiv, BivFlnk, EnhBiv, ReprPC, ReprPCWk | 13.8M bp (10.0%) | `egg2.wustl.edu/roadmap/` |

**Why the large Polycomb discrepancy (1.8% vs 10.0% on chr9)?** Both ENCFF140VIG and E116 are GM12878 15-state models, but they were trained on different histone mark panels:

- **Roadmap E116 "core marks" model**: trained on 5 marks — H3K4me3, H3K4me1, H3K36me3, **H3K27me3**, H3K9me3. H3K27me3 is the canonical Polycomb/PRC2 mark, giving this model strong power to resolve repressed chromatin into distinct sub-states (TssBiv, BivFlnk, EnhBiv, ReprPC, ReprPCWk).
- **ENCFF140VIG**: trained on a different set of ENCODE ChIP-seq marks for GM12878. Without the same H3K27me3-driven resolution, most ambiguous/weakly-repressed regions collapse into the Quiescent state instead of being called as Polycomb.

**Practical impact**: the HMMC reference notebook (`../HMMC/extra_notebooks/Plot_Polycomb.ipynb`) uses the Roadmap E116 model. For reproducing figures or benchmarking against HMMC, use the E116 file. The ENCFF140VIG model under-calls Polycomb by ~5.5× on chr9 relative to the Roadmap gold standard.

The merge map in `chromgp/datasets/chromhmm.py` handles all three naming conventions (ENCODE 15-state, ENCODE 18-state, Roadmap E116 numbered-prefix style).

### ChromHMM state grouping

Fine-grained ChromHMM states are merged into 5 coarse biological groups **before** bin-majority assignment (in `hic.py`, via `chromhmm.merge_chromhmm_groups`). The merge map in `chromgp/datasets/chromhmm.py` handles all three naming conventions above. This reduces extreme Quiescent imbalance and makes minority groups more interpretable.

| Group           | States included                                                              |
|-----------------|------------------------------------------------------------------------------|
| Active          | Tss, TssFlnk, TssFlnkD, TssFlnkU, Enh1, Enh2, EnhG1, EnhG2, TssA, EnhA1, EnhA2, EnhWk |
| Transcribed     | Tx, TxWk                                                                    |
| Heterochromatin | Het, ZNF/Rpts                                                               |
| Polycomb        | ReprPC, ReprPCWk, Biv, TssBiv, EnhBiv                                      |
| Quiescent       | Quies (and any unmapped states)                                             |

### Notebooks

## Git conventions

**Never commit or push unless explicitly told to.** Make code changes freely, but always wait for the user to say "commit" or "push" before running any `git commit` or `git push` command.

## Tmux conventions

When launching a long-running command (training, preprocessing, etc.):

1. **Check which session we're in**: `tmux display-message -p '#{session_name}:#{window_index}.#{pane_index}'`
2. **Open a pane below the current one**: `tmux split-window -v -p 30 -t <session>:<window>.<pane>` (30% height works well for a scrolling log)
3. **Activate conda and run**: `conda activate chromgp && chromgp ...` so output is visible without stealing focus from the main session.

This keeps training progress visible in a dedicated bottom pane while the top pane stays free for interactive work.

The `notebooks/` directory is the actual experimental surface — each notebook plugs a different GPzoo model (`SVGP`, `WSVGP`, `WVNNGP`, `VNNGP`, `MGGP_WSVGP`) and kernel (`batched_RBF`, `BatchedBrownianKernel`, `batched_MGGP_RBF`) into `ChromGP` or `IntegratedForceGP` and runs it on either a synthetic shape or a real `.mcool` Hi-C contact map. There is no shared notebook utility module beyond `chromgp/` itself; expect duplicated boilerplate across notebooks.
