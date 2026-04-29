# ChromGP Pipeline Plan

## Overview

Build a CLI-first pipeline mirroring `Spatial-Factorization`'s architecture, adapted for Hi-C (and later ChIP-seq) inputs. The user runs:

```bash
chromgp generate   -c configs/<dataset>/general.yaml   # expand to per-model configs
chromgp preprocess -c configs/<dataset>/<model>.yaml   # mcool → standardized arrays
chromgp train      -c configs/<dataset>/<model>.yaml   # fit GP, save checkpoint + ELBO
chromgp analyze    -c configs/<dataset>/<model>.yaml   # metrics, inferred 3D coords
chromgp figures    -c configs/<dataset>/<model>.yaml   # publication figures
```

**Models supported:** `SVGP`, `MGGP_SVGP`, `LCGP`, `MGGP_LCGP`. No PNMF — ChromGP infers 3D coordinates whose pairwise distances explain the observed contact matrix; that's a different problem from non-negative matrix factorization into latent factor loadings.

**Targets `Y`** are stored on disk as an `(N, D)` matrix where `N` = bins (the axis the GP covers) and `D` = columns of signal — Hi-C contact replicates for Hi-C, histone marks for ChIP-seq, or both concatenated for hybrid. Centering is applied per-column at training time via `y - y.mean(dim=0)` because ChromGP uses a zero-mean Gaussian prior; this differs from SF (Poisson model, no centering).

**Inputs:** `.mcool` files referenced by absolute path in the YAML config, with a chromosome / region selector (e.g. `chr1` or `chr1:1200000-50000000`). ChIP-seq is a future input type (separate dataset loader); ChromHMM state segmentations (BED-style) are an auxiliary input that supplies group labels for MGGP runs.

**MGGP groups come from one of two mutually exclusive sources:**

1. **Multiple chromosomes** — each chromosome in `regions:` is its own group. Use this to compare 3D structure across whole chromosomes.
2. **ChromHMM states within a single chromosome** — a BED file from ChromHMM tags every bin with a chromatin state (active TSS, enhancer, polycomb, heterochromatin, …). Each state becomes a group, so MGGP can share structure within a state and contrast across states inside one chromosome.

A given run picks one or the other, not both.

**Status legend:** `⬜` NOT DONE | `🟨` IN PROGRESS | `🟩` DONE

---

## Target package layout

```
chromgp/
├── __init__.py
├── __main__.py                      # python -m chromgp
├── cli.py                           # click group + 5 subcommands
├── config.py                        # Config dataclass + YAML loader
├── generate.py                      # general.yaml → per-model variants
├── runner.py                        # (later) parallel multi-model orchestration
├── models.py                        # ChromGP / IntegratedForceGP (existing)
├── simulations.py                   # (existing — keep for synthetic configs)
├── utilities.py                     # (existing — train/train_batched, may evolve)
├── analysis.py                      # reconstruction metrics, 3D coord helpers
├── commands/
│   ├── __init__.py
│   ├── generate.py
│   ├── preprocess.py
│   ├── train.py
│   ├── analyze.py
│   └── figures.py
└── datasets/
    ├── __init__.py
    ├── base.py                      # HicData / ChipSeqData containers
    ├── hic.py                       # mcool loader, region parsing, contact-matrix transforms
    ├── preprocessed.py              # load saved X.npy / Y.npy / metadata.json
    └── chipseq.py                   # (later)

configs/
└── <dataset_id>/                    # one dir per .mcool / sample
    ├── general.yaml                 # union of all model params
    ├── svgp.yaml                    # generated
    ├── mggp_svgp.yaml               # generated
    ├── lcgp.yaml                    # generated
    └── mggp_lcgp.yaml               # generated

outputs/
└── <dataset_id>/<region_slug>/<model>/
    ├── preprocessed/
    │   ├── X.npy                    # (N,) bin midpoints, normalized
    │   ├── Y.npy                    # (N, D) signal matrix: (N, D) for Hi-C, (N, D) for ChIP-seq
    │   ├── contact_raw.npy          # (N, N) untransformed contact matrix (kept for analyze-stage metrics, Hi-C only)
    │   ├── C.npy                    # (N,) group labels (MGGP only)
    │   └── metadata.json
    ├── model.pth                    # train output
    ├── elbo_history.csv             # train output
    ├── coords_3d.npy                # analyze output — the inferred 3D structure
    ├── predicted_distance.npy       # analyze output (pairwise Euclidean from coords_3d)
    ├── metrics.json                 # analyze output
    └── figures/                     # figures stage
```

`<region_slug>` examples: `chr1`, `chr1_1200000-50000000`, `chr1+chr2+chr3` (MGGP multi-region).

---

## Region grammar (in YAML)

Single region, whole chromosome:
```yaml
preprocessing:
  region: chr1
```

Single region, base-pair window (UCSC-style):
```yaml
preprocessing:
  region: chr1:1200000-50000000
```

Multi-region, MGGP groups by chromosome:
```yaml
preprocessing:
  regions:
    - chr1
    - chr2:0-100000000
    - chr3
  groups_by: chromosome
```

Single-region, MGGP groups by ChromHMM state:
```yaml
preprocessing:
  region: chr14
  groups_by: chromhmm_state
  chromhmm_bed: /gladstone/engelhardt/lab/lchumpitaz/datasets/chromhmm/GM12878_18state.bed
  chromhmm_states: [TssA, EnhA1, EnhA2, ReprPC, Het]   # optional whitelist
```

Parser + group resolver live in `chromgp/datasets/hic.py`; output `<region_slug>` is filesystem-safe.

---

## Dataset conventions

All datasets map onto **`(X, Y, C)` with `Y` shaped `(N, D)`** — matching Spatial-Factorization's convention. `N` = bins (the axis the GP covers), `D` = columns of signal. Row-mean centering happens per feature at training time via `y - y.mean(dim=0)`.

| Field | Hi-C | ChIP-seq | Hybrid (future) |
|---|---|---|---|
| `X` shape | `(N,)` | `(N,)` | `(N,)` |
| `X` semantics | bin midpoints (bp) | bin midpoints (bp) | bin midpoints (bp) |
| `Y` shape | `(N, D)` | `(N, D)` | `(N, N+D)` |
| `Y` column (axis=1) | one Hi-C replicate's contact vector | one histone mark's signal | concat(contact-vector, all-marks) |
| `Y` row (axis=0) | one bin's contacts across replicates | one bin's marks | one bin across all signals |
| Centering axis | **per replicate** (`mean(dim=0)`) | **per mark** (`mean(dim=0)`) | per column (each replicate, each mark) |
| Centering reason | Gaussian prior is zero-mean | Gaussian prior is zero-mean | Gaussian prior is zero-mean |
| `C` shape | `(N,)` | `(N,)` | `(N,)` |
| `C` source | `chromosome` OR `chromhmm_state` | same | same |

**Hi-C details:**
- `D` = number of replicates. Each column is one Hi-C contact vector (flattened upper triangle of the `(N, N)` contact matrix after applying `contact_transform`).
- `Y[:, d]` = replicate `d`'s contacts across bins. The training loop centers each column to zero mean.
- ChIP-seq *also* predicts a 3D structure — the same 3D output head works, just with a different `Y` signal driving the ELBO.

**ChIP-seq details:**
- `D` = number of histone marks. Each column is one mark's signal across bins (after `signal_transform`).
- Inputs are BigWig / bedGraph per mark, aligned to the same binning as a paired Hi-C config.

**Hybrid (future) Hi-C + ChIP-seq:**
- `Y` shape becomes `(N, N + D)` — first `N` columns are one Hi-C replicate's contact vector, next `D` columns are the per-mark signals.
- The GP still covers `N` bins and predicts 3D coords; the ELBO just sees more columns of evidence. This is a `contact_transform: hybrid` mode.

**Note:** ChromGP's existing code stores `(D, N)` and centers on `dim=1`. Part of Stage 3a is transposing to `(N, D)` and switching centering to `dim=0`. This is not for consistency with SF (SF is Poisson, no centering) — it's for consistency across ChromGP's own Gaussian-model variants and for clearer axis semantics.

---

## Config schema (sketch — finalize during Stage 1)

```yaml
name: 4DN_GM12878_chr14_svgp
seed: 67
dataset: 4DNFIXP4QG5B           # matches a directory under configs/
output_dir: outputs/4DNFIXP4QG5B

preprocessing:
  mcool_path: /gladstone/engelhardt/lab/lchumpitaz/hi-c/mcools/4DNFIXP4QG5B.mcool
  resolution: 25000             # bp per bin
  region: chr14                 # or chr14:20000000-100000000, or `regions: [...]`
  balance: true                 # use cooler ICE-balanced contacts
  contact_transform: log1p      # {log1p, obs_over_exp, raw}
  num_replicates: 16            # M, for synthetic multi-replicate generation
  noise_level: 0.15             # only when sampling distance replicates
  # MGGP-only fields (ignored when model.groups: false)
  groups_by: chromosome         # {chromosome, chromhmm_state} — one or the other, not both
  chromhmm_bed: ~               # path to ChromHMM segmentation BED (only when groups_by == chromhmm_state)
  chromhmm_states: ~            # optional whitelist of state names

model:
  prior: SVGP                   # {SVGP, LCGP}
  groups: false                 # MGGP if true
  E: 1                          # MC samples in forward
  n_components: 3               # 3D output
  kernel: RBF                   # {RBF, Brownian, Matern32}
  lengthscale: 8.0
  sigma: 1.0
  train_lengthscale: false
  num_inducing: 800
  cholesky_mode: exp
  noise: 0.1
  jitter: 1e-5
  integrated_force: false       # if true, use IntegratedForceGP

training:
  max_iter: 20000
  learning_rate: 2e-3
  optimizer: Adam
  device: gpu
  batch_size: null              # null = full-batch
  shuffle: true
```

Field categories (for `generate` filtering): `COMMON_MODEL_FIELDS`, `SPATIAL_FIELDS`, `LCGP_FIELDS`, `MGGP_FIELDS`. Mirror SF's pattern in `spatial_factorization/generate.py`.

---

## 5-Stage Plan

### Stage 0: CLI scaffold & install ⬜

**Goal:** `chromgp --help` works, prints the 5 subcommands.

- [ ] Add `click>=8.0` to `pyproject.toml` deps.
- [ ] Add `[project.scripts] chromgp = "chromgp.cli:cli"` to `pyproject.toml`.
- [ ] `chromgp/cli.py` — click group + 5 stub subcommands, each loading config and calling `commands.<stage>.run(config)`.
- [ ] `chromgp/__main__.py` — calls `cli()`.
- [ ] `chromgp/commands/{generate,preprocess,train,analyze,figures}.py` — stub `run(config_path)` that just prints "not implemented".
- [ ] `pip install -e .` in the `chromgp` env, verify:
  ```
  chromgp --help
  chromgp generate --help
  ```

---

### Stage 1: Generate command + Config dataclass ⬜

**Goal:** `chromgp generate -c configs/4DNFIXP4QG5B/general.yaml` writes the four model-specific YAMLs.

- [ ] `chromgp/config.py` — `Config` dataclass with `name`, `seed`, `dataset`, `preprocessing`, `model`, `training`, `output_dir`. `Config.from_yaml(path)` and `to_yaml(path)`.
- [ ] `chromgp/generate.py` — `MODEL_VARIANTS` list; field-filtering tables; entry point that reads `general.yaml` and writes per-model variants into the same dir.
- [ ] `configs/4DNFIXP4QG5B/general.yaml` — first real example using the lab's existing mcool. Single chromosome (`chr14`) at 25 kb is a reasonable starter.
- [ ] `chromgp/commands/generate.py` — invokes `chromgp.generate`.
- [ ] Verify generated `svgp.yaml`, `mggp_svgp.yaml`, `lcgp.yaml`, `mggp_lcgp.yaml` round-trip through `Config.from_yaml`.

---

### Stage 2: Preprocess command ⬜

**Goal:** `chromgp preprocess -c …/svgp.yaml` reads a mcool region and writes standardized `X.npy` / `Y.npy` / `metadata.json`.

- [ ] `chromgp/datasets/hic.py`:
  - `parse_region(s) -> (chrom, start|None, end|None)` — accepts `chr1`, `chr1:1.2M-50M`, etc.
  - `HicLoader.load(mcool_path, resolution, region, balance) -> HicData`
  - `HicData` container holds the contact matrix, bin coordinates, and chromosome metadata.
- [ ] `contact_transform` modes (Y stays in contact space — no distance conversion at preprocess time):
  - `raw` — `Y = contact` straight through.
  - `log1p` — `Y = log1p(contact)`.
  - `obs_over_exp` — `Y = contact / expected(genomic_separation)`. Use `cooltools.expected_cis` to estimate the expected curve, then divide. Cancels the trivial polymer distance-decay so the model focuses on structure beyond it.
- [ ] After transform, flatten the upper triangle of the `(N, N)` contact matrix into a vector of length `N(N-1)/2`. For `num_replicates > 1`, sample noisy replicates (via `chromgp/simulations.py:generate_simulations`) **before flattening** — each replicate gets its own flattened vector.
- [ ] Store `Y.npy` as `(N, D)` where `D = num_replicates` and each column is one replicate's flattened contact vector. Centering is done per-column (`mean(dim=0)`) during training, not here.
- [ ] Store `contact_raw.npy` separately (dense, untransformed, unflattened) for analyze-stage correlation metrics.
- [ ] **Group resolver** (`groups_by`):
  - `chromosome` — used with `regions:` (multi-region). Concatenate `X` / `Y` blocks across regions, assign each bin its source chromosome's index. `C` has one entry per stacked row.
  - `chromhmm_state` — used with single `region:`. Load the ChromHMM BED, intersect each Hi-C bin with `bioframe.overlap`, assign the dominant state per bin. Filter to `chromhmm_states:` whitelist if given.
- [ ] **ChromHMM loader** in `chromgp/datasets/hic.py` (or `chromgp/datasets/chromhmm.py` if it grows): parses 4–9 column BED, returns a `bioframe`-compatible DataFrame.
- [ ] `chromgp/commands/preprocess.py` — orchestrates load + transform + save into `outputs/<dataset>/<region_slug>/preprocessed/`.
- [ ] Verify against the existing `ChromGP SVGP Cooler.ipynb` — row-count and value-range sanity checks should match.

---

### Stage 3: Train command ⬜

**Goal:** `chromgp train -c …/svgp.yaml` fits the GP, saves `model.pth` + `elbo_history.csv`.

- [ ] `chromgp/commands/train.py`:
  - Loads `Config` and the saved `preprocessed/` arrays.
  - Builds the GPzoo backbone from `(prior, groups, integrated_force)`:
    - `SVGP` → `gpzoo.gp.SVGP` or `WSVGP`
    - `MGGP_SVGP` → `gpzoo.gp.MGGP_WSVGP`
    - `LCGP` → `gpzoo.gp.WVNNGP` (current ChromGP nearest analog — confirm during Stage 3a)
    - `MGGP_LCGP` → TBD (may not exist in GPzoo yet — flag as gap)
  - Wraps in `ChromGP` (or `IntegratedForceGP` if `model.integrated_force: true`).
  - Calls `chromgp.utilities.train` / `train_batched` based on `training.batch_size`.
  - Saves checkpoint + ELBO CSV.
- [ ] **Stage 3a (GPzoo API audit + axis refactor):** Two checks before declaring Stage 3 done:
  1. **GPzoo API:** Existing notebooks were written against an older GPzoo. Run a smoke train on synthetic data (helix from `simulations.make_helix`) using the SVGP path and confirm it still converges. If GPzoo API drifted, fix `chromgp/models.py` and `chromgp/utilities.py` to match the current API.
  2. **Transpose to `(N, D)` convention:** ChromGP's existing code stores `Y` as `(D, N)` and centers on `dim=1`. Refactor `chromgp/utilities.py:train` / `train_batched` to expect `(N, D)` with centering on `dim=0`. Also reshape Hi-C preprocess output to store columns-as-replicates (one column = one contact vector) rather than rows-as-replicates. Note: this is not about matching SF (SF is Poisson, no centering); it's about consistent axis semantics across ChromGP's own Gaussian-model variants.
- [ ] Optional `--video` flag for trajectory MP4 (reuse `simulations.create_animation`).
- [ ] Optional `--resume` flag mirroring SF's behavior.

---

### Stage 4: Analyze command ⬜

**Goal:** Extract the inferred 3D coordinates from the saved model and compute reconstruction quality. Train saves `model.pth`; analyze is where `coords_3d.npy` is *produced* from it.

- [ ] `chromgp/analysis.py`:
  - `extract_coords(model, X) -> (N, 3)` — reload model, take `qZ.mean.T` (or run `process_F` for `IntegratedForceGP`), return ordered 3D positions.
  - `predicted_distance(coords) -> (N, N)` — Euclidean pairwise distances on the inferred coords.
  - `contact_correlation(D_pred, contact_obs)` — Pearson / Spearman vs. the original (untransformed) contact matrix from preprocess.
  - `genome_distance_decay(D_pred, X)` — predicted distance vs. genomic separation curve.
- [ ] `chromgp/commands/analyze.py`:
  - Loads model + preprocessed (need both the transformed `Y.npy` and the raw contact for correlation; persist both at preprocess time).
  - Writes `coords_3d.npy` (the headline output), `predicted_distance.npy`, `metrics.json`.

---

### Stage 5: Figures command ⬜

**Goal:** Publication-ready PNGs / MP4s under `outputs/.../figures/`.

- [ ] `chromgp/commands/figures.py`:
  - `elbo_curve.png` — from `elbo_history.csv`.
  - `structure_3d.html` (plotly) and `structure_3d.png` (matplotlib).
  - `contact_recon_vs_true.png` — side-by-side heatmaps (mirrors `simulations.create_animation`'s middle/right panels).
  - `distance_decay.png` — log-log distance vs. genomic separation, expected ~−1 slope.
  - Optional `trajectory.mp4` if training was run with `--video`.

---

## Migration notes

- Existing `notebooks/` stay as exploration scratchpads. Don't delete; some are referenced for parameter intuitions.
- The SVGP notebooks predate recent GPzoo changes and may not run end-to-end against the current API. **Don't fix them ad-hoc** — Stage 3a will surface any breakage and the fix lives in `chromgp/models.py` / `chromgp/utilities.py`, not in the notebook copies.
- The `chromgp/utilities.py:train` ELBO contract (`y - y.mean(dim=1)` row centering, `(pY, qZ, qU, pU)` forward signature) is load-bearing for both notebooks and the new CLI. Preserve it.

---

## Future stages (not in this plan)

- **ChIP-seq dataset:** `chromgp/datasets/chipseq.py` ingesting BigWig / bedGraph per histone mark, aligned to the same binning as a paired Hi-C config. Adds a `data_type: chipseq` branch in preprocess that emits `Y` with `rows = marks`, `cols = bins`. Reuses the same `(X, Y, C)` shape contract and the same training loop — see *Dataset conventions* above. ChromHMM upstream: the BED files used by `groups_by: chromhmm_state` are typically derived from a ChIP-seq panel via ChromHMM, so once we ingest ChIP-seq directly we may wire ChromHMM as a generated artifact rather than an external file.
- **Multi-cell-type / multi-condition MGGP:** different group sources — e.g. one mcool per cell type with cell type as the group key, or paired ChromHMM segmentations across cell types.
- **Multi-resolution training:** coarse-to-fine schedule across resolutions in a single config.
- **Runner / job orchestration:** port SF's `runner.py` for parallel training across model variants on multi-GPU nodes.

---

## Code references (to copy patterns from)

| Pattern | Source |
|---|---|
| Click CLI structure | `Spatial-Factorization/spatial_factorization/cli.py` |
| Config dataclass + YAML | `Spatial-Factorization/spatial_factorization/config.py` |
| Generate (general → variants) | `Spatial-Factorization/spatial_factorization/generate.py` |
| Field-filter tables | same file, `COMMON_MODEL_FIELDS` etc. |
| Per-stage command module | `Spatial-Factorization/spatial_factorization/commands/*.py` |
| Dataset loader pattern | `Spatial-Factorization/spatial_factorization/datasets/slideseq.py` |
| 5-stage planning doc style | `Spatial-Factorization/SLIDESEQ_PNMF_PLANNING.md` (commit `061b914`) |
