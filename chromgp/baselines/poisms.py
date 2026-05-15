"""PoisMS baseline — drives the official R package (Tuzhilina, Hastie & Segal 2022).

We do NOT reimplement PoisMS. The published R package is the canonical
implementation; we call it via ``Rscript`` from a sibling conda env that has
the package installed. This keeps the head-to-head comparison faithful to
the original method.

Pipeline:
1. Pull raw integer Hi-C counts for the config's region via
   :func:`chromgp.baselines.common.load_raw_counts`.
2. Write the counts to a temporary CSV.
3. Invoke ``poisms_fit.R`` through ``Rscript`` (in the ``r-poisms`` conda env).
4. Read back the (N, 3) coordinate matrix.
5. Apply the same probe-footprint FISH validation used by analyze.py.
6. Save outputs to ``outputs/<dataset>/<region>/poisms/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..config import Config
from .common import load_raw_counts, fish_evaluate_positions, save_baseline_outputs


R_ENV_NAME = os.environ.get("CHROMGP_POISMS_R_ENV", "r-poisms")
R_FIT_SCRIPT = Path(__file__).with_name("poisms_fit.R")


def _resolve_rscript() -> str:
    """Locate Rscript inside the ``r-poisms`` conda env."""
    base = os.environ.get("CHROMGP_R_BIN")
    if base:
        return base
    candidates = [
        f"/gladstone/engelhardt/home/lchumpitaz/miniconda3/envs/{R_ENV_NAME}/bin/Rscript",
        f"/gladstone/engelhardt/lab/lchumpitaz/miniconda3/envs/{R_ENV_NAME}/bin/Rscript",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError(
        f"Could not find Rscript in conda env {R_ENV_NAME!r}; "
        "set $CHROMGP_R_BIN to override."
    )


def fit_poisms(counts: np.ndarray, df: int = 5, maxepoch: int = 100) -> dict:
    """Fit PoisMS on a square N×N integer count matrix.

    Args:
        counts: ``(N, N)`` integer Hi-C counts (will be symmetrized).
        df: B-spline degrees of freedom (PoisMS paper default = 5).
        maxepoch: outer-loop epoch cap.

    Returns:
        dict with ``positions`` (N, 3 ndarray) and the R sidecar metadata
        (``beta``, ``loss``, ``epoch``, ``elapsed_s``).
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.shape[0] != counts.shape[1]:
        raise ValueError(f"counts must be square, got {counts.shape}")

    rscript = _resolve_rscript()
    if not R_FIT_SCRIPT.exists():
        raise FileNotFoundError(f"R driver not found: {R_FIT_SCRIPT}")

    with tempfile.TemporaryDirectory() as td:
        counts_csv = Path(td) / "counts.csv"
        out_csv = Path(td) / "X.csv"
        # Write as plain CSV — R's read.csv handles large matrices fine.
        np.savetxt(counts_csv, counts, fmt="%d", delimiter=",")

        cmd = [rscript, str(R_FIT_SCRIPT), str(counts_csv), str(out_csv),
               str(df), str(maxepoch)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"PoisMS R script failed (exit {proc.returncode})\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        print(proc.stdout.strip())

        positions = np.loadtxt(out_csv, delimiter=",")
        sidecar_path = Path(str(out_csv) + ".json")
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}

    if positions.ndim != 2 or positions.shape != (counts.shape[0], 3):
        raise RuntimeError(
            f"unexpected PoisMS output shape {positions.shape}; "
            f"expected ({counts.shape[0]}, 3)"
        )

    return {"positions": positions, **sidecar}


def run(config_path: str, df: int = 5, maxepoch: int = 100) -> None:
    """Fit PoisMS on the Hi-C matrix specified by a ChromGP config."""
    cfg = Config.from_yaml(config_path)
    print(f"== PoisMS (R) on {cfg.dataset} / {cfg.preprocessing.get('region')} ==")

    counts, bin_midpoints_bp, _ = load_raw_counts(cfg)
    print(f"  counts: shape={counts.shape}, total={counts.sum():,}, "
          f"max={counts.max():,}")

    result = fit_poisms(counts, df=df, maxepoch=maxepoch)
    print(f"  PoisMS done: beta={result.get('beta'):.3f}, "
          f"epoch={result.get('epoch')}, elapsed={result.get('elapsed_s'):.1f}s")

    fish_eval = fish_evaluate_positions(cfg, result["positions"], bin_midpoints_bp)
    extra = {
        "beta": result.get("beta"),
        "loss": result.get("loss"),
        "epoch": result.get("epoch"),
        "iter_total": result.get("iter_total"),
        "df": result.get("df", df),
        "maxepoch": result.get("maxepoch", maxepoch),
        "elapsed_s": result.get("elapsed_s"),
        "implementation": "official PoisMS R package (Tuzhilina et al. 2022)",
    }
    out_dir = save_baseline_outputs(cfg, "poisms", result["positions"], fish_eval, extra)

    if fish_eval and fish_eval["metrics"]:
        m = fish_eval["metrics"]
        print(f"  FISH: {m['n_probes_used']} probes, "
              f"Spearman = {m['pairwise_spearman']:+.4f}, "
              f"log-Pearson = {m['log_pairwise_pearson']:+.4f}, "
              f"RMSD = {m['procrustes_rmsd_unitscaled']:.4f}")
    print(f"  Saved: {out_dir}")
