"""Pastis baseline — Poisson MDS for Hi-C 3D inference (Varoquaux 2014).

We do NOT reimplement Pastis. Calls the official ``hiclib/pastis`` package
(installed in the ``chromgp`` env from GitHub HEAD) via its Python API.

Pipeline:
1. Pull raw integer Hi-C counts via :func:`chromgp.baselines.common.load_raw_counts`.
2. Call ``pastis.optimization.pastis_algorithms.infer`` with a temp outdir.
3. Read the (N, 3) coordinate matrix that ``infer`` writes to disk.
4. Apply the same probe-footprint FISH validation used by analyze.py.
5. Save outputs to ``outputs/<dataset>/<region>/pastis/``.
"""

from __future__ import annotations

import time
import tempfile
import warnings
from pathlib import Path

import numpy as np

from ..config import Config
from .common import load_raw_counts, fish_evaluate_positions, save_baseline_outputs


def fit_pastis(
    counts: np.ndarray,
    alpha: float = -3.0,
    max_iter: int = 100,
    filter_threshold: float = 0.0,
    normalize: bool = False,
    seed: int = 0,
) -> dict:
    """Fit Pastis Poisson MDS on a square N×N integer count matrix.

    Args:
        counts: ``(N, N)`` integer Hi-C counts.
        alpha: power-law decay exponent (Pastis paper default = -3.0).
        max_iter: L-BFGS-B iteration cap.
        filter_threshold: drop bins with row-sum fraction below this; 0.0
            keeps all bins (consistent with the ``valid_mask`` already
            applied by :func:`load_raw_counts`).
        normalize: run ICE inside Pastis. We feed already-shaped counts and
            consume *raw* integer counts identically to PoisMS, so default False.
        seed: random seed for Pastis init.

    Returns:
        dict with ``positions`` (N, 3 ndarray) plus inference metadata.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pastis.fastio is deprecated.*")
        warnings.filterwarnings("ignore", message=".*API of this module is likely to change.*")
        warnings.filterwarnings("ignore", message=".*scipy.sparse.sputils.*")
        # Pastis 0.5.0 still calls scipy.sparse.sputils.get_index_dtype, which
        # is gone in modern scipy. Re-expose it from the private location so
        # pastis.optimization.counts.SparseCountsMatrix can find it.
        import scipy.sparse.sputils as _sputils
        if not hasattr(_sputils, "get_index_dtype"):
            from scipy.sparse._sputils import get_index_dtype as _gid
            _sputils.get_index_dtype = _gid
        from pastis.optimization.pastis_algorithms import infer

    counts = np.asarray(counts, dtype=float)
    if counts.shape[0] != counts.shape[1]:
        raise ValueError(f"counts must be square, got {counts.shape}")
    N = counts.shape[0]

    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _struct, info = infer(
                counts_raw=counts,
                lengths=np.array([N]),
                ploidy=1,
                outdir=td,
                alpha=alpha,
                max_iter=max_iter,
                filter_threshold=filter_threshold,
                normalize=normalize,
                seed=seed,
                verbose=False,
            )
        elapsed = time.time() - t0

        td_path = Path(td)
        converged_path = td_path / "struct_inferred.000.coords"
        nonconverged_path = td_path / "struct_nonconverged.000.coords"
        if converged_path.exists():
            coord_path = converged_path
        elif nonconverged_path.exists():
            coord_path = nonconverged_path
        else:
            available = sorted(p.name for p in td_path.iterdir())
            raise RuntimeError(
                f"Pastis did not produce a coords file in {td}; got {available}"
            )
        positions = np.loadtxt(coord_path)

    if positions.ndim != 2 or positions.shape != (N, 3):
        raise RuntimeError(
            f"unexpected Pastis output shape {positions.shape}; expected ({N}, 3)"
        )

    beta = info.get("beta")
    if isinstance(beta, (list, tuple)) and len(beta) == 1:
        beta = float(beta[0])

    return {
        "positions": positions,
        "alpha": float(info.get("alpha")) if info.get("alpha") is not None else None,
        "beta": beta,
        "obj": float(info.get("obj")) if info.get("obj") is not None else None,
        "converged": bool(info.get("converged", False)),
        "seed": int(info.get("seed", seed)),
        "max_iter": max_iter,
        "filter_threshold": filter_threshold,
        "normalize": normalize,
        "elapsed_s": float(elapsed),
        "implementation": (
            "Pastis 0.5.0 (Varoquaux 2014; hiclib/pastis@HEAD), "
            "pastis_algorithms.infer with Poisson likelihood"
        ),
    }


def run(config_path: str, alpha: float = -3.0, max_iter: int = 100) -> None:
    """Fit Pastis on the Hi-C matrix specified by a ChromGP config."""
    cfg = Config.from_yaml(config_path)
    print(f"== Pastis on {cfg.dataset} / {cfg.preprocessing.get('region')} ==")

    counts, bin_midpoints_bp, _ = load_raw_counts(cfg)
    print(f"  counts: shape={counts.shape}, total={counts.sum():,}, "
          f"max={counts.max():,}")

    result = fit_pastis(counts, alpha=alpha, max_iter=max_iter)
    a = result.get("alpha")
    b = result.get("beta")
    print(f"  Pastis done: alpha={a:.3f}, beta={b:.3f}, "
          f"obj={result.get('obj'):.3f}, converged={result.get('converged')}, "
          f"elapsed={result.get('elapsed_s'):.1f}s")

    fish_eval = fish_evaluate_positions(cfg, result["positions"], bin_midpoints_bp)
    extra = {
        "alpha": result.get("alpha"),
        "beta": result.get("beta"),
        "obj": result.get("obj"),
        "converged": result.get("converged"),
        "seed": result.get("seed"),
        "max_iter": result.get("max_iter"),
        "filter_threshold": result.get("filter_threshold"),
        "normalize": result.get("normalize"),
        "elapsed_s": result.get("elapsed_s"),
        "implementation": result.get("implementation"),
    }
    out_dir = save_baseline_outputs(cfg, "pastis", result["positions"], fish_eval, extra)

    if fish_eval and fish_eval["metrics"]:
        m = fish_eval["metrics"]
        print(f"  FISH: {m['n_probes_used']} probes, "
              f"Spearman = {m['pairwise_spearman']:+.4f}, "
              f"log-Pearson = {m['log_pairwise_pearson']:+.4f}, "
              f"RMSD = {m['procrustes_rmsd_unitscaled']:.4f}")
    print(f"  Saved: {out_dir}")
