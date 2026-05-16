"""Shared baseline infrastructure.

Provides:
- :func:`load_raw_counts`: pull integer Hi-C counts from the mcool for the
  same region the ChromGP pipeline used. PoisMS and Pastis both need raw
  counts, but the preprocessed dir stores ICE-balanced floats.
- :func:`fish_evaluate_positions`: apply the same probe-footprint aggregation
  + :func:`chromgp.analysis.fish_validation` used by analyze.py, so the
  comparison table is apples-to-apples.
- :func:`save_baseline_outputs`: write 3D coordinates + FISH artifacts +
  analysis.json to ``outputs/<dataset>/<region>/<method>/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..analysis import fish_validation
from ..config import Config
from ..datasets import load_preprocessed, load_wang2016


def load_raw_counts(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw integer Hi-C counts for the config's region.

    Replicates the bin filtering that the preprocess step applied so the
    returned matrix matches the bin order in the preprocessed dir.

    Returns:
        counts: ``(N, N)`` integer counts (np.int64), bins consistent with
            preprocessed ``X.npy``.
        bin_midpoints_bp: ``(N,)`` int array, bp midpoints of each bin.
        valid_mask: ``(N_full,)`` bool array — the mask preprocess used to
            keep / drop bins. Useful for reproducing the same filtering.
    """
    import cooler  # local import; cooler is heavy

    mcool = config.preprocessing["mcool_path"]
    resolution = int(config.preprocessing["resolution"])
    region = config.preprocessing["region"]

    clr = cooler.Cooler(f"{mcool}::resolutions/{resolution}")
    raw = clr.matrix(balance=False).fetch(region)
    raw = np.asarray(raw, dtype=np.int64)

    region_dir = Path(config.output_dir) / config.region_slug
    data = load_preprocessed(region_dir)
    valid_mask = data.valid_mask.cpu().numpy() if data.valid_mask is not None else np.ones(raw.shape[0], dtype=bool)

    counts = raw[np.ix_(valid_mask, valid_mask)]
    bin_midpoints_bp = data.X.cpu().numpy().astype(np.int64)
    return counts, bin_midpoints_bp, valid_mask


def fish_evaluate_positions(
    config: Config,
    positions: np.ndarray,
    bin_midpoints_bp: np.ndarray,
) -> dict:
    """Apply the same probe-footprint FISH evaluation used by ChromGP.

    Args:
        config: Config with ``preprocessing.fish:`` block.
        positions: ``(N, 3)`` baseline 3D coordinates.
        bin_midpoints_bp: ``(N,)`` bin midpoints in bp (same length as positions).

    Returns:
        dict with the same keys analyze.py writes under ``fish_validation``,
        plus the probe-level arrays needed for downstream figure code.
    """
    fish_cfg = config.preprocessing.get("fish")
    if fish_cfg is None:
        return {}
    fish_path = fish_cfg.get("path") if isinstance(fish_cfg, dict) else fish_cfg
    fish_chrom = fish_cfg.get("chrom") if isinstance(fish_cfg, dict) else None
    resolution = int(config.preprocessing.get("resolution", 25000))

    ref = load_wang2016(fish_path, chrom=fish_chrom)
    bin_lists = ref.bin_indices_per_probe(bin_midpoints_bp, pad_bp=resolution // 2)

    probe_positions = np.full((ref.n_probes, 3), np.nan, dtype=float)
    n_bins_per_probe = np.zeros(ref.n_probes, dtype=int)
    for p, bins in enumerate(bin_lists):
        n_bins_per_probe[p] = len(bins)
        if len(bins) > 0:
            probe_positions[p] = positions[bins].mean(axis=0)

    fish_med = ref.median_distance_matrix()
    metrics = fish_validation(fish_med, probe_positions)

    diff = probe_positions[:, None, :] - probe_positions[None, :, :]
    probe_pred_dist = np.linalg.norm(diff, axis=-1)

    return {
        "metrics": metrics,
        "fish_distance": fish_med,
        "probe_positions": probe_positions,
        "probe_pred_dist": probe_pred_dist,
        "n_bins_per_probe": n_bins_per_probe,
        "ref": ref,
        "median_bins_per_probe": int(np.median(n_bins_per_probe)),
        "mean_bins_per_probe": float(np.mean(n_bins_per_probe)),
    }


def save_baseline_outputs(
    config: Config,
    method: str,
    positions: np.ndarray,
    fish_eval: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> Path:
    """Write baseline outputs to ``outputs/<dataset>/<region>/<method>/``.

    Mirrors the artifacts ChromGP analyze.py writes so figures + comparison
    code can read either with the same key set.

    Args:
        config: Loaded Config.
        method: e.g. ``"poisms"`` or ``"pastis"``.
        positions: ``(N, 3)`` baseline coordinates aligned with the
            preprocessed bin order.
        fish_eval: result of :func:`fish_evaluate_positions`, or None.
        extra_meta: any extra method-specific metadata to record.

    Returns:
        The output directory path.
    """
    region_slug = config.region_slug
    out_dir = Path(config.output_dir) / region_slug / method
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "positions.npy", positions)

    pred = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    np.save(out_dir / "predicted_distance.npy", pred)

    fish_meta = None
    if fish_eval:
        np.save(out_dir / "fish_distance.npy", fish_eval["fish_distance"])
        np.save(out_dir / "fish_probe_positions.npy", fish_eval["probe_positions"])
        np.save(out_dir / "fish_predicted_distance.npy", fish_eval["probe_pred_dist"])
        np.save(out_dir / "fish_bins_per_probe.npy", fish_eval["n_bins_per_probe"])
        np.save(out_dir / "fish_probe_midpoints.npy", fish_eval["ref"].probe_midpoints)
        m = fish_eval["metrics"]
        fish_meta = {
            "source": fish_eval["ref"].source,
            "chrom": fish_eval["ref"].chrom,
            "assembly": fish_eval["ref"].assembly,
            "n_cells": fish_eval["ref"].n_cells,
            "n_probes_total": fish_eval["ref"].n_probes,
            "n_probes_used": m["n_probes_used"],
            "n_pairs_used": m["n_pairs_used"],
            "pairwise_spearman": m["pairwise_spearman"],
            "log_pairwise_pearson": m["log_pairwise_pearson"],
            "procrustes_rmsd_unitscaled": m["procrustes_rmsd_unitscaled"],
            "resolution": int(config.preprocessing.get("resolution", 25000)),
            "median_bins_per_probe": fish_eval["median_bins_per_probe"],
            "mean_bins_per_probe": fish_eval["mean_bins_per_probe"],
        }

    meta = {
        "method": method,
        "dataset": config.dataset,
        "region": config.preprocessing.get("region"),
        "n_bins": positions.shape[0],
        "fish_validation": fish_meta,
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return out_dir
