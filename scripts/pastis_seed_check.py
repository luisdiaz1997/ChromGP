"""One-off: re-run Pastis chr20 25 kb with seeds 1 and 2 for stability check.

Seed=0 (the default-run result, +0.602 Spearman, 30 min wall) is suspected of
being an outlier init. Writes each run to outputs/.../chr20/pastis_seed<k>/
without disturbing the canonical outputs/.../chr20/pastis/ artifacts.
"""
from __future__ import annotations

import sys

from chromgp.config import Config
from chromgp.baselines.common import (
    load_raw_counts, fish_evaluate_positions, save_baseline_outputs,
)
from chromgp.baselines.pastis import fit_pastis


def main() -> int:
    cfg = Config.from_yaml("configs/4DNFIJTOIGOI/chr20/svgp.yaml")
    print(f"== Pastis seed-sensitivity on {cfg.dataset} / chr20 25 kb ==")

    counts, bin_midpoints_bp, _ = load_raw_counts(cfg)
    print(f"  counts: shape={counts.shape}, total={counts.sum():,}, max={counts.max():,}")

    for seed in (1, 2):
        print(f"\n--- seed={seed} ---")
        result = fit_pastis(counts, alpha=-3.0, max_iter=100, seed=seed)
        a, b = result["alpha"], result["beta"]
        print(f"  Pastis done: alpha={a:.3f}, beta={b:.3f}, "
              f"obj={result['obj']:.3f}, converged={result['converged']}, "
              f"elapsed={result['elapsed_s']:.1f}s")

        fish_eval = fish_evaluate_positions(cfg, result["positions"], bin_midpoints_bp)
        extra = {
            "alpha": result["alpha"], "beta": result["beta"], "obj": result["obj"],
            "converged": result["converged"], "seed": result["seed"],
            "max_iter": result["max_iter"],
            "filter_threshold": result["filter_threshold"],
            "normalize": result["normalize"],
            "elapsed_s": result["elapsed_s"],
            "implementation": result["implementation"],
        }
        out_dir = save_baseline_outputs(
            cfg, f"pastis_seed{seed}", result["positions"], fish_eval, extra,
        )
        if fish_eval and fish_eval["metrics"]:
            m = fish_eval["metrics"]
            print(f"  FISH: {m['n_probes_used']} probes, "
                  f"Spearman = {m['pairwise_spearman']:+.4f}, "
                  f"log-Pearson = {m['log_pairwise_pearson']:+.4f}, "
                  f"RMSD = {m['procrustes_rmsd_unitscaled']:.4f}")
        print(f"  Saved: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
