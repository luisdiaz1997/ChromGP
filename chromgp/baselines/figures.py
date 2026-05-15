"""Comparison figures across methods × chromosomes.

Currently provides:
- :func:`plot_method_bar`: grouped bar of Spearman by chromosome × method.
- :func:`plot_method_scatter_grid`: per-chromosome FISH-vs-predicted scatter
  with one column per method.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


_METHOD_COLOR = {
    "svgp": "#377eb8",         # blue — ChromGP
    "mggp_svgp": "#4daf4a",    # green — ChromGP MGGP
    "poisms": "#e41a1c",       # red — PoisMS
    "pastis": "#ff7f00",       # orange — Pastis
    "hsa": "#984ea3",          # purple — HSA
}

_METHOD_LABEL = {
    "svgp": "ChromGP (SVGP)",
    "mggp_svgp": "ChromGP (MGGP-SVGP)",
    "poisms": "PoisMS",
    "pastis": "Pastis",
    "hsa": "HSA",
}


def _save_rgb(fig, path: Path, **kwargs) -> None:
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)
    img = Image.open(path)
    if img.mode == "RGBA":
        img.convert("RGB").save(path)
    plt.close(fig)


def collect_metrics(dataset_dir: Path, regions: list[str], methods: list[str]) -> pd.DataFrame:
    rows = []
    for region in regions:
        for method in methods:
            ajson = Path(dataset_dir) / region / method / "analysis.json"
            if not ajson.exists():
                continue
            fv = json.loads(ajson.read_text()).get("fish_validation") or {}
            rows.append({
                "region": region,
                "method": method,
                "spearman": fv.get("pairwise_spearman"),
                "log_pearson": fv.get("log_pairwise_pearson"),
                "rmsd": fv.get("procrustes_rmsd_unitscaled"),
                "n_probes": fv.get("n_probes_used"),
            })
    return pd.DataFrame(rows)


def plot_method_bar(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "spearman",
    metric_label: str | None = None,
    title: str | None = None,
) -> None:
    """Grouped bar of one metric across (region × method).

    ``df`` is expected to be the long-form DataFrame from
    :func:`collect_metrics`.
    """
    regions = sorted(df["region"].unique())
    methods = list(dict.fromkeys(df["method"].tolist()))

    n_regions = len(regions)
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(5, 1.5 * n_regions), 4))
    x = np.arange(n_regions)
    for j, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("region")
        vals = [sub.loc[r, metric] if r in sub.index else np.nan for r in regions]
        offsets = x + (j - (n_methods - 1) / 2) * width
        ax.bar(offsets, vals, width=width * 0.9,
               color=_METHOD_COLOR.get(m, f"C{j}"),
               label=_METHOD_LABEL.get(m, m), edgecolor="black", linewidth=0.4)
        for xi, v in zip(offsets, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01 * np.sign(v if v != 0 else 1),
                        f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel(metric_label or metric)
    ax.set_xlabel("Chromosome")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="best", frameon=True)
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    _save_rgb(fig, output_path)


def plot_method_scatter_grid(
    dataset_dir: Path, regions: list[str], methods: list[str],
    output_path: Path,
) -> None:
    """Grid of FISH-vs-predicted scatter, rows = chromosomes, cols = methods."""
    dataset_dir = Path(dataset_dir)
    n_rows = len(regions)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)
    for i, region in enumerate(regions):
        for j, method in enumerate(methods):
            ax = axes[i, j]
            mdir = dataset_dir / region / method
            fd = mdir / "fish_distance.npy"
            pd_ = mdir / "fish_predicted_distance.npy"
            if not (fd.exists() and pd_.exists()):
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            fish = np.load(fd)
            pred = np.load(pd_)
            valid = np.all(np.isfinite(pred), axis=1) if pred.ndim == 2 \
                    else np.ones(pred.shape[0], dtype=bool)
            P = fish.shape[0]
            iu = np.triu_indices(P, k=1)
            fv = fish[iu]; pv = pred[iu]
            mask = np.isfinite(fv) & np.isfinite(pv)
            ax.scatter(fv[mask], pv[mask], s=10, alpha=0.55,
                       color=_METHOD_COLOR.get(method, "steelblue"),
                       edgecolor="none")
            # Read Spearman from analysis.json for title
            ajson = mdir / "analysis.json"
            rho = None
            if ajson.exists():
                meta = json.loads(ajson.read_text())
                rho = (meta.get("fish_validation") or {}).get("pairwise_spearman")
            label = _METHOD_LABEL.get(method, method)
            ttl = f"{region} · {label}"
            if rho is not None and np.isfinite(rho):
                ttl += f"\nρ = {rho:+.3f}"
            ax.set_title(ttl, fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel("FISH median distance (μm)")
            if j == 0:
                ax.set_ylabel("Predicted distance (a.u.)")
    fig.tight_layout()
    _save_rgb(fig, output_path)


def run(dataset_dir: str, regions: str, methods: str, out_dir: str) -> None:
    """Build bar + scatter-grid comparison figures.

    Args:
        dataset_dir: e.g. ``"outputs/4DNFIJTOIGOI"``.
        regions: comma-separated, e.g. ``"chr20,chr21,chr22"``.
        methods: comma-separated, e.g. ``"svgp,poisms"``.
        out_dir: where to write the PNGs.
    """
    regs = [r.strip() for r in regions.split(",") if r.strip()]
    meths = [m.strip() for m in methods.split(",") if m.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = collect_metrics(Path(dataset_dir), regs, meths)
    plot_method_bar(df, out / "fish_spearman_bar.png", "spearman",
                    "FISH pairwise Spearman ρ",
                    title="FISH validation — Spearman ρ by chromosome")
    plot_method_bar(df, out / "fish_log_pearson_bar.png", "log_pearson",
                    "FISH log-distance Pearson r",
                    title="FISH validation — log-distance Pearson r")
    plot_method_scatter_grid(Path(dataset_dir), regs, meths,
                              out / "fish_scatter_grid.png")
    print(f"  Saved: {out}/fish_spearman_bar.png")
    print(f"  Saved: {out}/fish_log_pearson_bar.png")
    print(f"  Saved: {out}/fish_scatter_grid.png")
