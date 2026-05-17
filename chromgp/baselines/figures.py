"""Comparison figures across methods × chromosomes.

Currently provides:
- :func:`plot_method_bar`: grouped bar of Spearman by chromosome × method.
- :func:`plot_method_scatter_grid`: per-chromosome FISH-vs-predicted scatter
  with one column per method.

All publication outputs are written as paired PNG (300 dpi raster) + PDF
(vector); pass any ``output_path`` ending in ``.png`` and the ``.pdf``
sidecar is created alongside.
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
from scipy.stats import spearmanr


# Publication font defaults — applied at import so every figure picks them up.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,   # embed TrueType in PDF (editable text in Illustrator)
    "ps.fonttype": 42,
})


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

# Canonical left-to-right ordering for figures + tables. Methods not listed
# here are appended in the order the caller supplied.
_METHOD_ORDER = ["pastis", "poisms", "svgp", "mggp_svgp", "hsa"]


def _canonical_order(methods: list[str]) -> list[str]:
    """Return ``methods`` sorted by the canonical publication order."""
    in_order = [m for m in _METHOD_ORDER if m in methods]
    extra = [m for m in methods if m not in _METHOD_ORDER]
    return in_order + extra


def _save_pub(fig, path: Path, **kwargs) -> None:
    """Save a figure as PNG + PDF (vector) for publication.

    Always writes the requested path. If the path's suffix is ``.png``,
    the matching ``.pdf`` is saved alongside. PNG is flattened to RGB so
    it embeds cleanly in Word/Overleaf; PDF is left as vector.
    """
    path = Path(path)
    kwargs.setdefault("bbox_inches", "tight")
    png_kwargs = {**kwargs, "dpi": 300}

    fig.savefig(path, **png_kwargs)
    if path.suffix.lower() == ".png":
        img = Image.open(path)
        if img.mode == "RGBA":
            img.convert("RGB").save(path)

        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, **kwargs)
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


def _bootstrap_ci(
    dataset_dir: Path, region: str, method: str,
    metric: str = "spearman", n_boot: int = 1000, seed: int = 0,
) -> tuple[float, float] | None:
    """Return (lo, hi) 95% bootstrap CI of ``metric`` over probe pairs.

    Resamples upper-triangle (FISH, predicted) probe-pair distances with
    replacement and recomputes the metric per resample. Returns ``None``
    if the per-pair arrays are missing.
    """
    mdir = Path(dataset_dir) / region / method
    fd = mdir / "fish_distance.npy"
    pdp = mdir / "fish_predicted_distance.npy"
    if not (fd.exists() and pdp.exists()):
        return None
    fish = np.load(fd)
    pred = np.load(pdp)
    P = fish.shape[0]
    iu = np.triu_indices(P, k=1)
    fv = fish[iu]
    pv = pred[iu]
    mask = np.isfinite(fv) & np.isfinite(pv)
    fv = fv[mask]; pv = pv[mask]
    if fv.size < 5:
        return None

    rng = np.random.default_rng(seed)
    n = fv.size
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "spearman":
            samples[b] = spearmanr(fv[idx], pv[idx]).correlation
        elif metric == "log_pearson":
            fl = np.log(np.maximum(fv[idx], 1e-12))
            pl = np.log(np.maximum(pv[idx], 1e-12))
            samples[b] = np.corrcoef(fl, pl)[0, 1]
        else:
            return None
    samples = samples[np.isfinite(samples)]
    if samples.size < 10:
        return None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def plot_method_bar(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "spearman",
    metric_label: str | None = None,
    title: str | None = None,
    value_fmt: str = "{:.2f}",
    bootstrap_ci: bool = False,
    dataset_dir: Path | None = None,
    ci_n_boot: int = 1000,
) -> None:
    """Grouped bar of one metric across (region × method).

    Args:
        df: long-form DataFrame from :func:`collect_metrics`.
        output_path: target file. ``.png`` will produce ``.png`` + ``.pdf``.
        value_fmt: format string for the on-bar value labels (no leading
            sign by default).
        bootstrap_ci: if True, draw 95% bootstrap-CI whiskers on each bar.
            Requires ``dataset_dir`` to locate the per-pair ``.npy`` files.
        dataset_dir: dataset root (e.g. ``outputs/4DNFIJTOIGOI``).
    """
    regions = sorted(df["region"].unique())
    methods = _canonical_order(list(dict.fromkeys(df["method"].tolist())))

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

        if bootstrap_ci and dataset_dir is not None:
            for r, xi, v in zip(regions, offsets, vals):
                if not np.isfinite(v):
                    continue
                ci = _bootstrap_ci(dataset_dir, r, m, metric=metric, n_boot=ci_n_boot)
                if ci is None:
                    continue
                lo, hi = ci
                yerr = [[max(0.0, v - lo)], [max(0.0, hi - v)]]
                ax.errorbar(xi, v, yerr=yerr, fmt="none",
                            color="black", capsize=3, linewidth=1)

        for xi, v in zip(offsets, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.01 * np.sign(v if v != 0 else 1),
                        value_fmt.format(v), ha="center",
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
    _save_pub(fig, output_path)


def plot_method_scatter_grid(
    dataset_dir: Path, regions: list[str], methods: list[str],
    output_path: Path,
) -> None:
    """Grid of FISH-vs-predicted scatter, rows = chromosomes, cols = methods."""
    dataset_dir = Path(dataset_dir)
    methods = _canonical_order(methods)
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
            P = fish.shape[0]
            iu = np.triu_indices(P, k=1)
            fv = fish[iu]; pv = pred[iu]
            mask = np.isfinite(fv) & np.isfinite(pv)
            ax.scatter(fv[mask], pv[mask], s=10, alpha=0.55,
                       color=_METHOD_COLOR.get(method, "steelblue"),
                       edgecolor="none")
            ajson = mdir / "analysis.json"
            rho = None
            if ajson.exists():
                meta = json.loads(ajson.read_text())
                rho = (meta.get("fish_validation") or {}).get("pairwise_spearman")
            label = _METHOD_LABEL.get(method, method)
            ttl = f"{region} · {label}"
            if rho is not None and np.isfinite(rho):
                ttl += f"\nρ = {rho:.3f}"
            ax.set_title(ttl, fontsize=10)
            ax.grid(True, alpha=0.3)
            if i == n_rows - 1:
                ax.set_xlabel("FISH median distance (μm)")
            if j == 0:
                ax.set_ylabel("Predicted distance (a.u.)")
    fig.tight_layout()
    _save_pub(fig, output_path)


def run(dataset_dir: str, regions: str, methods: str, out_dir: str,
        bootstrap_ci: bool = False) -> None:
    """Build bar + scatter-grid comparison figures.

    Args:
        dataset_dir: e.g. ``"outputs/4DNFIJTOIGOI"``.
        regions: comma-separated, e.g. ``"chr20,chr21,chr22"``.
        methods: comma-separated, e.g. ``"svgp,poisms,pastis"``.
        out_dir: where to write the figure pack.
        bootstrap_ci: draw 95% bootstrap-CI whiskers on the Spearman bar.
    """
    regs = [r.strip() for r in regions.split(",") if r.strip()]
    meths = [m.strip() for m in methods.split(",") if m.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = Path(dataset_dir)

    df = collect_metrics(ds, regs, meths)
    plot_method_bar(df, out / "fish_spearman_bar.png", "spearman",
                    "FISH pairwise Spearman ρ",
                    title="FISH-distance Spearman ρ",
                    bootstrap_ci=bootstrap_ci, dataset_dir=ds)
    plot_method_bar(df, out / "fish_log_pearson_bar.png", "log_pearson",
                    "FISH log-distance Pearson r",
                    title="FISH log-distance Pearson r",
                    bootstrap_ci=bootstrap_ci, dataset_dir=ds)
    plot_method_scatter_grid(ds, regs, meths,
                              out / "fish_scatter_grid.png")
    for stem in ("fish_spearman_bar", "fish_log_pearson_bar", "fish_scatter_grid"):
        print(f"  Saved: {out}/{stem}.png  (+ .pdf)")
