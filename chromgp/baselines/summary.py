"""Build a comparison table across methods (ChromGP, PoisMS, …) × chromosomes.

Reads ``analysis.json`` for each (region, method) combination from
``outputs/<dataset>/<region>/<method>/`` and emits a pandas DataFrame plus
a Markdown / LaTeX-ready text dump.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


_METHOD_PRETTY = {
    "svgp": "ChromGP (SVGP)",
    "mggp_svgp": "ChromGP (MGGP-SVGP)",
    "poisms": "PoisMS (R)",
    "pastis": "Pastis",
}


def collect_fish_results(dataset_dir: str | Path,
                          regions: list[str],
                          methods: list[str]) -> pd.DataFrame:
    """Walk dataset_dir/<region>/<method>/analysis.json for FISH metrics.

    Returns one row per (region, method), columns:
        region, method, n_probes_used, n_pairs_used,
        pairwise_spearman, log_pairwise_pearson, procrustes_rmsd_unitscaled.
    """
    rows = []
    dataset_dir = Path(dataset_dir)
    for region in regions:
        for method in methods:
            ajson = dataset_dir / region / method / "analysis.json"
            if not ajson.exists():
                rows.append({"region": region, "method": method,
                             "missing": True})
                continue
            with open(ajson) as f:
                meta = json.load(f)
            fv = meta.get("fish_validation") or {}
            rows.append({
                "region": region,
                "method": method,
                "n_probes_used": fv.get("n_probes_used"),
                "n_pairs_used": fv.get("n_pairs_used"),
                "pairwise_spearman": fv.get("pairwise_spearman"),
                "log_pairwise_pearson": fv.get("log_pairwise_pearson"),
                "procrustes_rmsd_unitscaled": fv.get("procrustes_rmsd_unitscaled"),
                "median_bins_per_probe": fv.get("median_bins_per_probe"),
                "missing": False,
            })
    return pd.DataFrame(rows)


def pivot_spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Wide form: rows = chromosomes, columns = methods, values = Spearman."""
    p = df.pivot(index="region", columns="method", values="pairwise_spearman")
    p.columns = [_METHOD_PRETTY.get(c, c) for c in p.columns]
    return p


def to_markdown_table(df: pd.DataFrame, value_fmt: str = "{:+.3f}") -> str:
    """Render a pivoted table as Markdown."""
    def fmt(x):
        return value_fmt.format(x) if pd.notna(x) else "—"
    cols = list(df.columns)
    header = "| " + " | ".join(["Chromosome"] + cols) + " |"
    sep = "|" + "|".join(["---:"] * (len(cols) + 1)) + "|"
    rows = []
    for region, row in df.iterrows():
        rows.append("| " + " | ".join([region] + [fmt(row[c]) for c in cols]) + " |")
    return "\n".join([header, sep] + rows)


def render_summary(dataset_dir: str | Path,
                   regions: list[str],
                   methods: list[str]) -> dict:
    """One-shot: collect + pivot + print + return a dict of artifacts."""
    long_df = collect_fish_results(dataset_dir, regions, methods)
    spearman = pivot_spearman(long_df.dropna(subset=["pairwise_spearman"]))
    pearson = long_df.pivot(index="region", columns="method", values="log_pairwise_pearson")
    pearson.columns = [_METHOD_PRETTY.get(c, c) for c in pearson.columns]
    rmsd = long_df.pivot(index="region", columns="method", values="procrustes_rmsd_unitscaled")
    rmsd.columns = [_METHOD_PRETTY.get(c, c) for c in rmsd.columns]

    print("\n=== FISH pairwise Spearman ρ ===")
    print(to_markdown_table(spearman))
    print("\n=== FISH log-distance Pearson r ===")
    print(to_markdown_table(pearson))
    print("\n=== Procrustes RMSD (unit-scaled, aux) ===")
    print(to_markdown_table(rmsd))

    return {"long": long_df, "spearman": spearman,
            "log_pearson": pearson, "rmsd": rmsd}
