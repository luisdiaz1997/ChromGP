#!/usr/bin/env python3
"""Generate specificity_classes.png for dissertation Figure ch5-specificity-classes.

Computes per-bin structural specificity s_i (eq. 5.specificity) from
GM12878 MGGP chr19 group-conditional 3D positions and produces a
2-panel figure: specificity track along chr19 + ChromHMM enrichment per class.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Configuration ──────────────────────────────────────────────────
MODEL_DIR = Path("outputs/4DNFIXP4QG5B/chr19/mggp_svgp")
GW_DIR = MODEL_DIR / "groupwise_positions"
PREPROC_DIR = Path("outputs/4DNFIXP4QG5B/chr19/preprocessed")
OUT_PATH = Path(__file__).resolve().parent / "specificity_classes.png"

THRESH_SPECIFIC = 0.7
THRESH_UNIVERSAL = 0.1

# ChromHMM colors (matching ChromGP figures convention)
STATE_COLORS = {
    "Active":          "#E41A1C",
    "Transcribed":     "#377EB8",
    "Heterochromatin": "#4DAF4A",
    "Polycomb":        "#984EA3",
    "Quiescent":       "#FF7F00",
}


def main():
    # ── Load unconditional and group-conditional positions ─────────
    Z_uncond = np.load(GW_DIR / "unconditional.npy")  # (N, 3)
    group_files = sorted(GW_DIR.glob("group_*.npy"))
    Z_cond = {}
    for gf in group_files:
        g = int(gf.stem.split("_")[1])
        Z_cond[g] = np.load(gf)  # (N, 3)

    N = Z_uncond.shape[0]
    G = len(Z_cond)

    # ── Load group labels and names ─────────────────────────────────
    C = np.load(PREPROC_DIR / "C.npy")
    with open(PREPROC_DIR / "metadata.json") as f:
        meta = json.load(f)
    group_names = meta.get("group_names", [str(i) for i in range(G)])

    # ── Compute per-bin specificity s_i ─────────────────────────────
    # s_i = max_c ||mu_i^(c) - mū_i|| / ||mū_i||
    s = np.zeros(N)
    dominant = np.zeros(N, dtype=int)
    for i in range(N):
        mu_bar = Z_uncond[i]  # (3,)
        norm_bar = np.linalg.norm(mu_bar)
        if norm_bar < 1e-8:
            s[i] = 0.0
            dominant[i] = 0
            continue
        shifts = np.zeros(G)
        for g in range(G):
            shifts[g] = np.linalg.norm(Z_cond[g][i] - mu_bar) / norm_bar
        s[i] = shifts.max()
        dominant[i] = shifts.argmax()

    # ── Classify bins ────────────────────────────────────────────────
    specific_mask = s > THRESH_SPECIFIC
    universal_mask = s < THRESH_UNIVERSAL
    enriched_mask = ~specific_mask & ~universal_mask

    pct_specific = 100 * specific_mask.sum() / N
    pct_enriched = 100 * enriched_mask.sum() / N
    pct_universal = 100 * universal_mask.sum() / N
    print(f"State-specific: {pct_specific:.1f}%")
    print(f"State-enriched: {pct_enriched:.1f}%")
    print(f"Universal:      {pct_universal:.1f}%")

    # ── ChromHMM composition per class ───────────────────────────────
    class_masks = {
        "State-specific": specific_mask,
        "State-enriched": enriched_mask,
        "Universal": universal_mask,
    }
    class_colors = {"State-specific": "#E41A1C", "State-enriched": "#377EB8", "Universal": "#4DAF4A"}

    composition = {}
    for cls_name, mask in class_masks.items():
        counts = {}
        for g in range(G):
            counts[group_names[g]] = (C[mask] == g).sum()
        composition[cls_name] = counts

    # ── Load genomic coordinates ────────────────────────────────────
    X = np.load(PREPROC_DIR / "X.npy")  # (N,) bin midpoints in bp
    chrom_start_mb = X.min() / 1e6
    chrom_end_mb = X.max() / 1e6

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))

    # --- Panel A: specificity track ---
    ax1 = fig.add_axes([0.08, 0.55, 0.88, 0.38])

    x_mb = X / 1e6
    # Color each bin by its class
    colors = np.where(specific_mask, class_colors["State-specific"],
              np.where(universal_mask, class_colors["Universal"],
                       class_colors["State-enriched"]))
    ax1.scatter(x_mb, s, c=colors, s=3, alpha=0.6, edgecolors="none")
    ax1.axhline(y=THRESH_SPECIFIC, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.axhline(y=THRESH_UNIVERSAL, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.text(chrom_end_mb * 0.99, THRESH_SPECIFIC * 1.05, "state-specific", fontsize=9,
             ha="right", va="bottom", color="gray", fontstyle="italic")
    ax1.text(chrom_end_mb * 0.99, THRESH_UNIVERSAL * 0.95, "universal", fontsize=9,
             ha="right", va="top", color="gray", fontstyle="italic")

    ax1.set_xlim(chrom_start_mb, chrom_end_mb)
    ax1.set_ylabel("Specificity  $s_i$", fontsize=12)
    ax1.set_title("Per-bin structural specificity along GM12878 chr19", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Genomic position (Mb)", fontsize=11)

    # Legend for classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=class_colors["State-specific"], label=f"State-specific ({pct_specific:.0f}%)"),
        Patch(facecolor=class_colors["State-enriched"], label=f"State-enriched ({pct_enriched:.0f}%)"),
        Patch(facecolor=class_colors["Universal"], label=f"Universal ({pct_universal:.0f}%)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    # --- Panel B: ChromHMM composition per class ---
    bar_names = list(class_masks.keys())
    bar_groups = list(group_names)
    n_classes = len(bar_names)
    n_groups = len(bar_groups)
    x_pos = np.arange(n_classes)
    bar_width = 0.7

    ax2 = fig.add_axes([0.08, 0.08, 0.88, 0.35])

    # Stacked bar chart
    bottom = np.zeros(n_classes)
    for g in range(n_groups):
        vals = [composition[cls][bar_groups[g]] / composition[cls].get(bar_groups[g], 1) * 100
                for cls in bar_names]
        # Normalize to percentages
        totals = [sum(composition[cls].values()) for cls in bar_names]
        vals_pct = [100 * composition[cls][bar_groups[g]] / max(t, 1) for cls, t in zip(bar_names, totals)]
        color = STATE_COLORS.get(bar_groups[g], "#999999")
        ax2.bar(x_pos, vals_pct, bar_width, bottom=bottom, color=color,
                label=bar_groups[g], edgecolor="white", linewidth=0.5)
        bottom += vals_pct

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_names, fontsize=11)
    ax2.set_ylabel("Proportion of bins (%)", fontsize=11)
    ax2.set_title("ChromHMM-state composition by specificity class", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax2.set_ylim(0, 105)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    from PIL import Image as PILImage
    PILImage.open(OUT_PATH).convert("RGB").save(OUT_PATH)
    print(f"\nSaved: {OUT_PATH.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
