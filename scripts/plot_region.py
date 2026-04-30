#!/usr/bin/env python
"""Plot Hi-C contact map + Polycomb coverage + ChromHMM group annotations for a region.

Usage:
    conda activate chromgp
    python scripts/plot_region.py [region] [--output PATH]

Default region: chr9:85,731,459-93,165,754
"""

import argparse
import json
import sys
from pathlib import Path

import bioframe as bf
import cooler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mp
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

COOLER_FILE = "/gladstone/engelhardt/lab/lchumpitaz/hi-c/mcools/4DNFIXP4QG5B.mcool"
RESOLUTION = 25000
PREPROCESSED = REPO / "outputs/4DNFIXP4QG5B/chr9/preprocessed"
DEFAULT_REGION = "chr9:85,731,459-93,165,754"

# From chromgp/commands/figures.py
_COARSE_GROUP_COLORS = {
    "Active":          "#e41a1c",
    "Transcribed":     "#4daf4a",
    "Heterochromatin": "#984ea3",
    "Polycomb":        "#377eb8",
    "Quiescent":       "#999999",
}

# ChromHMM fine → coarse merge
_CHROMHMM_MERGE = {
    # ENCODE 15-state
    'Tss': 'Active', 'TssFlnk': 'Active', 'TssFlnkD': 'Active',
    'TssFlnkU': 'Active', 'Enh1': 'Active', 'Enh2': 'Active',
    'EnhG1': 'Active', 'EnhG2': 'Active', 'TssA': 'Active',
    'EnhA1': 'Active', 'EnhA2': 'Active', 'EnhWk': 'Active',
    # Roadmap E116 15-state (numbered prefixes)
    '1_TssA': 'Active', '2_TssAFlnk': 'Active',
    '6_EnhG': 'Active', '7_Enh': 'Active',
    # Transcribed (ENCODE)
    'Tx': 'Transcribed', 'TxWk': 'Transcribed',
    # Transcribed (Roadmap E116)
    '3_TxFlnk': 'Transcribed', '4_Tx': 'Transcribed', '5_TxWk': 'Transcribed',
    # Heterochromatin (ENCODE)
    'Het': 'Heterochromatin', 'ZNF/Rpts': 'Heterochromatin',
    # Heterochromatin (Roadmap E116)
    '8_ZNF/Rpts': 'Heterochromatin', '9_Het': 'Heterochromatin',
    # Polycomb (ENCODE)
    'ReprPC': 'Polycomb', 'ReprPCWk': 'Polycomb', 'Biv': 'Polycomb',
    'TssBiv': 'Polycomb', 'EnhBiv': 'Polycomb',
    # Polycomb (Roadmap E116)
    '10_TssBiv': 'Polycomb', '11_BivFlnk': 'Polycomb',
    '12_EnhBiv': 'Polycomb', '13_ReprPC': 'Polycomb', '14_ReprPCWk': 'Polycomb',
    # Quiescent
    'Quies': 'Quiescent', '15_Quies': 'Quiescent',
}

GROUP_ORDER = ['Active', 'Transcribed', 'Heterochromatin', 'Polycomb', 'Quiescent']


def load_chromhmm(path):
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['chrom', 'start', 'end', 'state', 'score',
                            'strand', 'thickStart', 'thickEnd', 'itemRgb'])
    df['state'] = df['state'].map(_CHROMHMM_MERGE).fillna('Quiescent')
    return df[['chrom', 'start', 'end', 'state']]


def polycomb_coverage(bins, chromhmm):
    """Fraction of each bin covered by Polycomb ChromHMM segments."""
    bins = bins.copy()
    bins['bin_idx'] = np.arange(len(bins))
    pc = chromhmm[chromhmm['state'] == 'Polycomb']

    overlaps = bf.overlap(bins, pc, how='left', return_overlap=True,
                          cols1=['chrom', 'start', 'end'],
                          cols2=['chrom', 'start', 'end'])
    overlaps = overlaps.dropna(subset=['chrom_'])
    overlaps['overlap_bp'] = overlaps['overlap_end'] - overlaps['overlap_start']

    cov = overlaps.groupby('bin_idx')['overlap_bp'].sum()
    bin_width = bins['end'].values - bins['start'].values
    result = np.zeros(len(bins))
    result[cov.index.values.astype(int)] = cov.values / bin_width[cov.index.values.astype(int)]
    return result


def group_track_matrix(region_bins, region_C, group_names, valid_mask_region):
    """Build an RGB image strip (1 x N x 3) for the group annotations."""
    n = len(region_bins)
    rgb = np.ones((1, n, 3))  # white background for invalid/gap bins
    color_list = [_COARSE_GROUP_COLORS[g] for g in group_names]

    for i in range(n):
        if valid_mask_region[i] and region_C[i] >= 0:
            c_hex = color_list[region_C[i]]
            rgb[0, i] = mp.colors.to_rgb(c_hex)

    return rgb


def main():
    parser = argparse.ArgumentParser(description="Region plot: Hi-C + Polycomb + groups")
    parser.add_argument("region", nargs="?", default=DEFAULT_REGION,
                        help="Genomic region (default: chr9:85,731,459-93,165,754)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: auto-generated under outputs/)")
    args = parser.parse_args()

    region = args.region
    chrm, start, end = bf.parse_region(region)
    region_str = region.replace(":", "_").replace(",", "-")

    # ---- Load data ----
    hic = cooler.Cooler(f"{COOLER_FILE}::resolutions/{RESOLUTION}")
    all_bins = hic.bins()[:]

    C = np.load(PREPROCESSED / "C.npy")
    valid_mask = np.load(PREPROCESSED / "valid_mask.npy")
    meta = json.load(open(PREPROCESSED / "metadata.json"))
    group_names = meta["group_names"]

    chromhmm = load_chromhmm(meta["preprocessing"]["chromhmm_bed"])
    chromhmm_chr = chromhmm[chromhmm['chrom'] == chrm].copy()

    # ---- Bins for the region ----
    chr_bins = all_bins[all_bins['chrom'] == chrm].copy()
    chr_bins['chr_idx'] = np.arange(len(chr_bins))

    region_bins = bf.select(chr_bins, region).copy()

    if len(region_bins) == 0:
        print(f"No bins found in region {region}", file=sys.stderr)
        sys.exit(1)

    # Map valid_mask and C to region bins
    chr_indices = region_bins['chr_idx'].values  # indices into chr_bins
    region_valid = valid_mask[chr_indices]
    c_map = np.full(len(chr_bins), -1, dtype=int)
    c_map[valid_mask] = np.arange(len(C))
    c_indices = c_map[chr_indices]  # -1 where invalid
    region_C = np.full(len(region_bins), -1, dtype=int)
    valid_region = c_indices >= 0
    region_C[valid_region] = C[c_indices[valid_region]]

    # ---- Hi-C matrix ----
    mat = hic.matrix(balance=True).fetch(region, region)
    res = hic.binsize

    # ---- Polycomb coverage ----
    pc_cov = polycomb_coverage(region_bins, chromhmm_chr)

    # ---- Plot ----
    mp.rcParams['font.size'] = 9
    height, width = 9, 7
    fig = plt.figure(1, figsize=(width, height))
    gs = gridspec.GridSpec(height, width, figure=fig, wspace=2, hspace=0.2)

    # Panel 0 — Hi-C contact map (top 7/9)
    ax0 = plt.subplot(gs[:7, :])
    mat_masked = np.ma.masked_invalid(mat)
    mat_log = np.ma.log10(mat_masked + 5e-6)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('0.12')
    im = ax0.matshow(mat_log, cmap=cmap, aspect='auto', interpolation='none')

    ticks = ax0.get_xticks()
    # Clamp ticks to valid bin range
    ticks = ticks[(ticks >= 0) & (ticks < len(region_bins))]
    ticklabels = ((ticks * res / 1e3) + start / 1e3).astype(int)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(ticklabels)
    ax0.set_yticks(ticks)
    ax0.yaxis.set_ticks_position('left')
    ax0.set_yticklabels(ticklabels)
    ax0.set_title(f'{chrm}:{start:,}-{end:,}  ({len(region_bins)} bins)', pad=8)

    # Panel 1 — Polycomb coverage (1 row)
    ax1 = plt.subplot(gs[7:8, :])
    ax1.plot(region_bins['start'].values, pc_cov, color='#377eb8', lw=0.8)
    ax1.fill_between(region_bins['start'].values, 0, pc_cov,
                     color='#377eb8', alpha=0.3)
    ax1.margins(0)
    ax1.set_ylabel('Polycomb\ncoverage', rotation=90)
    ax1.get_xaxis().set_visible(False)
    ax1.yaxis.tick_right()
    ax1.set_ylim(0, 1.1)

    # Panel 2 — Group annotations (1 row)
    ax2 = plt.subplot(gs[8:9, :])
    track_rgb = group_track_matrix(region_bins, region_C, group_names, region_valid)
    ax2.matshow(track_rgb, aspect='auto')
    ax2.set_ylabel('ChromGP\ngroups', rotation=90)
    ax2.yaxis.tick_right()
    ax2.get_yaxis().set_ticks([])
    ax2.set_xlabel(f'Position along {chrm} (25 Kb)')
    ax2.get_xaxis().set_ticks([])

    # ---- Legend for groups ----
    legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=_COARSE_GROUP_COLORS[g],
                   markersize=10, label=g)
        for g in GROUP_ORDER if g in group_names
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    # ---- Save ----
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = REPO / "outputs/4DNFIXP4QG5B/chr9/figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"region_{region_str}.png"

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
