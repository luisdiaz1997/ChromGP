"""Generate figures: ELBO curve, training animation, and 3D reconstruction.

Matches SF plotting conventions: log-log ELBO, steelblue line, turbo colormap.
Animation follows the notebook 3-panel layout: 3D structure | recon distances | observed.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
# Force-register 3D projection BEFORE pyplot import (mpl 3.10 compat)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image

from ..config import Config
from ..datasets import load_preprocessed


# ---------------------------------------------------------------------------
# ELBO plot — matches SF's plot_elbo_curve exactly
# ---------------------------------------------------------------------------

def plot_elbo(elbo_history: np.ndarray, output_path: Path) -> None:
    """Plot ELBO training curve on log-log scale (SF convention).

    Args:
        elbo_history: 1D array of loss/ELBO values per iteration.
        output_path: Where to save elbo_curve.png.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = np.arange(1, len(elbo_history) + 1)
    ax.plot(iterations, elbo_history, linewidth=1.5, color="steelblue")

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("ELBO", fontsize=12)
    ax.set_title("Training Convergence (log-log)", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")

    ymin = np.min(elbo_history) * 1.1
    ymax = np.max(elbo_history) * 0.9
    ax.set_ylim(ymin, ymax)

    final_elbo = elbo_history[-1]
    ax.axhline(final_elbo, color="red", linestyle="--", alpha=0.5,
               label=f"Final: {final_elbo:.2e}")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Final 3D reconstruction figure — 3-panel: structure | recon | observed
# ---------------------------------------------------------------------------

# Fixed semantic colors for the 5 coarse ChromHMM groups
_COARSE_GROUP_COLORS: dict[str, str] = {
    "Active":          "#e41a1c",  # red    — promoters/enhancers
    "Transcribed":     "#4daf4a",  # green  — gene bodies
    "Heterochromatin": "#984ea3",  # purple — constitutive heterochromatin
    "Polycomb":        "#377eb8",  # blue   — polycomb repressed
    "Quiescent":       "#999999",  # gray   — inactive
}

# Fallback palette for generic / non-coarse group labels
_STATE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
]


def _get_group_colors(C: np.ndarray, group_names: list | None = None) -> np.ndarray:
    """Map integer group labels to hex color strings.

    Uses semantic colors when group_names are the 5 coarse ChromHMM groups,
    otherwise falls back to the generic _STATE_COLORS palette.
    """
    if group_names is not None:
        colors = [
            _COARSE_GROUP_COLORS.get(name, _STATE_COLORS[i % len(_STATE_COLORS)])
            for i, name in enumerate(group_names)
        ]
    else:
        n_groups = int(C.max()) + 1
        colors = _STATE_COLORS[:n_groups]
    return np.array(colors)[C.astype(int)]


def _expand_to_full(dense: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Expand (N,N) dense matrix to (N_full,N_full) with NaN in filtered positions."""
    N_full = len(valid_mask)
    full = np.full((N_full, N_full), np.nan)
    idx = np.where(valid_mask)[0]
    full[np.ix_(idx, idx)] = dense
    return full


def _expand_groups(C: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """Expand dense (N,) group labels to (N_full,) with -1 for gap bins."""
    if valid_mask is None:
        return C.astype(np.int64)
    N_full = len(valid_mask)
    C_full = np.full(N_full, -1, dtype=np.int64)
    C_full[valid_mask] = C
    return C_full


def _draw_chromhmm_track(ax, C_full: np.ndarray, group_names: list,
                         horizontal: bool = True) -> None:
    """Render ChromHMM state track as a color strip on ax. Gap bins (-1) are white."""
    from matplotlib.colors import to_rgb
    n = len(C_full)
    rgb = np.ones((n, 3), dtype=float)
    valid = C_full >= 0
    if valid.any():
        rgb[valid] = np.array([to_rgb(c) for c in _get_group_colors(C_full[valid], group_names)])
    if horizontal:
        ax.imshow(rgb[np.newaxis, :, :], aspect="auto", interpolation="nearest")
    else:
        ax.imshow(rgb[:, np.newaxis, :], aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _group_boundaries(C_full: np.ndarray) -> np.ndarray:
    """0.5-offset positions in matrix coords where ChromHMM group changes."""
    valid_idx = np.where(C_full >= 0)[0]
    if len(valid_idx) < 2:
        return np.zeros(0, dtype=float)
    change_mask = np.diff(C_full[valid_idx]) != 0
    return (valid_idx[1:][change_mask] - 0.5).astype(float)


def _set_genomic_ticks(ax, n_bins: int, resolution: int, start_bp: int) -> None:
    """Add genomic position tick labels in kb at the bottom x-axis only."""
    ax.xaxis.set_ticks_position("bottom")
    xticks = ax.xaxis.get_ticklocs()
    valid_xt = xticks[(xticks >= 0) & (xticks < n_bins)].astype(int)
    if len(valid_xt):
        ax.set_xticks(valid_xt)
        ax.set_xticklabels(
            ((valid_xt * resolution + start_bp) // 1000),
            fontsize=6, rotation=45, ha="right",
        )
    ax.set_yticks([])


def plot_reconstruction(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        C: np.ndarray | None = None,
                        group_names: list | None = None,
                        valid_mask: np.ndarray | None = None,
                        resolution: int | None = None,
                        start_bp: int = 0,
                        output_path: Path | None = None) -> None:
    """1x3 panel: 3D structure colored by group | reconstructed distances | observed.

    When C is provided, a ChromHMM state track is drawn above and to the left
    of each matrix panel.

    Args:
        Z: Latent 3D positions (N, 3).
        X: Genomic coordinates (N,) — used to order bins along chromosome.
        Y: Observed distance/contact matrix (N, N).
        C: Optional group labels (N,).
        group_names: Optional group name strings.
        valid_mask: Optional (N_full,) bool mask to expand reconstruction with NaN gaps.
        output_path: Where to save the figure.
    """
    tk = 0.25    # track thickness
    cbar_w = 0.3  # dedicated colorbar column width

    if C is not None:
        # cols: [3D | v-tk | recon | cbar | v-tk | obs | cbar]
        # rows: [h-track(thin) | main]
        # Figure sized so each GridSpec unit = 1 inch → matrix panels are
        # exactly ps×ps inches (square) regardless of track/cbar widths.
        ps = 3.5
        w_ratios = [ps, tk, ps, cbar_w, tk, ps, cbar_w]
        h_ratios = [tk, ps]
        fig = plt.figure(figsize=(sum(w_ratios), sum(h_ratios)))
        gs = gridspec.GridSpec(2, 7, figure=fig,
                               width_ratios=w_ratios,
                               height_ratios=h_ratios,
                               wspace=0.02, hspace=0.02)
        ax1    = fig.add_subplot(gs[:, 0], projection="3d")
        ax_ht2 = fig.add_subplot(gs[0, 2])
        ax_vt2 = fig.add_subplot(gs[1, 1])
        ax2    = fig.add_subplot(gs[1, 2])
        ax_cb2 = fig.add_subplot(gs[1, 3])
        ax_ht3 = fig.add_subplot(gs[0, 5])
        ax_vt3 = fig.add_subplot(gs[1, 4])
        ax3    = fig.add_subplot(gs[1, 5])
        ax_cb3 = fig.add_subplot(gs[1, 6])
    else:
        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

    # --- Panel 1: 3D structure ---
    ax1.plot(Z[:, 0], Z[:, 1], Z[:, 2], lw=0.6, color="lightgray", alpha=0.5, zorder=1)
    if C is not None:
        bin_colors = _get_group_colors(C, group_names)
        ax1.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=bin_colors, s=2.0,
                    alpha=0.85, edgecolors="none", zorder=2)
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_COARSE_GROUP_COLORS.get(
                           group_names[g] if group_names else "",
                           _STATE_COLORS[g % len(_STATE_COLORS)]),
                       markersize=5,
                       label=group_names[g] if group_names else f"Group {g}")
            for g in range(int(C.max()) + 1)
        ]
        ax1.legend(handles=legend_elements, loc="upper left", fontsize=5,
                   framealpha=0.7, handlelength=0.8)
    ax1.view_init(elev=20, azim=-100)
    ax1.set_title("3D Chromatin Structure", fontsize=12)

    # --- Panel 2: Reconstructed distances ---
    Z_t = torch.tensor(Z)
    recon_dist = torch.cdist(Z_t, Z_t).numpy()
    if valid_mask is not None:
        recon_dist = _expand_to_full(recon_dist, valid_mask)
    im2 = ax2.matshow(recon_dist, cmap="YlOrRd_r", aspect="auto")
    if C is None:
        ax2.set_title("Reconstructed Distances", fontsize=12)

    # --- Panel 3: Observed data ---
    Y_obs = np.log10(Y + 5e-6)
    im3 = ax3.matshow(Y_obs, cmap="YlOrRd", aspect="auto")
    if C is None:
        ax3.set_title("Observed Contact Matrix", fontsize=12)

    if C is not None:
        fig.colorbar(im2, cax=ax_cb2)
        fig.colorbar(im3, cax=ax_cb3)
    else:
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        plt.colorbar(im3, ax=ax3, fraction=0.046)

    # --- ChromHMM tracks (title placed on h-track axis so it sits above the track) ---
    if C is not None:
        C_full = _expand_groups(C, valid_mask)
        _draw_chromhmm_track(ax_ht2, C_full, group_names, horizontal=True)
        ax_ht2.set_title("Reconstructed Distances", fontsize=9, pad=3)
        _draw_chromhmm_track(ax_vt2, C_full, group_names, horizontal=False)
        _draw_chromhmm_track(ax_ht3, C_full, group_names, horizontal=True)
        ax_ht3.set_title("Observed Contact Matrix", fontsize=9, pad=3)
        _draw_chromhmm_track(ax_vt3, C_full, group_names, horizontal=False)

    # --- Genomic position ticks (plotmap convention: bottom + right) ---
    if resolution is not None:
        n_bins_full = recon_dist.shape[0]
        _set_genomic_ticks(ax2, n_bins_full, resolution, start_bp)
        _set_genomic_ticks(ax3, n_bins_full, resolution, start_bp)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}")
    else:
        return fig


# ---------------------------------------------------------------------------
# Training animation (GIF)
# ---------------------------------------------------------------------------

def create_training_animation(
    Zs: np.ndarray,           # (n_frames, N, 3) full-batch latent positions
    Y: np.ndarray,            # (N, N) observed contact matrix
    X: np.ndarray,            # (N,) genomic coordinates (for coloring)
    C: np.ndarray | None,     # (N,) group labels
    group_names: list | None,
    frame_iters: list,        # iteration numbers for each frame
    output_path: Path,
    valid_mask: np.ndarray | None = None,  # (N_full,) bool to expand recon with NaN gaps
    step: int = 1,            # frame stride
    fps: int = 15,
) -> None:
    """Create a training GIF showing 3D structure evolution + distance matrices.

    Follows the notebook 3-panel layout: 3D structure | reconstructed distances | observed.

    Args:
        Zs: Latent positions over training (n_frames, N, 3).
        Y: Observed contact/distance matrix (N, N).
        X: Genomic coordinates (N,).
        C: Optional group labels (N,).
        group_names: Optional group name strings.
        frame_iters: Iteration number for each frame.
        output_path: Output .gif path.
        valid_mask: Optional (N_full,) bool to expand reconstruction with NaN gaps.
        step: Frame stride (take every step-th frame).
        fps: Frames per second for output.
    """
    n_frames_total, N, D = Zs.shape
    frame_indices = list(range(0, n_frames_total, step))
    n_frames = len(frame_indices)

    s = 3.0 if N < 15000 else 100.0 / np.sqrt(N)

    # Pre-compute group colors once (reused across frames)
    bin_colors_anim = _get_group_colors(C, group_names) if C is not None else None

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Static observed matrix (panel 3)
    Y_obs = np.log10(Y + 5e-6)
    ax3.matshow(Y_obs, cmap="YlOrRd", aspect="auto")
    ax3.set_title("Observed Contact Matrix", fontsize=12)

    # Fixed palette: 253 turbo + gray + white + black (SF convention)
    turbo_colors = (mcm.turbo(np.linspace(0, 1, 253))[:, :3] * 255).astype(np.uint8)
    extras = np.array([[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8)
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(np.vstack([turbo_colors, extras]).flatten().tolist())

    def _render_frame(fi: int) -> np.ndarray:
        """Render a single frame and return RGB array."""
        ax1.cla()
        ax2.cla()

        z = Zs[frame_indices[fi]]
        it = frame_iters[frame_indices[fi]]

        # --- 3D structure ---
        ax1.plot(z[:, 0], z[:, 1], z[:, 2], lw=0.6, color="lightgray", alpha=0.5, zorder=1)
        if bin_colors_anim is not None:
            ax1.scatter(z[:, 0], z[:, 1], z[:, 2], c=bin_colors_anim, s=1.5,
                        alpha=0.8, edgecolors="none", zorder=2)
        else:
            ax1.plot(z[:, 0], z[:, 1], z[:, 2], lw=1.5)
        ax1.view_init(elev=20, azim=-100)
        ax1.set_title(f"3D Structure (iter {it})", fontsize=12)

        # --- Reconstructed distances ---
        Z_t = torch.tensor(z)
        recon = torch.cdist(Z_t, Z_t).numpy()
        if valid_mask is not None:
            recon = _expand_to_full(recon, valid_mask)
        ax2.matshow(recon, cmap="YlOrRd_r", aspect="auto")
        ax2.set_title("Reconstructed Distances", fontsize=12)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return buf[:, :, :3]

    gif_frames = []
    for fi in range(n_frames):
        buf = _render_frame(fi)
        gif_frames.append(Image.fromarray(buf).quantize(palette=palette_img, dither=0))
        if fi % 20 == 0:
            print(f"  {fi + 1}/{n_frames} frames rendered...")

    duration = int(1000 / fps)
    gif_frames[0].save(str(output_path), save_all=True,
                      append_images=gif_frames[1:],
                      loop=0, duration=duration, optimize=False)
    print(f"  Saved: {output_path} ({output_path.stat().st_size // 1024 // 1024}MB)")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Groupwise conditional 3D positions — one panel per ChromHMM group
# ---------------------------------------------------------------------------

def plot_groupwise_coordinates(
    Z_uncond: np.ndarray,
    groupwise_positions: dict,
    C: np.ndarray,
    group_names: list,
    output_path: Path,
    panel_size: float = 3.5,
) -> None:
    """Grid of 3D chromatin structures: unconditional + one per ChromHMM group.

    Each panel shows the full chromosome curve (bins connected in genomic order)
    colored by actual ChromHMM group membership. The unconditional panel uses
    the real group labels; conditional panels show the hypothetical structure
    if all bins had that group's kernel.

    Args:
        Z_uncond: (N, 3) unconditional posterior mean positions.
        groupwise_positions: {g: (N, 3)} conditional posterior per group.
        C: (N,) integer group labels for coloring.
        group_names: List of G group name strings.
        output_path: Where to save the figure.
        panel_size: Size of each subplot panel in inches.
    """
    n_groups = len(groupwise_positions)
    n_panels = 1 + n_groups  # unconditional + one per group

    # Lay out in a roughly square grid
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig = plt.figure(figsize=(panel_size * ncols, panel_size * nrows))

    bin_colors = _get_group_colors(C, group_names)  # (N,) hex strings by actual group

    # Compute shared cubic bounding box across all panels so sizes are comparable
    all_Z = np.vstack([Z_uncond] + list(groupwise_positions.values()))
    centers = (all_Z.max(axis=0) + all_Z.min(axis=0)) / 2
    half_span = (all_Z.max(axis=0) - all_Z.min(axis=0)).max() / 2 * 1.05
    lims = [(centers[i] - half_span, centers[i] + half_span) for i in range(3)]

    def _draw_panel(ax, Z, title):
        # Draw the chromosome as a thin curve
        ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], lw=0.6, color="lightgray", alpha=0.5, zorder=1)
        # Scatter bins colored by actual ChromHMM state
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=bin_colors, s=1.5,
                   alpha=0.8, edgecolors="none", zorder=2)
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=-100)

    # Panel 0: unconditional
    ax0 = fig.add_subplot(nrows, ncols, 1, projection="3d")
    _draw_panel(ax0, Z_uncond, "Unconditional")

    # Panels 1..G: conditional per group
    for g, Z_g in sorted(groupwise_positions.items()):
        ax = fig.add_subplot(nrows, ncols, g + 2, projection="3d")
        gname = group_names[g] if group_names and g < len(group_names) else f"Group {g}"
        _draw_panel(ax, Z_g, gname)

    # Legend: one colored dot per group
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_COARSE_GROUP_COLORS.get(
                       group_names[g] if group_names else "",
                       _STATE_COLORS[g % len(_STATE_COLORS)]),
                   markersize=5,
                   label=group_names[g] if group_names else f"Group {g}")
        for g in range(n_groups)
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=6, ncol=2, framealpha=0.8)

    fig.suptitle("Groupwise Conditional 3D Chromatin Positions", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Groupwise reconstructions — 3D + distance matrix per group
# ---------------------------------------------------------------------------

def plot_groupwise_reconstructions(
    Z_uncond: np.ndarray,
    groupwise_positions: dict,
    C: np.ndarray,
    group_names: list,
    valid_mask: np.ndarray | None,
    output_path: Path,
    panel_size: float = 3.5,
    resolution: int | None = None,
    start_bp: int = 0,
) -> None:
    """2-row grid: 3D structure (top) + reconstructed distances (bottom) per group.

    Columns: Unconditional + one per ChromHMM group (same order as
    groupwise_coordinates). Top row uses a shared cubic bounding box;
    bottom row uses a shared color range across all panels for direct
    visual comparison of pairwise distance scales.

    Args:
        Z_uncond: (N, 3) unconditional posterior mean positions.
        groupwise_positions: {g: (N, 3)} conditional posterior per group.
        C: (N,) integer group labels for coloring scatter dots.
        group_names: List of G group name strings.
        valid_mask: Optional (N_full,) bool to expand reconstruction with NaN gaps.
        output_path: Where to save the figure.
        panel_size: Width/height of each column in inches.
    """
    n_groups = len(groupwise_positions)
    n_cols = 1 + n_groups  # unconditional + one per group
    tk = 0.25  # track thickness relative to panel_size

    # Ordered list of (positions, title) for all columns
    all_structures = [(Z_uncond, "Unconditional")] + [
        (
            groupwise_positions[g],
            group_names[g] if group_names and g < len(group_names) else f"Group {g}",
        )
        for g in sorted(groupwise_positions.keys())
    ]

    # Shared cubic bounding box (same scale across all 3D panels)
    all_Z = np.vstack([Z for Z, _ in all_structures])
    centers = (all_Z.max(axis=0) + all_Z.min(axis=0)) / 2
    half_span = (all_Z.max(axis=0) - all_Z.min(axis=0)).max() / 2 * 1.05
    lims = [(centers[i] - half_span, centers[i] + half_span) for i in range(3)]

    # Pre-compute reconstruction matrices
    recon_mats = []
    for Z, _ in all_structures:
        Z_t = torch.tensor(Z)
        recon = torch.cdist(Z_t, Z_t).numpy()
        if valid_mask is not None:
            recon = _expand_to_full(recon, valid_mask)
        recon_mats.append(recon)

    # Anchor vmin/vmax to the unconditional panel so the scale matches reconstruction.png
    uncond_vals = recon_mats[0][~np.isnan(recon_mats[0])].ravel()
    vmin, vmax = float(uncond_vals.min()), float(uncond_vals.max())

    bin_colors = _get_group_colors(C, group_names)
    C_full = _expand_groups(C, valid_mask)

    # GridSpec: rows=[3D, h-track(thin), recon]
    #           cols=[v-track, main] × n_cols + [cbar]
    # The colorbar gets its own column so it never steals width from the recon
    # panels — that's what keeps the h-track and the matrix aligned (HMMC trick).
    ps = panel_size
    cbar_w = 0.25
    fig = plt.figure(figsize=(n_cols * (tk + ps) + cbar_w, 2 * ps + tk))
    gs = gridspec.GridSpec(3, 2 * n_cols + 1, figure=fig,
                           width_ratios=[tk, ps] * n_cols + [cbar_w],
                           height_ratios=[ps, tk, ps],
                           wspace=0.02, hspace=0.02)

    last_im = None
    for col_i, ((Z, title), recon_mat) in enumerate(zip(all_structures, recon_mats)):
        # 3D panel — spans v-track + main cols so it's centred above the recon
        ax3d = fig.add_subplot(gs[0, 2 * col_i: 2 * col_i + 2], projection="3d")
        ax3d.plot(Z[:, 0], Z[:, 1], Z[:, 2], lw=0.6, color="lightgray", alpha=0.5, zorder=1)
        ax3d.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=bin_colors, s=1.5,
                     alpha=0.8, edgecolors="none", zorder=2)
        ax3d.set_xlim(*lims[0])
        ax3d.set_ylim(*lims[1])
        ax3d.set_zlim(*lims[2])
        ax3d.set_title(title, fontsize=7, pad=2)
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=20, azim=-100)

        # h-track above recon (thin row, main col only)
        ax_ht = fig.add_subplot(gs[1, 2 * col_i + 1])
        _draw_chromhmm_track(ax_ht, C_full, group_names, horizontal=True)
        ax_ht.margins(0)
        ax_ht.set_xlim(-0.5, len(C_full) - 0.5)

        # v-track left of recon
        ax_vt = fig.add_subplot(gs[2, 2 * col_i])
        _draw_chromhmm_track(ax_vt, C_full, group_names, horizontal=False)
        ax_vt.margins(0)
        ax_vt.set_ylim(len(C_full) - 0.5, -0.5)

        # Reconstruction matrix — no colorbar here so its width is never stolen
        ax2d = fig.add_subplot(gs[2, 2 * col_i + 1])
        last_im = ax2d.matshow(recon_mat, cmap="YlOrRd_r", aspect="auto", vmin=vmin, vmax=vmax)
        if resolution is not None:
            _set_genomic_ticks(ax2d, recon_mat.shape[0], resolution, start_bp)
        else:
            ax2d.set_xticks([])
            ax2d.set_yticks([])

    # Single shared colorbar in the dedicated rightmost column
    ax_cbar = fig.add_subplot(gs[2, -1])
    fig.colorbar(last_im, cax=ax_cbar, label="Distance")
    ax_cbar.tick_params(labelsize=7)

    # Legend: colored patch per group, placed in the cbar column above the colorbar
    legend_elements = [
        mpatches.Patch(
            facecolor=_COARSE_GROUP_COLORS.get(
                group_names[g] if group_names else "",
                _STATE_COLORS[g % len(_STATE_COLORS)]),
            label=group_names[g] if group_names else f"Group {g}")
        for g in range(n_groups)
    ]
    ax_legend = fig.add_subplot(gs[0:2, -1])
    ax_legend.axis("off")
    ax_legend.legend(handles=legend_elements,
                     loc="center", fontsize=8,
                     framealpha=0.9, handlelength=1.5)

    fig.suptitle("Groupwise Conditional 3D Structures and Reconstructions",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config_path: str, animation: bool = False):
    """Generate all figures for a trained model.

    Reads elbo_history.csv, trajectory.npz (or legacy trajectory.npy),
    preprocessed data, and the model checkpoint. Produces:

        figures/elbo_curve.png
        figures/reconstruction.png
        figures/training_animation.gif  (only with --animation)
    """
    import torch.nn as nn
    import pandas as pd

    config = Config.from_yaml(config_path)

    region_slug = config.preprocessing.get("region", "unknown").replace(":", "_")
    model_name = config.model_name
    region_dir = Path(config.output_dir) / region_slug
    output_dir = region_dir / model_name          # model-specific outputs
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    data = load_preprocessed(region_dir)          # shared preprocessed dir
    scale = float(config.model.get("scale", 10000))
    data.X = data.X / scale
    N = data.n_bins
    X = data.X.numpy()
    # Use full matrix with NaN gaps for observed panel (notebook convention)
    if data.contact_raw_full is not None:
        Y_observed = data.contact_raw_full.numpy()  # (N_full, N_full) with NaN gaps
    elif data.contact_raw is not None:
        Y_observed = data.contact_raw.numpy()
    else:
        Y_observed = data.Y.numpy()
    C = data.C.numpy().astype(int) if data.C is not None else None
    group_names = data.group_names
    valid_mask_np = data.valid_mask.numpy() if data.valid_mask is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Genomic coordinate info for position tick labels ---
    prep = data.metadata.get("preprocessing", {})
    resolution = prep.get("resolution")
    _region_str = prep.get("region", "")
    if ":" in _region_str:
        _coords = _region_str.split(":")[1]
        start_bp = int(_coords.split("-")[0])
    else:
        start_bp = 0

    print(f"Data: {data}")

    # --- ELBO curve ---
    elbo_path = output_dir / "elbo_history.csv"
    if elbo_path.exists():
        df = pd.read_csv(elbo_path)
        elbo = df["elbo"].values
        plot_elbo(elbo, figures_dir / "elbo_curve.png")
    else:
        print(f"  ELBO history not found: {elbo_path}")

    # --- Load model checkpoint (shared between reconstruction + animation) ---
    checkpoint_path = output_dir / "checkpoints" / "model_final.pt"
    model = None
    use_groups = config.groups
    if checkpoint_path.exists():
        from ..commands.train import build_model
        model = build_model(
            config, X=data.X,
            C=data.C if use_groups else None,
            n_groups=data.n_groups if use_groups else 1,
        )
        model = model.to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  Checkpoint not found: {checkpoint_path}")

    # --- Final reconstruction figure ---
    gp_kwargs = {"groupsX": data.C.to(device)} if use_groups else {}
    if model is not None:
        with torch.no_grad():
            qZ, _, _ = model.gp(data.X.to(device), **gp_kwargs)
            Z_final = qZ.mean.T.cpu().numpy()  # (N, L)
        plot_reconstruction(Z_final, X, Y_observed, C, group_names,
                          valid_mask=valid_mask_np,
                          resolution=resolution, start_bp=start_bp,
                          output_path=figures_dir / "reconstruction.png")

    # --- Training animation (only with --animation) ---
    if animation:
        traj_npz = output_dir / "checkpoints" / "trajectory.npz"
        traj_npy = output_dir / "checkpoints" / "trajectory.npy"
        Zs_stack = None
        frame_iters = None

        if traj_npz.exists():
            print("Loading compact trajectory (mu + lengthscale)...")
            traj = np.load(traj_npz)
            traj_mus = traj["mu"]
            traj_steps = traj["steps"]
            traj_ls = traj.get("lengthscale", None)

            n_frames, L_traj, M_traj = traj_mus.shape
            print(f"  {n_frames} frames, mu shape=({L_traj}, {M_traj})")

            if L_traj == 3 and model is not None:
                train_ls = config.model.get("train_lengthscale", False)
                X_dev = data.X.to(device)

                print("  Reconstructing 3D positions from mu snapshots...")
                Zs_frames = []
                for fi in range(n_frames):
                    model.gp.mu = nn.Parameter(
                        torch.from_numpy(traj_mus[fi]).float().to(device)
                    )
                    if train_ls and traj_ls is not None:
                        model.gp.kernel.lengthscale = nn.Parameter(
                            torch.tensor(float(traj_ls[fi]), device=device)
                        )
                    with torch.no_grad():
                        qZ_i, _, _ = model.gp(X_dev, **gp_kwargs)
                        Zs_frames.append(qZ_i.mean.T.cpu().numpy())

                Zs_stack = np.stack(Zs_frames)
                frame_iters = list(traj_steps)
                print(f"  Reconstructed: {Zs_stack.shape}")
            else:
                print(f"  Skipping: need 3D latent space and checkpoint")

        elif traj_npy.exists():
            traj = np.load(traj_npy, allow_pickle=True)
            if traj.dtype == object:
                Zs_stack = np.stack(traj.tolist())
            else:
                Zs_stack = traj
            n_frames, N_traj, L_traj = Zs_stack.shape
            print(f"Legacy trajectory: {n_frames} frames, {N_traj} bins x {L_traj} dims")
            if L_traj == 3 and N_traj == N:
                total_steps = int(config.training.get("max_iter", 1000))
                frame_iters = list(range(0, total_steps + 1, 100))[:n_frames]
            else:
                print(f"  Skipping: shape mismatch (expected {N} bins, 3D)")
                Zs_stack = None
        else:
            print("  No trajectory found.")

        if Zs_stack is not None and frame_iters is not None:
            n_frames = Zs_stack.shape[0]
            gif_step = max(1, n_frames // 100)
            create_training_animation(
                Zs_stack, Y_observed, X, C, group_names, frame_iters,
                figures_dir / "training_animation.gif",
                valid_mask=valid_mask_np,
                step=gif_step, fps=10,
            )
    else:
        print("  Skipping training animation (use --animation to generate).")

    # --- Groupwise coordinates (reads analyze output) ---
    gw_dir = output_dir / "groupwise_positions"
    uncond_path = gw_dir / "unconditional.npy"
    if gw_dir.exists() and uncond_path.exists() and C is not None:
        Z_uncond = np.load(uncond_path)
        groupwise_positions = {}
        for p in sorted(gw_dir.glob("group_*.npy"),
                        key=lambda p: int(p.stem.split("_")[1])):
            g = int(p.stem.split("_")[1])
            groupwise_positions[g] = np.load(p)
        if groupwise_positions:
            plot_groupwise_coordinates(
                Z_uncond, groupwise_positions, C, group_names,
                figures_dir / "groupwise_coordinates.png",
            )
            plot_groupwise_reconstructions(
                Z_uncond, groupwise_positions, C, group_names,
                valid_mask=valid_mask_np,
                output_path=figures_dir / "groupwise_reconstructions.png",
                resolution=resolution, start_bp=start_bp,
            )
    else:
        print("  Skipping groupwise figures — run analyze first.")

    print("\nFigures complete.")
