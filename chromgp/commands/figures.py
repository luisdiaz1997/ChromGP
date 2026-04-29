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

# Distinct colors for up to 15 chromHMM states (colorbrewer-inspired)
_STATE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
]


def _get_group_colors(C: np.ndarray) -> np.ndarray:
    """Map integer group labels to hex color strings."""
    n_groups = int(C.max()) + 1
    colors = _STATE_COLORS[:n_groups]
    return np.array(colors)[C.astype(int)]


def plot_reconstruction(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                        C: np.ndarray | None = None,
                        group_names: list | None = None,
                        output_path: Path | None = None) -> None:
    """1x3 panel: 3D structure colored by group | reconstructed distances | observed.

    Args:
        Z: Latent 3D positions (N, 3).
        X: Genomic coordinates (N,) — used to order bins along chromosome.
        Y: Observed distance/contact matrix (N, N).
        C: Optional group labels (N,).
        group_names: Optional group name strings.
        output_path: Where to save the figure.
    """
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # --- Panel 1: 3D structure ---
    # Line plot connecting genomic bins in order (notebook convention).
    # This shows the chromatin fiber as a continuous 3D curve.
    ax1.plot(Z[:, 0], Z[:, 1], Z[:, 2], lw=1.0)
    ax1.view_init(elev=20, azim=-100)

    ax1.set_title("3D Chromatin Structure", fontsize=12)

    # --- Panel 2: Reconstructed distances ---
    Z_t = torch.tensor(Z)
    recon_dist = torch.cdist(Z_t, Z_t).numpy()
    im2 = ax2.matshow(recon_dist, cmap="YlOrRd_r", aspect="auto")
    ax2.set_title("Reconstructed Distances", fontsize=12)
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # --- Panel 3: Observed data ---
    Y_obs = np.log10(Y + 5e-6)
    im3 = ax3.matshow(Y_obs, cmap="YlOrRd", aspect="auto")
    ax3.set_title("Observed Contact Matrix", fontsize=12)
    plt.colorbar(im3, ax=ax3, fraction=0.046)

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
        step: Frame stride (take every step-th frame).
        fps: Frames per second for output.
    """
    n_frames_total, N, D = Zs.shape
    frame_indices = list(range(0, n_frames_total, step))
    n_frames = len(frame_indices)

    s = 3.0 if N < 15000 else 100.0 / np.sqrt(N)

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
        ax1.plot(z[:, 0], z[:, 1], z[:, 2], lw=1.0)
        ax1.view_init(elev=20, azim=-100)
        ax1.set_title(f"3D Structure (iter {it})", fontsize=12)

        # --- Reconstructed distances ---
        Z_t = torch.tensor(z)
        recon = torch.cdist(Z_t, Z_t).numpy()
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
# Main entry point
# ---------------------------------------------------------------------------

def run(config_path: str):
    """Generate all figures for a trained model.

    Reads elbo_history.csv, trajectory.npz (or legacy trajectory.npy),
    preprocessed data, and the model checkpoint. Produces:

        figures/elbo_curve.png
        figures/reconstruction.png
        figures/training_animation.gif
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
    N = data.n_bins
    X = data.X.numpy()
    Y_observed = data.contact_raw.numpy() if data.contact_raw is not None else data.Y.numpy()
    C = data.C.numpy().astype(int) if data.C is not None else None
    group_names = data.group_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        from ..commands.train import build_model_svgp, build_model_mggp_svgp
        if use_groups:
            model = build_model_mggp_svgp(config, X=data.X, C=data.C, n_groups=data.n_groups)
        else:
            model = build_model_svgp(config, X=data.X)
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
                          figures_dir / "reconstruction.png")

    # --- Training animation ---
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
            step=gif_step, fps=10,
        )

    print("\nFigures complete.")
