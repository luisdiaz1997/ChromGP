#!/usr/bin/env python3
"""Generate multiblock_conditional.png for dissertation.

Creates a synthetic 3-block structure with state-dependent interactions,
trains an MGGP SVGP, and produces a figure with ground-truth, full posterior,
and state-conditional posteriors.

Phases:
  python generate_multiblock_conditional.py              # train + save checkpoint + render
  python generate_multiblock_conditional.py --figures-only # re-render from saved data only
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chromgp.models import ChromGP
from gpzoo.gp import MGGP_SVGP
from gpzoo.kernels import batched_MGGP_Matern52
from gpzoo.modules import CholeskyParameter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic" / "multiblock_conditional" / "mggp_svgp"
DISS_FIG_DIR = PROJECT_ROOT / "Dissertation" / "figures" / "ch5"


def make_multiblock(N=600, K=3, within_scale=0.3, across_scale=1.5):
    torch.manual_seed(0)
    Z = torch.zeros(N, 3)
    centers = torch.randn(K, 3) * across_scale
    block_size = N // K
    groups = torch.zeros(N, dtype=torch.long)

    for k in range(K):
        start = k * block_size
        end = (k + 1) * block_size if k < K - 1 else N
        n_k = end - start
        Z[start:end] = centers[k] + torch.randn(n_k, 3) * within_scale
        groups[start:end] = k

    return Z, groups, centers


def procrustes(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    return Yc @ (U @ Vt) + X.mean(axis=0)


def _save_rgb(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    PILImage.open(path).convert("RGB").save(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    N = 600; K = 3; tau = 0.1; nu = 50.0; n_steps = 30000
    block_names = ["State A", "State B", "State C"]

    Z_true, groups, _ = make_multiblock(N, K)
    Z_true = Z_true - Z_true.mean(dim=0)
    Z_true = Z_true / Z_true.norm(dim=1).max()

    D_true = torch.cdist(Z_true, Z_true)
    lam = nu * torch.exp(-D_true**2 / (2 * tau**2))
    contacts = torch.poisson(lam)
    contacts = torch.tril(contacts) + torch.tril(contacts, -1).T

    X = torch.arange(N, dtype=torch.float32)
    M = min(300, N)

    input_kernel = batched_MGGP_Matern52(
        sigma=1.0, lengthscale=50.0,
        group_diff_param=0.1, n_groups=K,
    )
    gp = MGGP_SVGP(
        input_kernel, dim=1, M=M, jitter=1e-2,
        n_groups=K, cholesky_mode="exp", diagonal_only=False,
    )

    x_min, x_max = X.min().item(), X.max().item()
    padding = (x_max - x_min) * 0.02
    Z_init = torch.linspace(x_min + padding, x_max - padding, M).unsqueeze(-1)
    gp.Z = nn.Parameter(Z_init, requires_grad=False)

    groupsX = groups.unsqueeze(-1)
    groupsZ_init = groupsX[torch.randint(0, N, (M,))]
    gp.groupsZ = nn.Parameter(groupsZ_init, requires_grad=False)

    del gp.Lu
    L = 3
    gp.Lu = CholeskyParameter((L, M), mode="exp", diagonal_only=False)
    gp.mu = nn.Parameter(torch.randn(L, M) * 0.01)

    from gpzoo.kernels import batched_Matern52
    output_kernel = batched_Matern52(sigma=1.0, lengthscale=0.5)
    model = ChromGP(gp, output_kernel, noise=0.01, jitter=1e-2)

    X_dev = X.to(device)
    y_dev = contacts.to(device)
    C_dev = groups.to(device)
    model = model.to(device)

    t0 = time.time()
    print(f"Training MGGP N={N} M={M} K={K}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
    model.train()

    for step in range(n_steps):
        optimizer.zero_grad()
        pY, qZ, qU, pU = model(X_dev.squeeze(), groupsX=C_dev)
        y_norm = y_dev - y_dev.mean(dim=1, keepdims=True)

        L1 = pY.log_prob(y_norm).sum()
        L2 = model.gp.kl_divergence(qU, pU).sum()
        loss = -(L1 - L2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        if step % 10000 == 0:
            print(f"  step {step:5d}  ELBO={loss.item():.1f}")

    print(f"Final ELBO: {loss.item():.1f}")

    model.eval()
    with torch.no_grad():
        qF, _, _ = model.gp(X_dev, groupsX=C_dev)
        Z_uncond = qF.mean.T.cpu().numpy()

        Z_cond = {}
        for g in range(K):
            gX = torch.full((N,), g, dtype=torch.long, device=device)
            qF_g, _, _ = model.gp(X_dev, groupsX=gX)
            Z_cond[g] = qF_g.mean.T.cpu().numpy()

    Z_aligned = procrustes(Z_true.numpy(), Z_uncond)
    rmsd = np.sqrt(np.mean((Z_aligned - Z_true.numpy())**2))
    print(f"Unconditional Procrustes RMSD: {rmsd:.4f}")

    Z_cond_aligned = {}
    for g in range(K):
        Z_cond_aligned[g] = procrustes(Z_true.numpy(), Z_cond[g])

    # --- Save outputs ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "steps": n_steps,
        "train_time": time.time() - t0,
        "demo_config": {"N": N, "M": M, "K": K, "tau": tau, "nu": nu},
    }, OUTPUT_DIR / "checkpoints" / "model_final.pt")

    savez_kwargs = dict(
        X=X.numpy(),
        Z_true=Z_true.numpy(),
        contacts=contacts.numpy(),
        groups=groups.numpy(),
        Z_unconditional=Z_aligned,
        rmsd=np.array(rmsd),
        tau=np.array(tau),
        nu=np.array(nu),
    )
    for g in range(K):
        savez_kwargs[f"Z_conditional_{g}"] = Z_cond_aligned[g]
    np.savez(OUTPUT_DIR / "synthetic_data.npz", **savez_kwargs)

    print(f"  Saved: {OUTPUT_DIR / 'checkpoints' / 'model_final.pt'}")
    print(f"  Saved: {OUTPUT_DIR / 'synthetic_data.npz'}")

    run_figures()


def run_figures():
    data = np.load(OUTPUT_DIR / "synthetic_data.npz")
    Z_true = data["Z_true"]
    Z_aligned = data["Z_unconditional"]
    contacts = data["contacts"]
    g_np = data["groups"].astype(int)
    rmsd = float(data["rmsd"])
    K = int(g_np.max()) + 1
    block_names = ["State A", "State B", "State C"]
    state_colors = ["#E41A1C", "#377EB8", "#4DAF4A"]

    fig = plt.figure(figsize=(14, 9))

    for col, (title, Z_plot) in enumerate([
        ("Ground-truth multi-block", Z_true),
        (f"ChromGP full posterior (RMSD={rmsd:.3f})", Z_aligned),
    ]):
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        for g in range(K):
            mask = g_np == g
            ax.scatter(Z_plot[mask, 0], Z_plot[mask, 1], Z_plot[mask, 2],
                      c=state_colors[g], s=8, alpha=0.7, label=block_names[g])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()

    for g in range(K):
        ax = fig.add_subplot(2, 3, g + 4, projection="3d")
        Z_c = data[f"Z_conditional_{g}"]
        ax.scatter(Z_c[:, 0], Z_c[:, 1], Z_c[:, 2],
                  c=state_colors[g], s=8, alpha=0.7)
        ax.set_title(f"Conditional: {block_names[g]}", fontsize=10, fontweight="bold")
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()

    plt.tight_layout()

    figs_dir = OUTPUT_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out = figs_dir / "multiblock_conditional.png"
    _save_rgb(fig, out)

    DISS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, DISS_FIG_DIR / "multiblock_conditional.png")
    print(f"  Copied to: {DISS_FIG_DIR / 'multiblock_conditional.png'}")


def main():
    parser = argparse.ArgumentParser(description="Multiblock conditional demo (train + figure)")
    parser.add_argument("--figures-only", action="store_true",
                        help="Load saved data and render figures only (no training)")
    args = parser.parse_args()

    if args.figures_only:
        run_figures()
    else:
        run_train()


if __name__ == "__main__":
    main()
