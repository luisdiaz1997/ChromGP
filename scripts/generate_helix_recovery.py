#!/usr/bin/env python3
"""Generate helix_recovery.png for dissertation.

Trains a ChromGP SVGP on synthetic helix contacts using the ChromGP training util,
then produces a 3-panel figure: ground-truth, recovered, and contact maps.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chromgp.simulations import make_helix
from chromgp.models import ChromGP
from gpzoo.kernels import batched_RBF
from gpzoo.gp import SVGP
from gpzoo.modules import CholeskyParameter


def procrustes(X, Y):
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    return Yc @ (U @ Vt) + X.mean(axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"Device: {device}")

    # ── Synthetic data ──────────────────────────────────────────────
    N = 500
    tau = 0.1
    nu = 50.0
    Z_true = make_helix(num_points=N, radius=1.0, turns=3)
    Z_true = Z_true - Z_true.mean(dim=0)
    Z_true = Z_true / Z_true.norm(dim=1).max()

    D_true = torch.cdist(Z_true, Z_true)
    lam = nu * torch.exp(-D_true**2 / (2 * tau**2))
    contacts = torch.poisson(lam)
    contacts = torch.tril(contacts) + torch.tril(contacts, -1).T

    X = torch.arange(N, dtype=torch.float32)
    M = min(400, N)

    # ── Build model ─────────────────────────────────────────────────
    input_kernel = batched_RBF(sigma=1.0, lengthscale=20.0)
    gp = SVGP(input_kernel, dim=1, M=M, jitter=1e-2,
              cholesky_mode="exp", diagonal_only=False)

    x_min, x_max = X.min().item(), X.max().item()
    padding = (x_max - x_min) * 0.02
    Z_init = torch.linspace(x_min + padding, x_max - padding, M).unsqueeze(-1)
    gp.Z = torch.nn.Parameter(Z_init, requires_grad=False)
    del gp.Lu
    L = 3
    gp.Lu = CholeskyParameter((L, M), mode="exp", diagonal_only=False)
    gp.mu = torch.nn.Parameter(torch.randn(L, M) * 0.1)

    output_kernel = batched_RBF(sigma=1.0, lengthscale=1.0)
    model = ChromGP(gp, output_kernel, noise=0.1, jitter=1e-2)

    X_dev = X.to(device)
    y_dev = contacts.to(device)
    model = model.to(device)

    # ── Train (matching commands/train.py pattern) ─────────────────
    print(f"Training N={N} M={M}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30000, eta_min=1e-5)
    losses = []
    model.train()

    for step in range(30000):
        optimizer.zero_grad()
        pY, qZ, qU, pU = model(X_dev.squeeze())
        y_norm = y_dev - y_dev.mean(dim=1, keepdims=True)

        L1 = pY.log_prob(y_norm).sum()
        L2 = model.gp.kl_divergence(qU, pU).sum()
        loss = -(L1 - L2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % 10000 == 0:
            print(f"  step {step:5d}  ELBO={losses[-1]:.1f}")

    print(f"Final ELBO: {losses[-1]:.1f}")

    # ── Extract posterior mean ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, qZ, _, _ = model(X.to(device))
        Z_recovered = qZ.mean.T.cpu().numpy()

    # ── Procrustes alignment ────────────────────────────────────────
    Z_aligned = procrustes(Z_true.numpy(), Z_recovered)
    rmsd = np.sqrt(np.mean((Z_aligned - Z_true.numpy())**2))
    print(f"Procrustes RMSD: {rmsd:.4f}")

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    Z_np = Z_true.numpy()

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(Z_np[:, 0], Z_np[:, 1], Z_np[:, 2], c=colors, s=3, alpha=0.8)
    ax1.set_title("Ground-truth helix", fontsize=11, fontweight="bold")
    ax1.view_init(elev=20, azim=-60)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(Z_aligned[:, 0], Z_aligned[:, 1], Z_aligned[:, 2], c=colors, s=3, alpha=0.8)
    ax2.set_title(f"ChromGP recovery\n(Procrustes RMSD = {rmsd:.3f})", fontsize=11, fontweight="bold")
    ax2.view_init(elev=20, azim=-60)
    ax2.set_axis_off()

    Z_al_t = torch.tensor(Z_aligned, dtype=torch.float32)
    recon_dist = torch.cdist(Z_al_t, Z_al_t)
    recon_rate = nu * torch.exp(-recon_dist**2 / (2 * tau**2))

    ax3a = fig.add_subplot(2, 3, 3)
    ax3a.matshow(np.log10(contacts.numpy() + 1), cmap="YlOrRd_r", aspect="auto")
    ax3a.set_title("Observed contacts (log10)", fontsize=10, fontweight="bold")
    ax3a.set_xticks([]); ax3a.set_yticks([])

    ax3b = fig.add_subplot(2, 3, 6)
    ax3b.matshow(np.log10(recon_rate.numpy() + 1), cmap="YlOrRd_r", aspect="auto")
    ax3b.set_title("Reconstructed contacts (log10)", fontsize=10, fontweight="bold")
    ax3b.set_xticks([]); ax3b.set_yticks([])

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "helix_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
