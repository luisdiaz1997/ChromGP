#!/usr/bin/env python3
"""Generate helix_recovery.png — matches the working notebook pipeline exactly."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.distributions as distributions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpzoo.kernels import batched_RBF
from gpzoo.gp import SVGP
from gpzoo.utilities import add_jitter


# ── Exact ChromGP from the working notebook ─────────────────────────
class NotebookChromGP(nn.Module):
    def __init__(self, gp, kernel, noise=0.1, jitter=1e-5):
        super().__init__()
        self.gp = gp; self.kernel = kernel; self.jitter = jitter
        self.noise = nn.Parameter(torch.tensor(noise))

    def forward(self, X, E=1, verbose=False, **kwargs):
        N = len(X)
        noise = torch.nn.functional.softplus(self.noise)
        qZ, qU, pU = self.gp(X, verbose=verbose, **kwargs)
        Z = qZ.rsample().T  # (L,N) -> (N,L)
        Z = torch.squeeze(Z)
        Kzz = self.kernel.forward(Z, Z)
        Kzz = Kzz.contiguous()
        Kzz = add_jitter(Kzz, self.jitter)
        Kzz.view(-1)[::N+1] += (noise**2)
        pY = distributions.MultivariateNormal(torch.zeros_like(torch.squeeze(X)), Kzz)
        return pY, qZ, qU, pU


def make_helix(num_points=1000, radius=1.0, turns=5):
    t = torch.linspace(0, 2 * torch.pi * turns, num_points)
    return torch.stack([radius*torch.cos(t), radius*torch.sin(t), t/(torch.pi*turns)], dim=1)


def procrustes(X, Y):
    Xc = X - X.mean(axis=0); Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    return Yc @ (U @ Vt) + X.mean(axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # ── Synthetic data ──────────────────────────────────────────────
    N = 1000; tau = 0.1; nu = 50.0
    Z_true = make_helix(N, radius=1.0, turns=4)
    Z_true = Z_true - Z_true.mean(0)
    Z_true = Z_true / Z_true.norm(dim=1).max()

    D_true = torch.cdist(Z_true, Z_true)
    lam = nu * torch.exp(-D_true**2 / (2*tau**2))
    contacts = torch.poisson(lam)
    contacts = torch.tril(contacts) + torch.tril(contacts, -1).T

    X = torch.linspace(0, 2*np.pi*4, N)
    M = N; jitter = 1e-5

    # ── Build model (exact notebook pattern) ────────────────────────
    idx = torch.sort(torch.multinomial(torch.ones(N), M, replacement=False))[0]
    Z_ind = X[idx]

    kernel_gp = batched_RBF(sigma=1.0, lengthscale=0.7)
    gp = SVGP(kernel_gp, M=M, jitter=jitter)
    gp.Lu = nn.Parameter(1e-2 * torch.eye(M).expand(3, M, M).clone())
    gp.Z = nn.Parameter(Z_ind.unsqueeze(-1), requires_grad=False)
    mu_init = torch.squeeze(torch.stack((
        torch.sin(np.pi * gp.Z / 10.0),
        torch.cos(np.pi * gp.Z / 10.0),
        torch.sin(np.pi * gp.Z / 10.0),
    )))
    gp.mu = nn.Parameter(mu_init)

    out_kernel = batched_RBF(sigma=1.0, lengthscale=0.7)
    model = NotebookChromGP(gp, out_kernel, noise=0.1, jitter=1e-3)  # higher jitter for stability

    model.gp.kernel.lengthscale.requires_grad = False
    model.gp.kernel.sigma.requires_grad = False
    model.kernel.lengthscale.requires_grad = False
    model.kernel.sigma.requires_grad = False

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.02)

    print(f"Training N={N} M={M}")
    model.train()
    for step in range(10000):
        opt.zero_grad()
        pY, qZ, qU, pU = model(X.to(device).squeeze())
        yn = contacts.to(device) - contacts.to(device).mean(dim=1, keepdims=True)
        L1 = pY.log_prob(yn).sum()
        L2 = torch.sum(model.gp.kl_divergence(qU))
        (L1 - L2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % 2000 == 0:
            print(f"  step {step:4d}  ELBO={(L1-L2).item():.1f}")

    print(f"Final ELBO: {(L1-L2).item():.1f}")

    # ── Extract posterior mean ──────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, qZ, _, _ = model(X.to(device).squeeze())
        Z_rec = qZ.mean.T.cpu().numpy()  # (N, 3) — mean is (L,N), .T gives (N,L)

    Z_aligned = procrustes(Z_true.numpy(), Z_rec)
    rmsd = np.sqrt(np.mean(np.sum((Z_aligned - Z_true.numpy())**2, axis=1)))
    print(f"Procrustes RMSD: {rmsd:.4f}")

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    Z_np = Z_true.numpy()

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(Z_np[:,0], Z_np[:,1], Z_np[:,2], c=colors, s=2, alpha=0.8)
    ax1.set_title("Ground-truth helix", fontsize=11, fontweight="bold")
    ax1.view_init(elev=20, azim=-60); ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(Z_aligned[:,0], Z_aligned[:,1], Z_aligned[:,2], c=colors, s=2, alpha=0.8)
    ax2.set_title(f"ChromGP recovery\n(Procrustes RMSD = {rmsd:.3f})", fontsize=11, fontweight="bold")
    ax2.view_init(elev=20, azim=-60); ax2.set_axis_off()

    Z_al_t = torch.tensor(Z_aligned, dtype=torch.float32)
    recon_dist = torch.cdist(Z_al_t, Z_al_t)
    recon_rate = nu * torch.exp(-recon_dist**2 / (2*tau**2))

    ax3a = fig.add_subplot(2, 3, 3)
    ax3a.matshow(np.log10(contacts.numpy() + 1), cmap="YlOrRd_r", aspect="auto")
    ax3a.set_title("Observed contacts", fontsize=10, fontweight="bold")
    ax3a.set_xticks([]); ax3a.set_yticks([])

    ax3b = fig.add_subplot(2, 3, 6)
    ax3b.matshow(np.log10(recon_rate.numpy() + 1), cmap="YlOrRd_r", aspect="auto")
    ax3b.set_title("Reconstructed contacts", fontsize=10, fontweight="bold")
    ax3b.set_xticks([]); ax3b.set_yticks([])

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "helix_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
