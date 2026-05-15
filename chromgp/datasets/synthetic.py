"""Synthetic dataset loader.

Generates a ground-truth 3D shape (helix for now), simulates a Poisson
contact map from pairwise distances, and returns it as a GenomicData so
the standard preprocess -> train -> analyze -> figures pipeline applies.

The ground-truth Z_true is stashed in metadata['Z_true'] so the preprocess
command can persist it as Z_true.npy alongside X/Y/C/contact_raw.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import GenomicData


def _make_helix(N: int, radius: float, turns: float) -> torch.Tensor:
    t = torch.linspace(0, 2 * torch.pi * turns, N)
    return torch.stack([radius * torch.cos(t), radius * torch.sin(t),
                        t / (torch.pi * turns)], dim=1)


def _poisson_contacts_spiral(Z: torch.Tensor, replicates: int = 16, z_noise: float = 0.01) -> torch.Tensor:
    """Sum of `replicates` symmetric Poisson contact maps from noisy copies of Z.

    Notebook convention: rate = 1 / (1 + D^2).
    """
    total = torch.zeros(Z.shape[0], Z.shape[0])
    for _ in range(replicates):
        Z_eff = Z + z_noise * torch.randn_like(Z)
        D = torch.cdist(Z_eff, Z_eff)
        lam = 1.0 / (1.0 + D**2)
        samples = torch.poisson(lam)
        total = total + (torch.tril(samples) + torch.tril(samples, -1).T)
    return total


def _poisson_contacts_hastie(Z: torch.Tensor, beta: float = 5.0) -> torch.Tensor:
    """Single symmetric Poisson contact map under the PoisMS (Hastie et al. 2022) model.

    log lambda_ij = beta - ||x_i - x_j||^2
    C_ij ~ Poisson(lambda_ij)
    """
    D = torch.cdist(Z, Z)
    lam = torch.exp(beta - D**2)
    samples = torch.poisson(lam)
    return torch.tril(samples) + torch.tril(samples, -1).T


class SyntheticLoader:
    """Generate a synthetic dataset (3D shape + Poisson contacts).

    Triggered by preprocessing.assay == 'synthetic'. Supported shapes:
      - 'helix'

    Returns
    -------
    GenomicData
        X (N,) bin midpoints (linear, in arbitrary units),
        Y (N, N) contact matrix (D=N replicates),
        contact_raw (N, N) same matrix,
        C (N,) zeros (single group),
        metadata['Z_true'] = (N, 3) ground-truth positions.
    """

    def load(self, preprocessing: dict) -> GenomicData:
        shape = preprocessing.get("shape", "helix")
        N = int(preprocessing.get("N", 1000))
        seed = int(preprocessing.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        if shape == "helix":
            radius = float(preprocessing.get("radius", 1.0))
            turns = float(preprocessing.get("turns", 4))
            Z_true = _make_helix(N, radius=radius, turns=turns)
            x_max = float(preprocessing.get("x_max", 2 * np.pi * turns))
        else:
            raise ValueError(f"Unknown synthetic shape: {shape}")

        Z_true = Z_true - Z_true.mean(0)
        Z_true = Z_true / Z_true.norm(dim=1).max()

        contact_model = str(preprocessing.get("contact_model", "spiral")).lower()
        if contact_model == "hastie":
            beta = float(preprocessing.get("beta", 5.0))
            contacts = _poisson_contacts_hastie(Z_true, beta=beta)
        else:
            replicates = int(preprocessing.get("replicates", 16))
            z_noise = float(preprocessing.get("z_noise", 0.01))
            contacts = _poisson_contacts_spiral(Z_true, replicates=replicates, z_noise=z_noise)

        X = torch.linspace(0.0, x_max, N)

        return GenomicData(
            X=X.float(),
            Y=contacts.float(),
            C=torch.zeros(N, dtype=torch.long),
            n_groups=1,
            group_names=["All"],
            contact_raw=contacts.float(),
            metadata={
                "shape": shape,
                "N": N,
                "Z_true": Z_true.numpy(),
            },
        )
