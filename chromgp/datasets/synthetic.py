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


def _poisson_contacts(Z: torch.Tensor, tau: float, nu: float) -> torch.Tensor:
    D = torch.cdist(Z, Z)
    lam = nu * torch.exp(-D**2 / (2 * tau**2))
    contacts = torch.poisson(lam)
    return torch.tril(contacts) + torch.tril(contacts, -1).T


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

        tau = float(preprocessing.get("tau", 0.1))
        nu = float(preprocessing.get("nu", 50.0))
        contacts = _poisson_contacts(Z_true, tau=tau, nu=nu)

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
                "tau": tau,
                "nu": nu,
                "Z_true": Z_true.numpy(),
            },
        )
