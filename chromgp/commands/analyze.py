"""Analyze a trained ChromGP model.

Currently computes groupwise conditional 3D positions for MGGP models,
following the SF convention: for each ChromHMM group g, run the GP forward
pass with all bins forced to group g to get the conditional posterior mean
Z_g (N, 3). Results are saved to groupwise_positions/ for the figures stage.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..datasets import load_preprocessed


def _compute_groupwise_positions(
    model: nn.Module,
    X: torch.Tensor,
    n_groups: int,
    device: torch.device,
) -> dict:
    """Conditional posterior 3D positions for each ChromHMM group.

    For group g, forces groupsX = g for every bin and runs the GP forward to
    get the posterior mean under that group's kernel. This shows the hypothetical
    3D structure if all chromatin were in state g.

    Args:
        model: Trained ChromGP model with MGGP_SVGP prior.
        X: Bin midpoints (N,) on CPU.
        n_groups: Number of ChromHMM groups G.
        device: Compute device.

    Returns:
        Dict mapping group index → (N, 3) numpy array of 3D positions.
    """
    X_dev = X.to(device)
    positions = {}
    with torch.no_grad():
        for g in range(n_groups):
            groupsX_g = torch.full((len(X),), g, dtype=torch.long, device=device)
            qZ, _, _ = model.gp(X_dev, groupsX=groupsX_g)
            positions[g] = qZ.mean.T.cpu().numpy()  # (N, L)
    return positions


def run(config_path: str):
    """Analyze a trained ChromGP model and save intermediate results.

    Outputs (under <output_dir>/<region>/<model>/):
      - groupwise_positions/group_{g}.npy  (N, 3) conditional 3D positions per group
      - groupwise_positions/unconditional.npy  (N, 3) standard posterior mean
      - analysis.json  metadata
    """
    config = Config.from_yaml(config_path)

    region_slug = config.preprocessing.get("region", "unknown").replace(":", "_")
    model_name = config.model_name
    region_dir = Path(config.output_dir) / region_slug
    output_dir = region_dir / model_name
    checkpoint_path = output_dir / "checkpoints" / "model_final.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train first.")

    data = load_preprocessed(region_dir)
    scale = float(config.model.get("scale", 10000))
    data.X = data.X / scale
    print(f"Data: {data}")
    print(f"  X scaled by 1/{scale:.0f} -> range [{data.X.min().item():.1f}, {data.X.max().item():.1f}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    from ..commands.train import build_model
    use_groups = config.groups
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

    gw_dir = output_dir / "groupwise_positions"
    gw_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale group_*.npy files from a previous run with different n_groups
    for stale in gw_dir.glob("group_*.npy"):
        stale.unlink()

    # --- Unconditional posterior (standard forward, actual group labels) ---
    gp_kwargs = {"groupsX": data.C.to(device)} if use_groups else {}
    with torch.no_grad():
        qZ, _, _ = model.gp(data.X.to(device), **gp_kwargs)
        Z_uncond = qZ.mean.T.cpu().numpy()  # (N, L)
    np.save(gw_dir / "unconditional.npy", Z_uncond)
    print(f"  Saved unconditional positions: {Z_uncond.shape}")

    # --- Groupwise conditional posteriors (MGGP only) ---
    if use_groups:
        positions = _compute_groupwise_positions(model, data.X, data.n_groups, device)
        for g, Z_g in positions.items():
            np.save(gw_dir / f"group_{g}.npy", Z_g)
        print(f"  Saved {data.n_groups} groupwise position arrays")
    else:
        print("  Skipping groupwise positions (model has no groups).")

    # --- analysis.json ---
    meta = {
        "n_bins": data.n_bins,
        "n_groups": data.n_groups,
        "group_names": data.group_names,
        "use_groups": use_groups,
        "model_name": model_name,
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nAnalysis complete.")
