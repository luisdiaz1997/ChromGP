"""Analyze a trained ChromGP model.

Computes groupwise conditional 3D positions for MGGP models,
following the SF convention: for each ChromHMM group g, run the GP forward
pass with all bins forced to group g to get the conditional posterior mean
Z_g (N, 3). Results are saved to groupwise_positions/ for the figures stage.

Also computes SCC (Stratum-adjusted Correlation Coefficient, Yang et al. 2017)
between observed contacts and reconstructed distances.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..analysis import scc, scc_groupwise
from ..config import Config
from ..datasets import load_preprocessed


def _compute_pc1_3d_alignment(Y: np.ndarray, Z: np.ndarray) -> dict:
    """PC1 of the Hi-C correlation matrix + best linear 3D direction.

    Returns a dict with:
      pc1            (N,) z-scored PC1 of corr(Y)
      direction      (3,) unit vector in 3D that best aligns with pc1
      direction_r    Pearson r of (Z @ direction) with pc1
      r_per_axis     dict {x,y,z} -> Pearson r with pc1
      r_radial       Pearson r of ||Z - mean|| with pc1
    """
    Y_safe = np.nan_to_num(Y, nan=np.nanmean(Y))
    corr = np.nan_to_num(np.corrcoef(Y_safe), nan=0.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    pc1 = eigvecs[:, -1] * np.sqrt(max(float(eigvals[-1]), 0.0))
    pc1 = (pc1 - pc1.mean()) / (pc1.std() + 1e-12)

    Z_c = Z - Z.mean(0)
    r = np.linalg.norm(Z_c, axis=1)
    r_per_axis = {ax: float(np.corrcoef(Z_c[:, i], pc1)[0, 1]) for i, ax in enumerate("xyz")}
    r_radial = float(np.corrcoef(r, pc1)[0, 1])

    coef, *_ = np.linalg.lstsq(Z_c, pc1, rcond=None)
    direction = coef / (np.linalg.norm(coef) + 1e-12)
    proj = Z_c @ direction
    direction_r = float(np.corrcoef(proj, pc1)[0, 1])

    return {
        "pc1": pc1.astype(np.float32),
        "direction": direction.astype(np.float32),
        "direction_r": direction_r,
        "r_per_axis": r_per_axis,
        "r_radial": r_radial,
    }


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

    scc_results = {}
    if data.contact_raw is None:
        print("\nSkipping SCC: no Hi-C contact_raw matrix in preprocessed data.")
    else:
        # --- SCC (Stratum-adjusted Correlation Coefficient) ---
        print("\nComputing SCC...")
        contact_raw = data.contact_raw.cpu().numpy() if isinstance(data.contact_raw, torch.Tensor) else data.contact_raw
        resolution = config.preprocessing.get("resolution", 25000)

        if use_groups and data.C is not None:
            C = data.C.cpu().numpy() if isinstance(data.C, torch.Tensor) else data.C
            scc_results = scc_groupwise(
                contact_raw, Z_uncond, C, data.group_names, resolution=resolution,
            )
            print(f"  SCC overall: {scc_results['overall']['scc']:.4f}")
            for k, v in scc_results.items():
                if k != "overall":
                    print(f"  SCC {v.get('name', k)}: {v['scc']:.4f}")
        else:
            scc_results = scc(contact_raw, Z_uncond, resolution=resolution)
            print(f"  SCC: {scc_results['scc']:.4f}")

    # Save predicted distance matrix from unconditional positions
    pred_dist = np.linalg.norm(
        Z_uncond[:, None, :] - Z_uncond[None, :, :], axis=-1
    )
    np.save(output_dir / "predicted_distance.npy", pred_dist)
    print(f"  Saved predicted_distance: {pred_dist.shape}")

    # --- Per-bin structural specificity (MGGP only) ---
    specificity = None
    dominant_state = None
    if use_groups:
        print("\nComputing per-bin specificity...")
        specificity = np.zeros(data.n_bins)
        dominant_state = np.zeros(data.n_bins, dtype=int)
        for i in range(data.n_bins):
            mu_bar = Z_uncond[i]
            norm_bar = np.linalg.norm(mu_bar)
            if norm_bar < 1e-8:
                continue
            shifts = np.array([np.linalg.norm(positions[g][i] - mu_bar) / norm_bar
                              for g in range(data.n_groups)])
            specificity[i] = shifts.max()
            dominant_state[i] = shifts.argmax()
        np.save(output_dir / "specificity.npy", specificity)
        np.save(output_dir / "dominant_state.npy", dominant_state)
        print(f"  Saved specificity: {specificity.shape}")
        print(f"  State-specific (>0.7): {(specificity > 0.7).sum()/data.n_bins*100:.1f}%")
        print(f"  State-enriched (0.1-0.7): {((specificity >= 0.1) & (specificity <= 0.7)).sum()/data.n_bins*100:.1f}%")
        print(f"  Universal (<0.1): {(specificity < 0.1).sum()/data.n_bins*100:.1f}%")

    # --- Hi-C PC1 + 3D axis alignment ---
    pc1_meta = None
    if data.contact_raw is not None and Z_uncond.shape[1] == 3:
        print("\nComputing Hi-C PC1 and 3D axis alignment...")
        Y_for_pc = data.Y.cpu().numpy() if isinstance(data.Y, torch.Tensor) else data.Y
        if Y_for_pc.ndim == 2 and Y_for_pc.shape[0] == Y_for_pc.shape[1] == Z_uncond.shape[0]:
            res = _compute_pc1_3d_alignment(Y_for_pc, Z_uncond)
            np.save(output_dir / "pc1.npy", res["pc1"])
            pc1_meta = {
                "direction": res["direction"].tolist(),
                "direction_r": res["direction_r"],
                "direction_r2": res["direction_r"] ** 2,
                "r_per_axis": res["r_per_axis"],
                "r_radial": res["r_radial"],
            }
            print(f"  PC1 best-fit 3D direction: {[round(v, 3) for v in pc1_meta['direction']]}")
            print(f"  r(proj, PC1) = {res['direction_r']:+.3f}  (R² = {pc1_meta['direction_r2']:.3f})")
            print(f"  per-axis r: {pc1_meta['r_per_axis']}")
            print(f"  r(||Z||, PC1) = {res['r_radial']:+.3f}")
        else:
            print(f"  Skipping PC1: Y shape {Y_for_pc.shape} doesn't match (N,N)")

    # --- analysis.json ---
    meta = {
        "n_bins": data.n_bins,
        "n_groups": data.n_groups,
        "group_names": data.group_names,
        "use_groups": use_groups,
        "model_name": model_name,
        "scc": scc_results,
        "pc1_3d_alignment": pc1_meta,
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\nAnalysis complete.")
