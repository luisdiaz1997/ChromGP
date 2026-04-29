"""Train a ChromGP model (SVGP or MGGP_SVGP prior).

Training loop with batch scaling following Probabilistic-NMF conventions:
    loss = -(L1 - L2)
      L1 = log p(y|F)     -- scaled by N/batch_size (and D/y_batch_size if used)
      L2 = KL[q(U)||p(U)] -- scaled by N/M (SVGP inducing-point KL)

When config.groups=True the MGGP_SVGP prior is used: each genomic bin is
assigned a ChromHMM group label (stored as C.npy) and the kernel modulates
covariance based on group membership, following the PNMF/SF convention.

Supports --resume to continue training from a saved checkpoint.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import trange

from ..config import Config
from ..datasets import load_preprocessed
from ..models import ChromGP


def _save_elbo_history(elbo_history: list, output_dir: Path) -> None:
    """Save ELBO history (both CSV and numpy formats)."""
    df = pd.DataFrame({"iteration": range(len(elbo_history)), "elbo": elbo_history})
    df.to_csv(output_dir / "elbo_history.csv", index=False)
    np.save(output_dir / "elbo_history.npy", np.array(elbo_history))


def _append_elbo_history(new_elbo_history: list, output_dir: Path) -> None:
    """Append new ELBO values to existing elbo_history.csv/npy (SF convention)."""
    csv_path = output_dir / "elbo_history.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        start_iter = int(existing_df["iteration"].max()) + 1
    else:
        existing_df = pd.DataFrame({"iteration": [], "elbo": []})
        start_iter = 0

    new_df = pd.DataFrame({
        "iteration": range(start_iter, start_iter + len(new_elbo_history)),
        "elbo": new_elbo_history,
    })
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    np.save(output_dir / "elbo_history.npy", combined["elbo"].values)


def _append_trajectory(new_traj: dict, traj_path: Path) -> None:
    """Append new segment trajectory frames to existing trajectory.npz."""
    old = np.load(traj_path)
    traj_data = {
        "mu": np.concatenate([old["mu"], new_traj["mu"]], axis=0),
        "steps": np.concatenate([old["steps"], new_traj["steps"]], axis=0),
    }
    if "lengthscale" in old and "lengthscale" in new_traj:
        traj_data["lengthscale"] = np.concatenate([old["lengthscale"], new_traj["lengthscale"]])
    elif "lengthscale" in new_traj:
        traj_data["lengthscale"] = new_traj["lengthscale"]
    np.savez(traj_path, **traj_data)


def _init_groupsZ(Z: torch.Tensor, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Assign a group label to each inducing point by nearest-bin lookup.

    Args:
        Z: Inducing point positions (M, 1).
        X: Bin midpoints (N,).
        C: Group labels for bins (N,), integer 0..G-1.

    Returns:
        groupsZ: Group labels for inducing points (M,), LongTensor.
    """
    # Nearest neighbour in 1D: |Z_i - X_j| minimum over j
    dists = torch.abs(Z.squeeze(-1).unsqueeze(1) - X.unsqueeze(0))  # (M, N)
    nn_idx = dists.argmin(dim=1)  # (M,)
    return C[nn_idx].long()


def build_model_svgp(config: Config, X: torch.Tensor = None) -> nn.Module:
    """ChromGP(SVGP) with batched (L,M) variational parameters."""
    from gpzoo.gp import SVGP
    from gpzoo.kernels import batched_RBF, batched_Matern32
    from gpzoo.modules import CholeskyParameter

    L = config.model.get("n_components", 3)
    M = config.model.get("num_inducing", 800)
    jitter = float(config.model.get("jitter", 1e-5))
    noise = float(config.model.get("noise", 0.1))
    ls = float(config.model.get("lengthscale", 5e6))
    out_ls = float(config.model.get("output_lengthscale", 1.0))
    sigma = float(config.model.get("sigma", 1.0))
    cholesky_mode = config.model.get("cholesky_mode", "exp")

    # Input kernel on 1D genomic coordinates — Matern32 is default (SF convention)
    kernel_name = config.model.get("kernel", "Matern32").lower()
    if kernel_name == "matern32":
        svgp_kernel = batched_Matern32(sigma=sigma, lengthscale=ls)
    elif kernel_name == "rbf":
        svgp_kernel = batched_RBF(sigma=sigma, lengthscale=ls)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")
    gp = SVGP(svgp_kernel, dim=1, M=M, jitter=jitter,
               cholesky_mode=cholesky_mode, diagonal_only=False)

    if X is not None:
        x_min, x_max = X.min().item(), X.max().item()
        padding = (x_max - x_min) * 0.02
        Z_init = torch.linspace(x_min + padding, x_max - padding, M).unsqueeze(-1)
        gp.Z = nn.Parameter(Z_init, requires_grad=False)

    del gp.Lu
    gp.Lu = CholeskyParameter((L, M), mode=cholesky_mode, diagonal_only=False)
    gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)

    output_kernel = batched_RBF(sigma=sigma, lengthscale=out_ls)
    model = ChromGP(gp, output_kernel, noise=noise, jitter=jitter)

    svgp_kernel.sigma.requires_grad = False
    if not config.model.get("train_lengthscale", False):
        svgp_kernel.lengthscale.requires_grad = False
    output_kernel.sigma.requires_grad = False
    output_kernel.lengthscale.requires_grad = False

    return model


def build_model_mggp_svgp(
    config: Config,
    X: torch.Tensor = None,
    C: torch.Tensor = None,
    n_groups: int = 1,
) -> nn.Module:
    """ChromGP(MGGP_SVGP) with per-group kernel and batched (L,M) variational parameters.

    The MGGP kernel modulates covariance between bins by their ChromHMM group
    membership via a group_diff_param, following the PNMF/SF pattern.

    Args:
        config: Experiment configuration.
        X: Bin midpoints (N,) for inducing point initialisation.
        C: Group labels (N,) for nearest-bin groupsZ initialisation.
        n_groups: Number of unique ChromHMM groups G.
    """
    from gpzoo.gp import MGGP_SVGP
    from gpzoo.kernels import batched_MGGP_RBF, batched_MGGP_Matern32, batched_RBF
    from gpzoo.modules import CholeskyParameter

    L = config.model.get("n_components", 3)
    M = config.model.get("num_inducing", 800)
    jitter = float(config.model.get("jitter", 1e-5))
    noise = float(config.model.get("noise", 0.1))
    ls = float(config.model.get("lengthscale", 5e6))
    out_ls = float(config.model.get("output_lengthscale", 1.0))
    sigma = float(config.model.get("sigma", 1.0))
    cholesky_mode = config.model.get("cholesky_mode", "exp")
    group_diff_param = float(config.model.get("group_diff_param", 1.0))

    # Input kernel on 1D genomic coordinates — Matern32 is default (SF convention)
    kernel_name = config.model.get("kernel", "Matern32").lower()
    if kernel_name == "matern32":
        mggp_kernel = batched_MGGP_Matern32(
            sigma=sigma, lengthscale=ls,
            group_diff_param=group_diff_param, n_groups=n_groups,
        )
    elif kernel_name == "rbf":
        mggp_kernel = batched_MGGP_RBF(
            sigma=sigma, lengthscale=ls,
            group_diff_param=group_diff_param, n_groups=n_groups,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    gp = MGGP_SVGP(
        mggp_kernel, dim=1, M=M, jitter=jitter,
        n_groups=n_groups, cholesky_mode=cholesky_mode, diagonal_only=False,
    )

    # Inducing points: linspace across data range, groups by nearest bin
    if X is not None:
        x_min, x_max = X.min().item(), X.max().item()
        padding = (x_max - x_min) * 0.02
        Z_init = torch.linspace(x_min + padding, x_max - padding, M).unsqueeze(-1)
        gp.Z = nn.Parameter(Z_init, requires_grad=False)

        if C is not None:
            groupsZ = _init_groupsZ(Z_init, X, C)
            gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)

    del gp.Lu
    gp.Lu = CholeskyParameter((L, M), mode=cholesky_mode, diagonal_only=False)
    gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)

    output_kernel = batched_RBF(sigma=sigma, lengthscale=out_ls)
    model = ChromGP(gp, output_kernel, noise=noise, jitter=jitter)

    mggp_kernel.sigma.requires_grad = False
    mggp_kernel.group_diff_param.requires_grad = False
    if not config.model.get("train_lengthscale", False):
        mggp_kernel.lengthscale.requires_grad = False
    output_kernel.sigma.requires_grad = False
    output_kernel.lengthscale.requires_grad = False

    return model


def build_model_lcgp(config: Config, X: torch.Tensor) -> nn.Module:
    """ChromGP(LCGP) with batched (L,M,R) variational parameters."""
    from gpzoo.gp import LCGP
    from gpzoo.kernels import batched_RBF, batched_Matern32
    from gpzoo.utilities import estimate_lcgp_rank
    from gpzoo.knn_utilities import calculate_knn

    L = config.model.get("n_components", 3)
    N = len(X)
    M = N  # LCGP uses all points as inducing points
    jitter = float(config.model.get("jitter", 1e-5))
    noise = float(config.model.get("noise", 0.1))
    ls = float(config.model.get("lengthscale", 5e6))
    out_ls = float(config.model.get("output_lengthscale", 1.0))
    sigma = float(config.model.get("sigma", 1.0))
    K = int(config.model.get("K", 50))
    neighbors = config.model.get("neighbors", "probabilistic")

    kernel_name = config.model.get("kernel", "Matern32").lower()
    if kernel_name == "matern32":
        lcgp_kernel = batched_Matern32(sigma=sigma, lengthscale=ls)
    elif kernel_name == "rbf":
        lcgp_kernel = batched_RBF(sigma=sigma, lengthscale=ls)
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    gp = LCGP(lcgp_kernel, dim=1, M=M, jitter=jitter, K=K)

    Z_init = X.unsqueeze(-1).clone()
    gp.Z = nn.Parameter(Z_init, requires_grad=False)

    # Initialize Lu as raw nn.Parameter
    domain_range = (float(X.min()), float(X.max()))
    R = estimate_lcgp_rank(ls, domain_range, dim=1, p=0.9)
    R = max(1, min(R, 250))
    
    del gp.Lu
    gp.Lu = nn.Parameter(torch.randn(L, M, R) * (1.0 / R ** 0.5))
    gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)

    # Precompute KNN indices
    raw = calculate_knn(gp, Z_init, strategy=neighbors)
    gp.knn_idx = raw[:, :-1]
    gp.knn_idz = raw[:, 1:]

    output_kernel = batched_RBF(sigma=sigma, lengthscale=out_ls)
    model = ChromGP(gp, output_kernel, noise=noise, jitter=jitter)

    lcgp_kernel.sigma.requires_grad = False
    if not config.model.get("train_lengthscale", False):
        lcgp_kernel.lengthscale.requires_grad = False
    output_kernel.sigma.requires_grad = False
    output_kernel.lengthscale.requires_grad = False

    return model


def build_model_mggp_lcgp(
    config: Config,
    X: torch.Tensor = None,
    C: torch.Tensor = None,
    n_groups: int = 1,
) -> nn.Module:
    """ChromGP(MGGP_LCGP) with batched (L,M,R) variational parameters."""
    from gpzoo.gp import MGGP_LCGP
    from gpzoo.kernels import batched_MGGP_RBF, batched_MGGP_Matern32, batched_RBF
    from gpzoo.utilities import estimate_lcgp_rank
    from gpzoo.knn_utilities import calculate_knn

    L = config.model.get("n_components", 3)
    N = len(X)
    M = N
    jitter = float(config.model.get("jitter", 1e-5))
    noise = float(config.model.get("noise", 0.1))
    ls = float(config.model.get("lengthscale", 5e6))
    out_ls = float(config.model.get("output_lengthscale", 1.0))
    sigma = float(config.model.get("sigma", 1.0))
    group_diff_param = float(config.model.get("group_diff_param", 1.0))
    K = int(config.model.get("K", 50))
    neighbors = config.model.get("neighbors", "probabilistic")

    kernel_name = config.model.get("kernel", "Matern32").lower()
    if kernel_name == "matern32":
        mggp_kernel = batched_MGGP_Matern32(
            sigma=sigma, lengthscale=ls,
            group_diff_param=group_diff_param, n_groups=n_groups,
        )
    elif kernel_name == "rbf":
        mggp_kernel = batched_MGGP_RBF(
            sigma=sigma, lengthscale=ls,
            group_diff_param=group_diff_param, n_groups=n_groups,
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    gp = MGGP_LCGP(
        mggp_kernel, dim=1, M=M, jitter=jitter,
        n_groups=n_groups, K=K,
    )

    Z_init = X.unsqueeze(-1).clone()
    gp.Z = nn.Parameter(Z_init, requires_grad=False)

    if C is not None:
        groupsZ = _init_groupsZ(Z_init, X, C)
        gp.groupsZ = nn.Parameter(groupsZ, requires_grad=False)
    else:
        groupsZ = None

    domain_range = (float(X.min()), float(X.max()))
    R = estimate_lcgp_rank(ls, domain_range, dim=1, p=0.9)
    R = max(1, min(R, 250))
    
    del gp.Lu
    gp.Lu = nn.Parameter(torch.randn(L, M, R) * (1.0 / R ** 0.5))
    gp.mu = nn.Parameter(torch.randn(L, M) * 1.0)

    raw = calculate_knn(
        gp, Z_init, strategy=neighbors,
        multigroup=True,
        groupsX=groupsZ, groupsZ=groupsZ,
    )
    gp.knn_idx = raw[:, :-1]
    gp.knn_idz = raw[:, 1:]

    output_kernel = batched_RBF(sigma=sigma, lengthscale=out_ls)
    model = ChromGP(gp, output_kernel, noise=noise, jitter=jitter)

    mggp_kernel.sigma.requires_grad = False
    mggp_kernel.group_diff_param.requires_grad = False
    if not config.model.get("train_lengthscale", False):
        mggp_kernel.lengthscale.requires_grad = False
    output_kernel.sigma.requires_grad = False
    output_kernel.lengthscale.requires_grad = False

    return model

# Keep old name as alias for backward compat with any external callers
build_model = build_model_svgp


def run(config_path: str, resume: bool = False, video: bool = False):
    """Train a ChromGP model (SVGP or MGGP_SVGP based on config.groups).

    Loads preprocessed data, builds the model, runs the training loop
    (optionally resuming from a checkpoint), and saves:
      - checkpoints/model_final.pt   (full checkpoint)
      - checkpoints/trajectory.npy    (latent means per step)
      - elbo_history.csv / .npy       (SF-style loss history)
      - training.json                 (SF-style training metadata)
    """
    config = Config.from_yaml(config_path)

    # --- Paths ---
    region_slug = config.preprocessing.get("region", "unknown").replace(":", "_")
    model_name = config.model_name
    region_dir = Path(config.output_dir) / region_slug
    output_dir = region_dir / model_name          # model-specific outputs
    save_dir = output_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed(region_dir)          # shared preprocessed dir
    N = data.n_bins
    D = data.n_features
    print(f"Data: {data}")

    use_groups = config.groups
    if use_groups and data.C is None:
        raise ValueError(
            "config.groups=True but no group labels found in preprocessed data. "
            "Re-run preprocess with groups_by set."
        )

    # --- Device ---
    device_cfg = config.training.get("device", "gpu")
    if device_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Build model ---
    M = config.model.get("num_inducing", 800)
    L = config.model.get("n_components", 3)
    prior_type = config.model.get("prior", "SVGP").upper()

    if use_groups:
        if prior_type == "LCGP":
            model = build_model_mggp_lcgp(
                config, X=data.X, C=data.C, n_groups=data.n_groups,
            )
            print(f"Model: ChromGP + MGGP_LCGP  (N={N}, L={L}, G={data.n_groups})")
            print(f"  group_diff_param: {config.model.get('group_diff_param', 1.0)}")
        else:
            model = build_model_mggp_svgp(
                config, X=data.X, C=data.C, n_groups=data.n_groups,
            )
            print(f"Model: ChromGP + MGGP_SVGP  (M={M}, L={L}, G={data.n_groups})")
            print(f"  group_diff_param: {config.model.get('group_diff_param', 1.0)}")
    else:
        if prior_type == "LCGP":
            model = build_model_lcgp(config, X=data.X)
            print(f"Model: ChromGP + LCGP  (N={N}, L={L})")
        else:
            model = build_model_svgp(config, X=data.X)
            print(f"Model: ChromGP + SVGP  (M={M}, L={L})")

    model = model.to(device)
    train_ls = config.model.get("train_lengthscale", False)
    print(f"  Input lengthscale: trainable={train_ls}")
    print(f"  Output lengthscale: frozen (={config.model.get('output_lengthscale', 1.0)})")

    # --- Optimizer ---
    lr = float(config.training.get("learning_rate", 2e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training parameters ---
    max_iter = int(config.training.get("max_iter", 1000))
    batch_size = config.training.get("batch_size", None)
    y_batch_size = config.training.get("y_batch_size", None)
    scale_kl_nm = config.model.get("scale_kl_NM", True)
    seed = int(config.seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Resume from checkpoint ---
    start_step = 0
    prev_n_iterations = 0
    losses = []

    Z_mus = []             # (n_snapshots, L, M)
    Z_lengthscales = []    # (n_snapshots,)
    Z_steps = []           # (n_snapshots,)
    checkpoint_path = save_dir / "model_final.pt"

    if resume:
        if not checkpoint_path.exists():
            print("  Note: no checkpoint found, training from scratch.")
            resume = False
        else:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            start_step = ckpt.get("steps", 0)

            training_json_path = output_dir / "training.json"
            if training_json_path.exists():
                with open(training_json_path) as f:
                    prev_meta = json.load(f)
                prev_n_iterations = prev_meta.get("n_iterations", 0)

            print(f"Resumed from step {start_step}"
                  f"  (cumulative: {prev_n_iterations} prev + {max_iter} new)")

    # SF convention: max_iter is new steps per segment, not cumulative total
    total_steps = start_step + max_iter

    def _make_scheduler(opt, ts):
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=ts,
            pct_start=0.3, div_factor=25.0,
            final_div_factor=1e4, cycle_momentum=False,
        )

    scheduler = _make_scheduler(optimizer, total_steps)

    # --- Data tensors ---
    X = data.X        # (N,)
    y = data.Y        # (N, D)  — note: rows=bins, cols=features/replicates
    C = data.C        # (N,) or None

    # Save original KNN indices on CPU for easy slicing during training
    if prior_type == "LCGP":
        knn_idx_full = model.gp.knn_idx.clone()

    # --- Training loop ---
    t0 = time.perf_counter()

    steps_to_run = total_steps - start_step
    pbar = trange(steps_to_run, desc="Training")
    for i in pbar:
        step = start_step + i
        # ---- Sample batches ----
        if batch_size is not None:
            x_bs = min(batch_size, N)
            idx = torch.multinomial(torch.ones(N), num_samples=x_bs, replacement=False)
            X_batch = X[idx].to(device)
            y_batch = y[:, idx].to(device)
            C_batch = C[idx].to(device) if C is not None else None
        else:
            idx = None
            x_bs = N
            X_batch = X.to(device)
            y_batch = y.to(device)
            C_batch = C.to(device) if C is not None else None

        if y_batch_size is not None:
            y_bs = min(y_batch_size, D)
            idy = torch.multinomial(torch.ones(D), num_samples=y_bs, replacement=False)
            y_batch = y_batch[idy]
        else:
            y_bs = D

        # ---- Forward pass ----
        optimizer.zero_grad()

        # MGGP_SVGP / MGGP_LCGP requires groupsX; SVGP ignores it via **kwargs
        fwd_kwargs = {}
        if use_groups:
            fwd_kwargs["groupsX"] = C_batch

        if prior_type == "LCGP":
            if idx is not None:
                model.gp.knn_idx = knn_idx_full[idx.cpu()]
            else:
                model.gp.knn_idx = knn_idx_full
            # Pass idx to utilize forward_train local optimization
            spatial_idx = idx
        else:
            spatial_idx = None

        pY, qZ, qU, pU = model(X_batch.squeeze(), idx=spatial_idx, **fwd_kwargs)

        y_norm = y_batch - y_batch.mean(dim=1, keepdims=True)

        L1 = pY.log_prob(y_norm).sum()
        if batch_size is not None:
            L1 = L1 * N / x_bs
        if y_batch_size is not None:
            L1 = L1 * D / y_bs

        if prior_type == "LCGP":
            L2 = model.gp.kl_divergence_full(qZ=None, idx=idx).sum()
        else:
            L2 = model.gp.kl_divergence(qU, pU).sum()
            if scale_kl_nm:
                L2 = L2 * N / M

        elbo = L1 - L2
        loss = -elbo
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(elbo.item())

        # Save variational parameters at checkpoint steps.
        # Only mu (L, M) and lengthscale — the posterior mean q(f*|X)
        # depends on mu + kernel, not on Lu. Full (N, L) positions are
        # reconstructed at figure time via model.gp(X).
        if step % 100 == 0 or step == total_steps - 1:
            Z_mus.append(model.gp.mu.detach().cpu().clone().numpy())
            if train_ls:
                Z_lengthscales.append(model.gp.kernel.lengthscale.item())
            Z_steps.append(step)

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"ELBO": f"{elbo.item():.3f}", "lr": f"{current_lr:.1e}"})

    train_time = time.perf_counter() - t0

    # --- Save checkpoint ---
    steps_completed = total_steps - start_step
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "steps": total_steps,               # cumulative total after this segment
        "config": config.to_dict(),
        "batch_size": batch_size,
        "y_batch_size": y_batch_size,
        "train_time": train_time,
    }
    torch.save(checkpoint, checkpoint_path)
    # Save trajectory as compact mu + optional lengthscale
    traj_data = {
        "mu": np.stack(Z_mus),              # (n_snapshots, L, M)
        "steps": np.array(Z_steps),          # (n_snapshots,)
    }
    if train_ls:
        traj_data["lengthscale"] = np.array(Z_lengthscales)
    traj_path = save_dir / "trajectory.npz"
    if resume and traj_path.exists():
        _append_trajectory(traj_data, traj_path)
    else:
        np.savez(traj_path, **traj_data)

    # --- Save ELBO history (SF convention) ---
    if resume:
        _append_elbo_history(losses, output_dir)
    else:
        _save_elbo_history(losses, output_dir)

    # --- Save training.json (SF convention) ---
    metadata = {
        "n_bins": N,
        "n_features": D,
        "n_components": L,
        "final_loss": float(losses[-1]),
        "training_time": train_time,
        "max_iter": max_iter,
        "steps_completed": steps_completed,
        "converged": False,
        "n_iterations": prev_n_iterations + steps_completed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": dict(config.model),
        "training_config": dict(config.training),
        "data_info": {
            "n_bins": N,
            "n_features": D,
            "n_groups": data.n_groups,
        },
    }
    if resume:
        metadata["resumed"] = True
    with open(output_dir / "training.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config.save_yaml(output_dir / "config.yaml")

    print(f"\nTraining complete. {total_steps - start_step} steps in {train_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    if batch_size is not None:
        print(f"  Batched: x={batch_size}" +
              (f", y={y_batch_size}" if y_batch_size is not None else ""))
