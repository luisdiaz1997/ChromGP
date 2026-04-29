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
    from gpzoo.kernels import batched_RBF
    from gpzoo.modules import CholeskyParameter

    L = config.model.get("n_components", 3)
    M = config.model.get("num_inducing", 800)
    jitter = float(config.model.get("jitter", 1e-5))
    noise = float(config.model.get("noise", 0.1))
    ls = float(config.model.get("lengthscale", 8.0))
    out_ls = float(config.model.get("output_lengthscale", 1.0))
    sigma = float(config.model.get("sigma", 1.0))
    cholesky_mode = config.model.get("cholesky_mode", "exp")

    svgp_kernel = batched_RBF(sigma=sigma, lengthscale=ls)
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
    from gpzoo.kernels import batched_MGGP_RBF, batched_RBF
    from gpzoo.modules import CholeskyParameter

    L = config.model.get("n_components", 3)
    M = config.model.get("num_inducing", 800)
    jitter = float(config.model.get("jitter", 1e-5))
    noise = float(config.model.get("noise", 0.1))
    ls = float(config.model.get("lengthscale", 8.0))
    out_ls = float(config.model.get("output_lengthscale", 1.0))
    sigma = float(config.model.get("sigma", 1.0))
    cholesky_mode = config.model.get("cholesky_mode", "exp")
    group_diff_param = float(config.model.get("group_diff_param", 1.0))

    # MGGP kernel: same as RBF but covariance modulated by group distance
    mggp_kernel = batched_MGGP_RBF(
        sigma=sigma, lengthscale=ls,
        group_diff_param=group_diff_param, n_groups=n_groups,
    )

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
    output_dir = Path(config.output_dir) / region_slug / model_name
    save_dir = output_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_preprocessed(output_dir)
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

    if use_groups:
        model = build_model_mggp_svgp(
            config, X=data.X, C=data.C, n_groups=data.n_groups,
        )
        print(f"Model: ChromGP + MGGP_SVGP  (M={M}, L={L}, G={data.n_groups})")
        print(f"  group_diff_param: {config.model.get('group_diff_param', 1.0)}")
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

    # --- Scheduler (OneCycleLR, SF convention) ---
    total_steps = int(config.training.get("max_iter", 1000))

    def _make_scheduler(opt, ts):
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=ts,
            pct_start=0.3, div_factor=25.0,
            final_div_factor=1e4, cycle_momentum=False,
        )

    scheduler = _make_scheduler(optimizer, total_steps)

    # --- Training parameters ---
    batch_size = config.training.get("batch_size", None)
    y_batch_size = config.training.get("y_batch_size", None)
    scale_kl_nm = config.model.get("scale_kl_NM", True)
    seed = int(config.seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Resume from checkpoint ---
    start_step = 0
    losses = []
    Zs = []
    checkpoint_path = save_dir / "model_final.pt"

    if resume:
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_step = ckpt.get("steps", 0)
            losses = ckpt.get("losses", [])

            scheduler = _make_scheduler(optimizer, total_steps)
            scheduler.last_epoch = start_step

            print(f"Resumed from step {start_step}/{total_steps}")

            if start_step >= total_steps:
                print("  Model already reached max_iter. Increase max_iter to continue training.")
                return
        else:
            print("No checkpoint found, training from scratch.")

    # --- Data tensors ---
    X = data.X        # (N,)
    y = data.Y        # (N, D)  — note: rows=bins, cols=features/replicates
    C = data.C        # (N,) or None

    # --- Training loop ---
    t0 = time.perf_counter()

    pbar = trange(start_step, total_steps, desc="Training", initial=start_step)
    for step in pbar:
        # ---- Sample batches ----
        if batch_size is not None:
            x_bs = min(batch_size, N)
            idx = torch.multinomial(torch.ones(N), num_samples=x_bs, replacement=False)
            X_batch = X[idx].to(device)
            y_batch = y[:, idx].to(device)
            C_batch = C[idx].to(device) if C is not None else None
        else:
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

        # MGGP_SVGP requires groupsX; SVGP ignores it via **kwargs
        fwd_kwargs = {}
        if use_groups:
            fwd_kwargs["groupsX"] = C_batch

        pY, qZ, qU, pU = model(X_batch.squeeze(), **fwd_kwargs)

        y_norm = y_batch - y_batch.mean(dim=1, keepdims=True)

        L1 = pY.log_prob(y_norm).sum()
        if batch_size is not None:
            L1 = L1 * N / x_bs
        if y_batch_size is not None:
            L1 = L1 * D / y_bs

        L2 = model.gp.kl_divergence(qU, pU).sum()
        if scale_kl_nm:
            L2 = L2 * N / M

        loss = -(L1 - L2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 100 == 0 or step == total_steps - 1:
            Zs.append(qZ.mean.T.detach().cpu().clone().numpy())

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"Loss": f"{loss.item():.3f}", "lr": f"{current_lr:.1e}"})

    train_time = time.perf_counter() - t0

    # --- Save checkpoint ---
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
        "losses": losses,
        "steps": total_steps,
        "batch_size": batch_size,
        "y_batch_size": y_batch_size,
        "start_step": start_step,
        "train_time": train_time,
    }
    torch.save(checkpoint, checkpoint_path)
    np.save(save_dir / "trajectory.npy", np.array(Zs, dtype=object))

    # --- Save ELBO history (SF convention) ---
    df = pd.DataFrame({"iteration": range(len(losses)), "elbo": losses})
    df.to_csv(output_dir / "elbo_history.csv", index=False)
    np.save(output_dir / "elbo_history.npy", np.array(losses))

    # --- Save training.json (SF convention) ---
    metadata = {
        "n_bins": N,
        "n_features": D,
        "n_components": L,
        "final_loss": float(losses[-1]),
        "training_time": train_time,
        "max_iter": total_steps,
        "steps_completed": total_steps - start_step,
        "converged": False,
        "n_iterations": total_steps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_config": dict(config.model),
        "training_config": dict(config.training),
        "data_info": {
            "n_bins": N,
            "n_features": D,
            "n_groups": data.n_groups,
        },
    }
    with open(output_dir / "training.json", "w") as f:
        json.dump(metadata, f, indent=2)

    config.save_yaml(output_dir / "config.yaml")

    print(f"\nTraining complete. {total_steps - start_step} steps in {train_time:.1f}s")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    if batch_size is not None:
        print(f"  Batched: x={batch_size}" +
              (f", y={y_batch_size}" if y_batch_size is not None else ""))
