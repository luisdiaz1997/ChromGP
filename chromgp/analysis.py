"""Analysis metrics for ChromGP models.

SCC (Stratum-adjusted Correlation Coefficient) from Yang et al. 2017 (HiCRep)
and FISH validation (Procrustes RMSD + pairwise-distance Spearman) following
the Tuzhilina/Hastie PoisMS convention.
"""

import numpy as np
from scipy.stats import spearmanr


def scc(
    contact_raw: np.ndarray,
    positions: np.ndarray,
    resolution: int = 25000,
    min_stratum: int = 3,
    max_stratum_kb: int | None = None,
) -> dict:
    """Stratum-adjusted Correlation Coefficient (Yang et al. 2017).

    Stratifies contact pairs by genomic distance, computes Spearman r between
    observed contacts and predicted (negative Euclidean) distances within each
    stratum, then aggregates by stratum size. This isolates 3D structural
    agreement from the trivial shared distance-decay.

    Args:
        contact_raw: (N, N) raw untransformed contact matrix (upper triangle used).
        positions: (N, L) 3D positions from the model posterior mean.
        resolution: bp per bin (default 25000).
        min_stratum: minimum bin-index difference to include (avoids diagonal noise).
        max_stratum_kb: maximum genomic distance in kb to include (default: all).

    Returns:
        Dict with ``scc`` (float) and ``strata`` (list of {d_kb, n_pairs, r, p}).
    """
    N = contact_raw.shape[0]
    if N < 2:
        return {"scc": float("nan"), "strata": []}

    # Pairwise Euclidean distances from 3D positions
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)  # (N, N)

    # Determine max stratum in bin-index units
    max_d = N - 1
    if max_stratum_kb is not None:
        max_d = min(max_d, max_stratum_kb * 1000 // resolution)

    strata = []
    weighted_sum = 0.0
    total_pairs = 0

    for d in range(min_stratum, max_d + 1):
        # Indices where |i - j| = d in the upper triangle
        rows = np.arange(N - d)
        cols = rows + d

        c = contact_raw[rows, cols]
        neg_dist = -dist[rows, cols]

        # Skip strata with no valid data
        valid = np.isfinite(c) & np.isfinite(neg_dist)
        if valid.sum() < 3:
            continue

        r, p = spearmanr(c[valid], neg_dist[valid])
        n_pairs = valid.sum()

        strata.append({
            "d_kb": d * resolution // 1000,
            "n_pairs": int(n_pairs),
            "r": float(r),
            "p": float(p),
        })
        weighted_sum += r * n_pairs
        total_pairs += n_pairs

    scc_val = weighted_sum / total_pairs if total_pairs > 0 else float("nan")

    return {"scc": float(scc_val), "strata": strata}


def scc_groupwise(
    contact_raw: np.ndarray,
    positions: np.ndarray,
    group_labels: np.ndarray,
    group_names: list[str],
    resolution: int = 25000,
    **kwargs,
) -> dict:
    """Per-group SCC using within-group bin pairs only.

    For each group g, only pairs (i, j) where both bins belong to group g
    are used. This measures how well each chromatin state's 3D structure
    is reconstructed independently of other states.

    Args:
        contact_raw: (N, N) raw contact matrix.
        positions: (N, L) 3D positions.
        group_labels: (N,) integer group labels.
        group_names: list of group name strings.
        resolution: bp per bin.

    Returns:
        Dict mapping ``group_{g}`` → SCC result dict, plus ``overall`` SCC.
    """
    results = {}
    for g in np.unique(group_labels):
        mask = group_labels == g
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue
        sub_contacts = contact_raw[np.ix_(idx, idx)]
        sub_positions = positions[idx]
        results[f"group_{g}"] = scc(sub_contacts, sub_positions, resolution=resolution, **kwargs)
        results[f"group_{g}"]["name"] = group_names[int(g)] if int(g) < len(group_names) else str(g)

    results["overall"] = scc(contact_raw, positions, resolution=resolution, **kwargs)
    return results


def _mds_embed(dist: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Classical (Torgerson) MDS: embed a symmetric distance matrix into R^n.

    Used to recover 3D reference coordinates from the FISH median-distance
    matrix (the per-cell coordinates are defined only up to a rigid motion,
    so we work from distances). The embedding is up to rotation/reflection,
    which the downstream Procrustes step resolves.
    """
    D = np.asarray(dist, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order][:n_components], a_min=0.0, a_max=None)
    eigvecs = eigvecs[:, order][:, :n_components]
    return eigvecs * np.sqrt(eigvals)


def _procrustes_rmsd(
    A: np.ndarray, B: np.ndarray, allow_reflection: bool = True
) -> tuple[float, np.ndarray]:
    """Best-fit rigid alignment (rotation + translation, no scale) of A → B.

    Returns the RMSD after alignment and the aligned copy of A. If
    ``allow_reflection`` (default), uses the full orthogonal group O(3);
    otherwise restricts to SO(3). FISH distance matrices are invariant to
    reflection, so the default matches Wang 2016 / Tuzhilina convention.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: A={A.shape} vs B={B.shape}")
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    Ac = A - a_mean
    Bc = B - b_mean
    M = Ac.T @ Bc
    U, _S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = U @ Vt
    A_aligned = Ac @ R + b_mean
    rmsd = float(np.sqrt(((A_aligned - B) ** 2).sum(axis=1).mean()))
    return rmsd, A_aligned


def fish_validation(
    fish_distance: np.ndarray,
    probe_positions: np.ndarray,
    min_pair_separation: int = 1,
) -> dict:
    """Compare probe-level FISH distances vs probe-level model distances.

    This is the PoisMS / DBMS convention [Tuzhilina, Hastie & Segal 2022,
    2024]: per-probe 3D positions are obtained by averaging the model's
    coordinates over all Hi-C bins inside the probe footprint, and the
    headline metric is the **Spearman correlation** between FISH median
    pairwise distances and predicted pairwise distances over the upper
    triangle of probe pairs.

    Pearson on log distances is reported as a secondary number (more
    sensitive to distance-amplitude calibration than Spearman) and a
    Procrustes RMSD against a 3D MDS embedding of the FISH distance matrix
    is reported as an auxiliary visualization-aid number — both are
    diagnostics, not headline.

    Args:
        fish_distance: ``(P, P)`` FISH median pairwise distance matrix
            (microns). NaNs on the diagonal are tolerated.
        probe_positions: ``(P, 3)`` per-probe 3D coordinates from the model,
            obtained by averaging bin-level coordinates inside each probe's
            genomic footprint. Probes with no overlapping bins should be
            passed as NaN rows and will be excluded.
        min_pair_separation: minimum ``|i-j|`` (probe index) included.
            Default 1 (skip self-pairs only).

    Returns:
        Dict with ``pairwise_spearman`` (headline), ``log_pairwise_pearson``,
        ``procrustes_rmsd_unitscaled``, ``n_probes_used``, ``n_pairs_used``.
    """
    fish_distance = np.asarray(fish_distance, dtype=float)
    probe_positions = np.asarray(probe_positions, dtype=float)
    if probe_positions.shape[0] != fish_distance.shape[0]:
        raise ValueError(
            f"row count mismatch: probe_positions has {probe_positions.shape[0]}, "
            f"fish_distance has {fish_distance.shape[0]}"
        )

    valid = np.all(np.isfinite(probe_positions), axis=1)
    if valid.sum() < 4:
        return {
            "pairwise_spearman": float("nan"),
            "log_pairwise_pearson": float("nan"),
            "procrustes_rmsd_unitscaled": float("nan"),
            "n_probes_used": int(valid.sum()),
            "n_pairs_used": 0,
        }

    probe_xyz = probe_positions[valid]
    fish_used = fish_distance[np.ix_(valid, valid)]
    P = probe_xyz.shape[0]

    diff = probe_xyz[:, None, :] - probe_xyz[None, :, :]
    chromgp_dist = np.linalg.norm(diff, axis=-1)

    iu = np.triu_indices(P, k=min_pair_separation)
    f_vals = fish_used[iu]
    c_vals = chromgp_dist[iu]
    mask = np.isfinite(f_vals) & np.isfinite(c_vals) & (f_vals > 0) & (c_vals > 0)
    n_pairs = int(mask.sum())

    if n_pairs < 3:
        rho = float("nan")
        log_pearson = float("nan")
    else:
        rho_res = spearmanr(f_vals[mask], c_vals[mask])
        rho = float(rho_res.correlation if hasattr(rho_res, "correlation") else rho_res[0])
        lf = np.log(f_vals[mask])
        lc = np.log(c_vals[mask])
        log_pearson = float(np.corrcoef(lf, lc)[0, 1])

    # Auxiliary Procrustes RMSD on unit-RMS-rescaled point clouds.
    fish_sym = np.where(np.isnan(fish_used), 0.0, fish_used)
    fish_sym = 0.5 * (fish_sym + fish_sym.T)
    np.fill_diagonal(fish_sym, 0.0)
    fish_xyz = _mds_embed(fish_sym, n_components=3)

    def _unit_scale(pts):
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        rms = np.sqrt((d[np.triu_indices_from(d, k=1)] ** 2).mean())
        return pts / rms if rms > 0 else pts

    rmsd, _ = _procrustes_rmsd(
        _unit_scale(probe_xyz), _unit_scale(fish_xyz), allow_reflection=True
    )

    return {
        "pairwise_spearman": rho,
        "log_pairwise_pearson": log_pearson,
        "procrustes_rmsd_unitscaled": rmsd,
        "n_probes_used": int(P),
        "n_pairs_used": n_pairs,
    }
