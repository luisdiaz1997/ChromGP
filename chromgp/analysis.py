"""Analysis metrics for ChromGP models.

SCC (Stratum-adjusted Correlation Coefficient) from Yang et al. 2017 (HiCRep).
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
