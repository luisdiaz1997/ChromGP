"""Preprocess command for Hi-C data."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..datasets import HiCLoader, GenomicData


def _filter_nans(data: GenomicData) -> GenomicData:
    """Remove bins with NaN in coordinates, signal, or groups.

    For Hi-C data:
    - NaN in X (bin midpoints): invalid genomic position
    - NaN in Y (contact vectors): bins with no contact data
    - Negative group codes indicate missing categories
    """
    mask = np.ones(data.n_bins, dtype=bool)

    # NaN in bin positions
    X_np = data.X.numpy()
    mask &= ~np.isnan(X_np)

    # NaN in signal matrix (any column)
    Y_np = data.Y.numpy()
    mask &= ~np.any(np.isnan(Y_np), axis=1)

    # Negative group codes indicate missing categories
    if data.C is not None:
        C_np = data.C.numpy()
        mask &= C_np >= 0

    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"  Removed {n_removed} bins with NaN values")

    # Chain NaN mask onto the full-length valid_mask (which already has HiCLoader keep)
    if data.valid_mask is not None:
        full_mask = np.zeros(len(data.valid_mask), dtype=bool)
        full_mask[data.valid_mask.numpy()] = mask
        valid_mask = torch.from_numpy(full_mask)
    else:
        valid_mask = torch.from_numpy(mask)

    if mask.all():
        return GenomicData(
            X=data.X, Y=data.Y, C=data.C,
            n_groups=data.n_groups, group_names=data.group_names,
            gc=data.gc,
            contact_raw=data.contact_raw,
            contact_raw_full=data.contact_raw_full,
            valid_mask=valid_mask,
            bin_coords=data.bin_coords,
            metadata=data.metadata,
        )

    t_mask = torch.from_numpy(mask)
    return GenomicData(
        X=data.X[t_mask],
        Y=data.Y[t_mask],
        C=data.C[t_mask] if data.C is not None else None,
        n_groups=data.n_groups,
        group_names=data.group_names,
        gc=data.gc[t_mask] if data.gc is not None else None,
        contact_raw=data.contact_raw[t_mask][:, t_mask] if data.contact_raw is not None else None,
        contact_raw_full=data.contact_raw_full,
        valid_mask=valid_mask,
        bin_coords=data.bin_coords.iloc[mask].reset_index(drop=True) if data.bin_coords is not None else None,
        metadata=data.metadata,
    )


def run(config_path: str):
    """Preprocess Hi-C dataset and save standardized files.

    Output files (following SF convention):
    - X.npy: (N,) bin midpoints
    - Y.npy: (N, D) signal matrix
    - C.npy: (N,) group codes (int64, 0..G-1)
    - contact_raw.npy: (N, N) raw contacts for analyze-stage metrics
    - metadata.json: n_bins, n_features, n_groups, group_names, preprocessing params

    Parameters
    ----------
    config_path : str
        Path to config YAML file.
    """
    # Load config
    config = Config.from_yaml(config_path)

    print(f"Preprocessing dataset: {config.dataset}")

    # Load raw data using HiCLoader
    loader = HiCLoader()
    data = loader.load(config.preprocessing)

    print(f"  Loaded: {data.n_bins} bins × {data.n_features} features")
    if data.n_groups > 0:
        print(f"  Groups: {data.n_groups} ({', '.join(data.group_names[:5])}{'...' if len(data.group_names) > 5 else ''})")

    # Remove bins with NaN values
    data = _filter_nans(data)
    print(f"  After NaN filter: {data.n_bins} bins")

    # Create output directory for preprocessed data
    # Output path: outputs/<dataset>/<region_slug>/preprocessed/  (shared across models)
    region_slug = config.preprocessing.get('region', 'unknown').replace(':', '_')
    output_dir = Path(config.output_dir) / region_slug / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays in standardized format
    np.save(output_dir / "X.npy", data.X.numpy())  # (N,)
    np.save(output_dir / "Y.npy", data.Y.numpy())  # (N, D)
    np.save(output_dir / "contact_raw.npy", data.contact_raw.numpy())  # (N, N)
    if data.contact_raw_full is not None:
        np.save(output_dir / "contact_raw_full.npy", data.contact_raw_full.numpy())  # (N_full, N_full) with NaN gaps
    if data.valid_mask is not None:
        np.save(output_dir / "valid_mask.npy", data.valid_mask.numpy())  # (N_full,) bool

    # Save GC content per bin
    if data.gc is not None:
        np.save(output_dir / "gc.npy", data.gc.numpy())  # (N,)

    # Save group codes (C)
    if data.C is not None:
        np.save(output_dir / "C.npy", data.C.numpy())  # (N,)
    else:
        # No groups - create single group (all zeros)
        np.save(output_dir / "C.npy", np.zeros(data.n_bins, dtype=np.int64))

    # Build metadata (matching SF's structure)
    group_names = data.group_names if data.group_names else ["All"]

    metadata = {
        "n_bins": data.n_bins,
        "n_features": data.n_features,
        "n_groups": data.n_groups if data.n_groups > 0 else 1,
        "group_names": group_names,
        "dataset": config.dataset,
        "model": config.model,
        "preprocessing": config.preprocessing,
        "training": config.training,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Report sizes
    x_size = (output_dir / "X.npy").stat().st_size / 1e6
    y_size = (output_dir / "Y.npy").stat().st_size / 1e6
    c_size = (output_dir / "C.npy").stat().st_size / 1e6
    contact_size = (output_dir / "contact_raw.npy").stat().st_size / 1e6
    gc_size = (output_dir / "gc.npy").stat().st_size / 1e6 if (output_dir / "gc.npy").exists() else 0

    print(f"Preprocessed data saved to: {output_dir}")
    print(f"  X: {data.X.shape} ({x_size:.1f} MB)")
    print(f"  Y: {data.Y.shape} ({y_size:.1f} MB)")
    print(f"  C: {metadata['n_groups']} groups ({c_size:.1f} MB)")
    print(f"  contact_raw: {data.contact_raw.shape} ({contact_size:.1f} MB)")
    if gc_size:
        print(f"  gc: {data.gc.shape} ({gc_size:.1f} MB)")
