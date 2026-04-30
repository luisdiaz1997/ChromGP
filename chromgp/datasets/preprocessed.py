"""Load preprocessed Hi-C data from disk.

Fast path to load standardized data without re-processing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .base import GenomicData


def load_preprocessed(output_dir: Union[Path, str]) -> GenomicData:
    """Load preprocessed data from directory.

    After running the preprocess command, this function provides a fast path
    to load the standardized data without re-processing.

    Parameters
    ----------
    output_dir : Path or str
        Output directory containing preprocessed/ subdirectory.

    Returns
    -------
    GenomicData
        Loaded preprocessed data.
    """
    output_dir = Path(output_dir)
    prep_dir = output_dir / "preprocessed"

    if not prep_dir.exists():
        raise ValueError(f"Preprocessed data directory not found: {prep_dir}")

    # Load arrays
    X = torch.from_numpy(np.load(prep_dir / "X.npy"))
    Y = torch.from_numpy(np.load(prep_dir / "Y.npy"))
    contact_raw = torch.from_numpy(np.load(prep_dir / "contact_raw.npy"))

    # Load GC content if available
    gc = None
    if (prep_dir / "gc.npy").exists():
        gc = torch.from_numpy(np.load(prep_dir / "gc.npy"))

    # Load full contact matrix with NaN gaps (for visualization)
    contact_raw_full = None
    if (prep_dir / "contact_raw_full.npy").exists():
        contact_raw_full = torch.from_numpy(np.load(prep_dir / "contact_raw_full.npy"))

    # Load valid-bin mask (N_full bool, True = bin survived all filtering)
    valid_mask = None
    if (prep_dir / "valid_mask.npy").exists():
        valid_mask = torch.from_numpy(np.load(prep_dir / "valid_mask.npy"))

    # Load groups if present
    C = None
    n_groups = 0
    group_names = None
    if (prep_dir / "C.npy").exists():
        C = torch.from_numpy(np.load(prep_dir / "C.npy"))
        n_groups = int(C.max().item()) + 1

    # Load metadata
    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)

    return GenomicData(
        X=X,
        Y=Y,
        C=C,
        n_groups=meta.get("n_groups", n_groups),
        group_names=meta.get("group_names"),
        gc=gc,
        contact_raw=contact_raw,
        contact_raw_full=contact_raw_full,
        valid_mask=valid_mask,
        metadata=meta,
    )
