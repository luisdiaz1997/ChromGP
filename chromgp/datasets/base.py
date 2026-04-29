"""Base classes and data containers for ChromGP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class GenomicData:
    """Container for genomic data (Hi-C, ChIP-seq, ATAC-seq, etc.).

    Attributes
    ----------
    X : torch.Tensor
        Bin midpoints (genomic coordinates), shape (N,).
    Y : torch.Tensor
        Signal matrix, shape (N, D) where D is number of replicates or features.
        For Hi-C: columns are flattened contact vectors per replicate.
        For ChIP-seq: columns are histone marks.
    C : torch.Tensor, optional
        Group labels for MGGP models, shape (N,).
        Integer-encoded (0, 1, 2, ..., G-1).
    n_groups : int
        Number of unique groups (0 if no groups).
    group_names : list of str, optional
        Names of groups (length G), e.g., ['Tss', 'Enh1', 'Het', ...].
    contact_raw : torch.Tensor, optional
        Raw (untransformed) contact matrix, shape (N, N). Kept for analyze-stage metrics.
    bin_coords : pandas.DataFrame, optional
        Bin coordinates with chrom, start, end columns.
    metadata : dict, optional
        Additional metadata (resolution, region, transform, etc.).
    """

    X: torch.Tensor  # (N,) bin midpoints
    Y: torch.Tensor  # (N, D) signal matrix
    C: Optional[torch.Tensor] = None  # (N,) group labels
    n_groups: int = 0
    group_names: Optional[List[str]] = None
    gc: Optional[torch.Tensor] = None  # (N,) GC fraction per bin
    contact_raw: Optional[torch.Tensor] = None  # (N, N) raw contacts
    bin_coords: Optional[object] = None  # DataFrame with chrom, start, end
    metadata: dict = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features (replicates/mark columns)."""
        return self.Y.shape[1]

    def to(self, device: torch.device) -> "GenomicData":
        """Move all tensors to specified device."""
        return GenomicData(
            X=self.X.to(device),
            Y=self.Y.to(device),
            C=self.C.to(device) if self.C is not None else None,
            n_groups=self.n_groups,
            group_names=self.group_names,
            gc=self.gc.to(device) if self.gc is not None else None,
            contact_raw=self.contact_raw.to(device) if self.contact_raw is not None else None,
            bin_coords=self.bin_coords,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"GenomicData(n_bins={self.n_bins}, n_features={self.n_features}, "
            f"n_groups={self.n_groups})"
        )
