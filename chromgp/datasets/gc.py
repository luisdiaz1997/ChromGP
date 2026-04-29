"""GC content computation for genomic bins.

Provides a single function to compute per-bin GC fraction given bin coordinates
and a reference genome FASTA. Works with any genomic data (Hi-C, ChIP-seq, etc.)
as long as bin coordinates are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch


def compute_gc(bin_coords: pd.DataFrame, fasta_path: Union[str, Path]) -> torch.Tensor:
    """Compute GC fraction per bin from a reference genome.

    Parameters
    ----------
    bin_coords : pd.DataFrame
        Bin coordinates with columns ['chrom', 'start', 'end'].
    fasta_path : str or Path
        Path to reference genome FASTA file.

    Returns
    -------
    torch.Tensor
        GC fraction per bin, shape (N,), dtype float32.
    """
    import bioframe as bf

    genome = bf.load_fasta(str(fasta_path))
    gc_df = bf.frac_gc(bin_coords[['chrom', 'start', 'end']], genome)
    return torch.from_numpy(gc_df['GC'].values.copy()).float()
