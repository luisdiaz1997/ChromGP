"""Hi-C data loader.

This module handles loading .mcool files and extracting contact matrices
for specified regions and resolutions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cooler
import numpy as np
import pandas as pd
import torch

from .base import GenomicData
from . import chromhmm


class HiCLoader:
    """Loader for Hi-C data from .mcool files.

    Usage:
        loader = HiCLoader()
        data = loader.load({
            'mcool_path': 'path/to/file.mcool',
            'resolution': 25000,
            'region': 'chr14',
            'balance': True,
            'contact_transform': 'log1p',
            ...
        })
    """

    def load(self, preprocessing: dict) -> GenomicData:
        """Load and preprocess Hi-C data.

        Parameters
        ----------
        preprocessing : dict
            Preprocessing parameters:
            - mcool_path: path to .mcool file
            - resolution: bin size in bp
            - region: genomic region ('chr14' or 'chr14:10M-50M')
            - balance: use ICE-balanced contacts
            - contact_transform: 'log1p', 'obs_over_exp', or 'raw'
            - num_replicates: number of synthetic replicates
            - noise_level: noise for synthetic replicates
            - groups_by: 'chromosome' (for multi-region) or 'chromhmm_state'
            - chromhmm_bed: path to ChromHMM BED (if groups_by == 'chromhmm_state')
            - chromhmm_states: optional whitelist of states

        Returns
        -------
        GenomicData
            Loaded and preprocessed Hi-C data.
        """
        # Extract parameters
        mcool_path = preprocessing['mcool_path']
        resolution = preprocessing['resolution']
        region = preprocessing['region']
        balance = preprocessing.get('balance', True)
        contact_transform = preprocessing.get('contact_transform', 'log1p')
        num_replicates = preprocessing.get('num_replicates', 1)
        noise_level = preprocessing.get('noise_level', 0.15)
        groups_by = preprocessing.get('groups_by')

        # Parse region string (chr14 or chr14:start-end)
        chrom, start, end = self._parse_region(region)

        # Load cooler at specified resolution
        clr = cooler.Cooler(f"{mcool_path}::/resolutions/{resolution}")

        # Fetch the contact matrix for the region
        matrix = self._fetch_matrix(clr, chrom, start, end, balance=balance)

        # Keep full matrix with NaN gaps for visualization
        matrix_full = matrix.copy()

        # Remove fully-NaN rows and columns (centromeric / low-mappability regions)
        good_rows = ~np.isnan(matrix).all(axis=1)
        good_cols = ~np.isnan(matrix).all(axis=0)
        keep = good_rows & good_cols
        valid_mask = torch.from_numpy(keep)
        matrix = matrix[np.ix_(keep, keep)]

        # Impute remaining sparse NaNs to 0
        matrix = np.nan_to_num(matrix, nan=0.0)

        # Create bin coordinates and filter to matching bins
        bin_coords = self._make_bins(clr, chrom, start, end, resolution)
        bin_coords = bin_coords.iloc[keep].reset_index(drop=True)

        # Compute GC content per bin
        gc = None
        gc_reference = preprocessing.get('gc_reference')
        if gc_reference:
            from .gc import compute_gc
            gc = compute_gc(bin_coords, gc_reference)

        # Apply contact transform
        matrix_transformed = self._apply_transform(matrix, contact_transform, clr, bin_coords)

        # Generate replicates if requested
        Y = self._generate_replicates(matrix_transformed, num_replicates, noise_level)

        # Flatten to vectors and create X (bin midpoints)
        X = torch.from_numpy(bin_coords['mid'].values.copy()).float()
        Y = torch.from_numpy(Y).float()  # (N, D)

        # Keep raw contact matrix for analyze-stage metrics
        contact_raw = torch.from_numpy(matrix).float()
        contact_raw_full = torch.from_numpy(matrix_full).float()

        # Handle groups
        C = None
        n_groups = 0
        group_names = None

        if groups_by == 'chromhmm_state':
            # Load ChromHMM annotations and assign states
            chromhmm_bed = preprocessing.get('chromhmm_bed')
            state_whitelist = preprocessing.get('chromhmm_states')

            if chromhmm_bed is None:
                raise ValueError("chromhmm_bed path required when groups_by == 'chromhmm_state'")

            chromhmm_df = chromhmm.load_chromhmm_bed(chromhmm_bed, state_whitelist)
            C = chromhmm.assign_chromhmm_states(bin_coords, chromhmm_df)
            group_names = chromhmm.get_state_names(chromhmm_df)
            n_groups = len(group_names)
        elif groups_by == 'chromosome':
            # For multi-region configs, each region is a group
            # This will be handled in multi-region loading
            pass

        # Build metadata
        metadata = {
            'resolution': resolution,
            'region': region,
            'chrom': chrom,
            'start': start,
            'end': end,
            'balance': balance,
            'contact_transform': contact_transform,
            'num_replicates': num_replicates,
            'noise_level': noise_level,
            'groups_by': groups_by,
            'n_bins': len(bin_coords),
        }

        return GenomicData(
            X=X,
            Y=Y,
            C=C,
            n_groups=n_groups,
            group_names=group_names,
            gc=gc,
            contact_raw=contact_raw,
            contact_raw_full=contact_raw_full,
            valid_mask=valid_mask,
            bin_coords=bin_coords,
            metadata=metadata,
        )

    @staticmethod
    def _parse_region(region: str) -> tuple[str, Optional[int], Optional[int]]:
        """Parse region string like 'chr14' or 'chr14:1200000-50000000'."""
        if ':' in region:
            chrom, coords = region.split(':')
            start, end = coords.split('-')
            return chrom, int(start), int(end)
        return region, None, None

    @staticmethod
    def _fetch_matrix(clr: cooler.Cooler, chrom: str, start: Optional[int], end: Optional[int], balance: bool = True) -> np.ndarray:
        """Fetch contact matrix for region."""
        if start is None or end is None:
            matrix = clr.matrix(balance=balance).fetch(chrom)
        else:
            matrix = clr.matrix(balance=balance).fetch((chrom, start, end))
        return matrix

    @staticmethod
    def _make_bins(clr: cooler.Cooler, chrom: str, start: Optional[int], end: Optional[int], resolution: int) -> pd.DataFrame:
        """Create bin coordinates DataFrame."""
        if start is None or end is None:
            # Whole chromosome
            chrom_size = clr.chromsizes[chrom]
            bins = clr.bins().fetch(chrom)
        else:
            bins = clr.bins().fetch((chrom, start, end))

        bins = bins[['chrom', 'start', 'end']].copy()
        bins['mid'] = (bins['start'] + bins['end']) // 2
        return bins

    @staticmethod
    def _apply_transform(matrix: np.ndarray, transform: str, clr: cooler.Cooler, bin_coords: pd.DataFrame) -> np.ndarray:
        """Apply contact transform (log1p, obs_over_exp, raw)."""
        if transform == 'raw':
            return matrix
        elif transform == 'log10':
            return np.log10(matrix + 5e-6)
        elif transform == 'obs_over_exp':
            # TODO: implement expected contact calculation
            # For now, return log1p as placeholder
            raise NotImplementedError("obs_over_exp transform not yet implemented")
        else:
            raise ValueError(f"Unknown transform: {transform}")

    @staticmethod
    def _generate_replicates(matrix: np.ndarray, num_replicates: int, noise_level: float) -> np.ndarray:
        """Generate noisy replicates from contact matrix.

        Args:
            matrix: (N, N) contact matrix
            num_replicates: number of replicates to generate
            noise_level: noise standard deviation

        Returns:
            Y: (N, D) where D = num_replicates * N, each replicate's full N×N matrix
                reshaped / concatenated along columns.  With num_replicates=1, Y is (N, N).
                With num_replicates>1, D = num_replicates * N.
        """
        N = matrix.shape[0]

        if num_replicates == 1:
            return matrix

        Y = np.zeros((N, N * num_replicates))
        for d in range(num_replicates):
            noise = noise_level * np.random.randn(N, N)
            # Symmetrize the noise to preserve the symmetry of contacts
            noise = (noise + noise.T) / 2
            Y[:, d * N : (d + 1) * N] = matrix + noise

        return Y
