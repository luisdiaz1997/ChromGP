"""ChromHMM state annotation functions for MGGP groups."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import bioframe as bf
import numpy as np
import pandas as pd
import torch


# Mapping from raw ChromHMM state codes to 5 coarse biological groups.
# Applied before bin-majority assignment so overlaps are merged first.
#
# Group       | States
# ------------|--------------------------------------------------------
# Active      | Active TSS, TSS Flanking, Upstream/Downstream Flank,
#             | Active Enhancer 1/2, Genic Enhancer G1/G2
# Transcribed | Transcribed, Weak Transcribed
# Heterochromatin | Heterochromatin, ZNF/Repeats
# Polycomb    | Polycomb Repressed, Bivalent
# Quiescent   | Quiescent
CHROMHMM_MERGE_MAP: dict[str, str] = {
    # Active regulatory
    'Tss':      'Active',
    'TssFlnk':  'Active',
    'TssFlnkD': 'Active',
    'TssFlnkU': 'Active',
    'Enh1':     'Active',
    'Enh2':     'Active',
    'EnhG1':    'Active',
    'EnhG2':    'Active',
    # 18-state equivalents
    'TssA':     'Active',
    'EnhA1':    'Active',
    'EnhA2':    'Active',
    'EnhWk':    'Active',
    # Transcribed
    'Tx':       'Transcribed',
    'TxWk':     'Transcribed',
    # Heterochromatin
    'Het':          'Heterochromatin',
    'ZNF/Rpts':     'Heterochromatin',
    # Polycomb / repressed
    'ReprPC':   'Polycomb',
    'ReprPCWk': 'Polycomb',
    'Biv':      'Polycomb',
    'TssBiv':   'Polycomb',
    'EnhBiv':   'Polycomb',
    # Quiescent
    'Quies':    'Quiescent',
}


def merge_chromhmm_groups(chromhmm_df: pd.DataFrame) -> pd.DataFrame:
    """Replace fine-grained state codes with 5 coarse biological groups.

    Unknown states fall back to 'Quiescent'. Call this before
    assign_chromhmm_states so the bin-majority overlap operates on
    merged groups rather than individual states.

    Parameters
    ----------
    chromhmm_df : pd.DataFrame
        ChromHMM annotations with columns: chrom, start, end, state.

    Returns
    -------
    pd.DataFrame
        Same structure but 'state' column replaced with coarse group name.
    """
    df = chromhmm_df.copy()
    df['state'] = df['state'].map(CHROMHMM_MERGE_MAP).fillna('Quiescent')
    return df


# Descriptive names for ChromHMM states (for pretty figures/labels)
_STATE_DESCRIPTIONS = {
    # 15-state model
    'Tss': 'Active TSS',
    'TssFlnk': 'TSS Flanking',
    'TssFlnkD': 'Downstream Flank',
    'TssFlnkU': 'Upstream Flank',
    'Tx': 'Transcribed',
    'TxWk': 'Weak Transcribed',
    'Enh1': 'Active Enhancer 1',
    'Enh2': 'Active Enhancer 2',
    'EnhG1': 'Genic Enhancer 1',
    'EnhG2': 'Genic Enhancer 2',
    'ZNF/Rpts': 'ZNF/Repeats',
    'Het': 'Heterochromatin',
    'ReprPC': 'Polycomb Repressed',
    'Biv': 'Bivalent',
    'Quies': 'Quiescent',
    # 18-state model
    'TssA': 'Active TSS',
    'TssBiv': 'Bivalent TSS',
    'EnhA1': 'Active Enhancer A1',
    'EnhA2': 'Active Enhancer A2',
    'EnhG1': 'Genic Enhancer G1',
    'EnhG2': 'Genic Enhancer G2',
    'EnhWk': 'Weak Enhancer',
    'EnhBiv': 'Bivalent Enhancer',
    'ReprPCWk': 'Weak Polycomb',
}


def load_chromhmm_bed(bed_path: Union[Path, str], state_whitelist: Optional[List[str]] = None) -> pd.DataFrame:
    """Load ChromHMM BED file into a DataFrame.

    Parameters
    ----------
    bed_path : Path or str
        Path to ChromHMM BED file (standard 4-9 column BED format).
        Expected columns: chrom, start, end, name, score, strand, thickStart, thickEnd, itemRgb
        The 4th column (name) contains the state name.
    state_whitelist : list of str, optional
        If provided, only these states are used.

    Returns
    -------
    pd.DataFrame
        ChromHMM annotations with columns: chrom, start, end, state
    """
    chromhmm = pd.read_csv(
        bed_path,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'state', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
    )

    # Filter to whitelist if provided
    if state_whitelist is not None:
        chromhmm = chromhmm[chromhmm['state'].isin(state_whitelist)]

    return chromhmm[['chrom', 'start', 'end', 'state']]


def assign_chromhmm_states(bins: pd.DataFrame, chromhmm: pd.DataFrame) -> torch.Tensor:
    """Assign chromatin states to bins using dominant overlap.

    Each bin is assigned the chromatin state that covers the most base pairs
    within that bin. Bins with no overlap are assigned the most common
    genome-wide state.

    Parameters
    ----------
    bins : pandas.DataFrame
        Bin coordinates with columns: chrom, start, end.
        One row per bin.
    chromhmm : pandas.DataFrame
        ChromHMM annotations with columns: chrom, start, end, state.

    Returns
    -------
    torch.Tensor
        Group labels, shape (N,). Integer-encoded states (0, 1, 2, ...).
    """
    # Get unique states and create mapping
    states = sorted(chromhmm['state'].unique())
    state_to_id = {state: i for i, state in enumerate(states)}

    # Assign a clean local index (0..N-1) — cooler bins have global genome-wide indices
    local_bins = bins.reset_index(drop=True)
    local_bins['bin_index'] = np.arange(len(local_bins))

    # Find overlaps between bins and ChromHMM segments
    overlaps = bf.overlap(
        local_bins,
        chromhmm,
        how='left',
        return_overlap=True,
        cols1=['chrom', 'start', 'end'],
        cols2=['chrom', 'start', 'end']
    )

    # Calculate overlap bp for each pair
    overlaps['overlap_bp'] = overlaps['overlap_end'] - overlaps['overlap_start']

    # Drop rows with no overlap (chrom_ will be NaN) or mismatched chromosomes
    overlaps = overlaps.dropna(subset=['state_'])

    # For each bin, pick the state with largest overlap
    dominant_idx = overlaps.loc[overlaps.groupby('bin_index')['overlap_bp'].idxmax()]

    # Get the state for each bin
    C = np.full(len(bins), -1, dtype=np.int64)
    state_col = [c for c in overlaps.columns if c.endswith('state_')][0]

    for _, row in dominant_idx.iterrows():
        C[int(row['bin_index'])] = state_to_id[row[state_col]]

    # Assign bins with no overlap to the most common state
    unassigned = C == -1
    if unassigned.sum() > 0:
        most_common_state = chromhmm['state'].value_counts().idxmax()
        most_common_id = state_to_id[most_common_state]
        C[unassigned] = most_common_id

    return torch.from_numpy(C)


def get_state_names(chromhmm: pd.DataFrame) -> List[str]:
    """Get sorted list of unique chromatin state names (descriptive).

    Always returns descriptive names for pretty figures/labels.
    Example: ['Active TSS', 'Active Enhancer 1', 'Heterochromatin', ...]

    Parameters
    ----------
    chromhmm : pandas.DataFrame
        ChromHMM annotations with 'state' column.

    Returns
    -------
    list of str
        Sorted descriptive state names.
    """
    state_codes = sorted(chromhmm['state'].unique())
    return [_STATE_DESCRIPTIONS.get(s, s) for s in state_codes]
