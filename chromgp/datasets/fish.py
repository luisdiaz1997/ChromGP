"""Multiplex-FISH dataset loaders for ChromGP validation.

Source: Wang et al. 2016, *Science* (DOI 10.1126/science.aaf8084), archived on
the 4D Nucleome data portal in FOF-CT v0.1 format. See ``docs/fish.md`` for
provenance, probe table, and citation.

The loader returns a :class:`FISHReference` that exposes:

* per-cell per-probe 3D spot coordinates (microns), with NaN for missing spots,
* per-pair **median** pairwise distance across cells (the standard reference
  matrix used by PoisMS / DBMS),
* per-probe genomic midpoints in GRCh38,
* :meth:`map_to_bins` — maps each probe to the bin index a Hi-C run at a given
  resolution would assign it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_FOF_CT_COLUMNS = ["Spot_ID", "Trace_ID", "X", "Y", "Z",
                   "Chrom", "Chrom_Start", "Chrom_End"]


@dataclass
class FISHReference:
    """Aggregated multiplex-FISH measurements for one chromosome.

    Attributes:
        coords: ``(n_cells, n_probes, 3)`` float array of spot positions
            (microns), with ``np.nan`` for spots not detected in a given cell.
        probe_midpoints: ``(n_probes,)`` int array, GRCh38 midpoint of each
            probe in base pairs, sorted ascending.
        probe_intervals: ``(n_probes, 2)`` int array of (start, end) in bp.
        chrom: chromosome string, e.g. ``"chr21"``.
        assembly: genome assembly, e.g. ``"GRCh38"``.
        source: free-text data source identifier (e.g. 4DN file accession).
    """

    coords: np.ndarray
    probe_midpoints: np.ndarray
    probe_intervals: np.ndarray
    chrom: str
    assembly: str
    source: str

    @property
    def n_cells(self) -> int:
        return self.coords.shape[0]

    @property
    def n_probes(self) -> int:
        return self.coords.shape[1]

    def median_distance_matrix(self) -> np.ndarray:
        """Per-pair median of cell-wise pairwise distances (microns).

        Returns ``(n_probes, n_probes)`` symmetric matrix. NaN entries appear
        only when no cell has both probes observed (rare).
        """
        # Per-cell pairwise distance: (n_cells, n_probes, n_probes)
        diff = self.coords[:, :, None, :] - self.coords[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(dist, axis=0)
        np.fill_diagonal(med, 0.0)
        return med

    def bin_indices_per_probe(
        self, bin_midpoints_bp: np.ndarray, pad_bp: int = 0
    ) -> list[np.ndarray]:
        """Return, for each probe, the indices of bins whose midpoint falls in
        the probe's genomic footprint ``[Chrom_Start - pad_bp, Chrom_End +
        pad_bp]``.

        This is the probe-footprint aggregation step used by PoisMS / DBMS:
        each FISH probe spans ~100 kb (~4 bins at 25 kb), and the 3D
        coordinate to compare against the FISH spot is the *average* of all
        Hi-C bin 3D coordinates inside that footprint.

        Args:
            bin_midpoints_bp: ``(N,)`` int array of Hi-C bin midpoints in bp.
            pad_bp: extra slack on each side, useful when probe intervals are
                narrower than one bin width. Default 0.

        Returns:
            Length ``n_probes`` list. Each entry is an integer index array
            into ``bin_midpoints_bp``. May be empty for probes whose footprint
            falls in a region that was filtered out of the Hi-C run
            (centromere, short arm, etc.).
        """
        bin_midpoints_bp = np.asarray(bin_midpoints_bp, dtype=np.int64)
        starts = self.probe_intervals[:, 0] - int(pad_bp)
        ends = self.probe_intervals[:, 1] + int(pad_bp)
        return [
            np.where((bin_midpoints_bp >= s) & (bin_midpoints_bp <= e))[0]
            for s, e in zip(starts, ends)
        ]

    def map_to_bins(self, resolution: int, region_start: int = 0) -> np.ndarray:
        """Map each probe midpoint to the Hi-C bin index it falls into.

        The convention matches ``chromgp/datasets/hic.py``: ``bin_index =
        (midpoint - region_start) // resolution``.

        Args:
            resolution: Hi-C bin width in bp (e.g. 25000).
            region_start: start coordinate of the Hi-C region in bp. Use 0
                when the whole chromosome was loaded; use the lower bound of
                ``chr21:A-B`` when a sub-region was loaded.

        Returns:
            ``(n_probes,)`` int array of bin indices. Probes outside
            ``[region_start, region_start + n_bins * resolution)`` are flagged
            with ``-1`` — caller is responsible for filtering.
        """
        offsets = self.probe_midpoints - int(region_start)
        bins = offsets // int(resolution)
        bins = np.where(offsets < 0, -1, bins)
        return bins.astype(np.int64)


def load_wang2016(
    path: str | Path,
    chrom: Optional[str] = None,
) -> FISHReference:
    """Load Wang et al. 2016 multiplex-FISH 4DN FOF-CT CSV.

    The file is expected to follow the layout documented in ``docs/fish.md``:
    ``##``-prefixed metadata, ``##columns=...`` schema line, then
    comma-separated rows ``Spot_ID, Trace_ID, X, Y, Z, Chrom, Chrom_Start,
    Chrom_End``.

    Args:
        path: Path to the FOF-CT CSV (e.g. ``4DNFIW2N41FQ_chr21.csv``).
        chrom: If supplied, restrict to spots on this chromosome (matches the
            FOF-CT ``Chrom`` column, which is a bare ``21`` / ``X`` — leading
            ``chr`` is stripped during the comparison).

    Returns:
        :class:`FISHReference` aggregating the per-cell spot table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FISH file not found: {path}")

    assembly = "unknown"
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("##genome_assembly="):
                assembly = line.split("=", 1)[1].strip().strip(",").strip()

    df = pd.read_csv(path, comment="#", header=None, names=_FOF_CT_COLUMNS)
    df = df.dropna(subset=["Trace_ID", "X", "Y", "Z", "Chrom_Start", "Chrom_End"])
    for col in ("Spot_ID", "Trace_ID", "Chrom_Start", "Chrom_End"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Trace_ID", "Chrom_Start", "Chrom_End"])
    # Chrom may parse as float (21.0). Coerce via numeric→Int64→str so we
    # land on '21' / 'X' rather than '21.0'.
    chrom_numeric = pd.to_numeric(df["Chrom"], errors="coerce")
    chrom_str = chrom_numeric.astype("Int64").astype(str)
    fallback = df["Chrom"].astype(str).str.replace("^chr", "", regex=True).str.strip()
    df["Chrom"] = chrom_str.where(chrom_numeric.notna(), fallback)

    if chrom is not None:
        target = chrom.replace("chr", "")
        df = df[df["Chrom"] == target].copy()
        if df.empty:
            raise ValueError(f"No spots on chrom={chrom!r} in {path}")

    # Probe intervals — unique (Chrom_Start, Chrom_End) sorted by start.
    probes_df = (
        df[["Chrom_Start", "Chrom_End"]]
        .drop_duplicates()
        .sort_values("Chrom_Start")
        .reset_index(drop=True)
    )
    probe_intervals = probes_df.to_numpy(dtype=np.int64)
    probe_midpoints = ((probe_intervals[:, 0] + probe_intervals[:, 1]) // 2).astype(np.int64)
    probe_index = {tuple(row): i for i, row in enumerate(probe_intervals.tolist())}

    traces = df["Trace_ID"].drop_duplicates().sort_values().to_numpy()
    trace_index = {int(t): i for i, t in enumerate(traces)}

    n_cells = len(traces)
    n_probes = len(probe_intervals)
    coords = np.full((n_cells, n_probes, 3), np.nan, dtype=np.float64)
    for row in df.itertuples(index=False):
        ci = trace_index[int(row.Trace_ID)]
        pi = probe_index[(int(row.Chrom_Start), int(row.Chrom_End))]
        coords[ci, pi] = (row.X, row.Y, row.Z)

    chrom_str = df["Chrom"].iloc[0]
    chrom_out = chrom_str if chrom_str.startswith("chr") else f"chr{chrom_str}"

    return FISHReference(
        coords=coords,
        probe_midpoints=probe_midpoints,
        probe_intervals=probe_intervals,
        chrom=chrom_out,
        assembly=assembly,
        source=path.name,
    )
