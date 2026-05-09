"""ChIP-seq/CTCF BigWig data loader.

This module mirrors the Hi-C loader interface, but builds the target matrix
from per-bin BigWig signal tracks instead of a contact matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import GenomicData
from . import chromhmm


class ChIPSeqLoader:
    """Loader for chromatin mark and CTCF signal from BigWig files."""

    def load(self, preprocessing: dict) -> GenomicData:
        """Load binned BigWig signals into a ``GenomicData`` object.

        Expected preprocessing keys:
        - bigwig_paths: mapping of mark name -> BigWig path, or list of paths
        - resolution: bin size in bp
        - region: genomic region, e.g. ``chr1`` or ``chr1:0-50000000``
        - signal_stat: pyBigWig statistic, usually ``mean`` or ``max``
        - signal_transform: ``raw``, ``log1p``, ``log10``, or ``zscore``
        - groups_by/chromhmm_bed/chromhmm_states: same as HiCLoader
        """
        bigwig_paths_cfg = preprocessing["bigwig_paths"]
        mark_names, bigwig_paths = self._normalize_bigwig_paths(bigwig_paths_cfg)
        resolution = int(preprocessing["resolution"])
        region = preprocessing["region"]
        signal_stat = preprocessing.get("signal_stat", "mean")
        signal_transform = preprocessing.get("signal_transform", "log1p")
        groups_by = preprocessing.get("groups_by")

        chrom, start, end = self._parse_region(region)
        chrom_size = self._chrom_size(bigwig_paths[0], chrom)
        if start is None:
            start = 0
        if end is None:
            end = chrom_size
        end = min(end, chrom_size)

        bin_coords = self._make_bins(chrom, start, end, resolution)
        Y = self._load_signal_matrix(
            bigwig_paths, bin_coords, stat=signal_stat,
            missing_value=float(preprocessing.get("missing_value", 0.0)),
        )
        Y = self._apply_transform(Y, signal_transform)

        gc = None
        gc_reference = preprocessing.get("gc_reference")
        if gc_reference:
            from .gc import compute_gc
            gc = compute_gc(bin_coords, gc_reference)

        C = None
        n_groups = 0
        group_names = None

        if groups_by == "chromhmm_state":
            chromhmm_bed = preprocessing.get("chromhmm_bed")
            state_whitelist = preprocessing.get("chromhmm_states")
            if chromhmm_bed is None:
                raise ValueError("chromhmm_bed path required when groups_by == 'chromhmm_state'")

            chromhmm_df = chromhmm.load_chromhmm_bed(chromhmm_bed, state_whitelist)
            chromhmm_df = chromhmm.merge_chromhmm_groups(chromhmm_df)
            C = chromhmm.assign_chromhmm_states(bin_coords, chromhmm_df)
            group_names = chromhmm.get_state_names(chromhmm_df)
            n_groups = len(group_names)
        elif groups_by == "chromosome":
            pass

        X = torch.from_numpy(bin_coords["mid"].values.copy()).float()
        Y_t = torch.from_numpy(Y).float()
        valid_mask = torch.ones(len(bin_coords), dtype=torch.bool)

        metadata = {
            "assay": "chipseq",
            "resolution": resolution,
            "region": region,
            "chrom": chrom,
            "start": start,
            "end": end,
            "bigwig_paths": {name: str(path) for name, path in zip(mark_names, bigwig_paths)},
            "mark_names": mark_names,
            "signal_stat": signal_stat,
            "signal_transform": signal_transform,
            "groups_by": groups_by,
            "n_bins": len(bin_coords),
        }

        return GenomicData(
            X=X,
            Y=Y_t,
            C=C,
            n_groups=n_groups,
            group_names=group_names,
            gc=gc,
            contact_raw=None,
            contact_raw_full=None,
            valid_mask=valid_mask,
            bin_coords=bin_coords,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_bigwig_paths(bigwig_paths_cfg: dict | list) -> tuple[list[str], list[Path]]:
        if isinstance(bigwig_paths_cfg, dict):
            mark_names = list(bigwig_paths_cfg.keys())
            paths = [Path(p) for p in bigwig_paths_cfg.values()]
        else:
            paths = [Path(p) for p in bigwig_paths_cfg]
            mark_names = [p.parent.name or p.stem for p in paths]

        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing BigWig file(s): " + ", ".join(missing))
        return mark_names, paths

    @staticmethod
    def _parse_region(region: str) -> tuple[str, Optional[int], Optional[int]]:
        if ":" in region:
            chrom, coords = region.split(":")
            start, end = coords.split("-")
            return chrom, int(start), int(end)
        return region, None, None

    @staticmethod
    def _chrom_size(bigwig_path: Path, chrom: str) -> int:
        import pyBigWig

        with pyBigWig.open(str(bigwig_path)) as bw:
            chroms = bw.chroms()
        if chrom not in chroms:
            raise ValueError(f"Chromosome {chrom!r} not found in {bigwig_path}")
        return int(chroms[chrom])

    @staticmethod
    def _make_bins(chrom: str, start: int, end: int, resolution: int) -> pd.DataFrame:
        starts = np.arange(start, end, resolution, dtype=np.int64)
        ends = np.minimum(starts + resolution, end)
        bins = pd.DataFrame({"chrom": chrom, "start": starts, "end": ends})
        bins["mid"] = (bins["start"] + bins["end"]) // 2
        return bins

    @staticmethod
    def _load_signal_matrix(
        bigwig_paths: list[Path],
        bin_coords: pd.DataFrame,
        stat: str = "mean",
        missing_value: float = 0.0,
    ) -> np.ndarray:
        import pyBigWig

        signals = np.zeros((len(bin_coords), len(bigwig_paths)), dtype=np.float32)
        intervals = list(bin_coords[["chrom", "start", "end"]].itertuples(index=False, name=None))

        for j, path in enumerate(bigwig_paths):
            with pyBigWig.open(str(path)) as bw:
                vals = []
                for chrom, start, end in intervals:
                    value = bw.stats(chrom, int(start), int(end), type=stat, exact=False)[0]
                    vals.append(missing_value if value is None or np.isnan(value) else value)
            signals[:, j] = np.asarray(vals, dtype=np.float32)

        return signals

    @staticmethod
    def _apply_transform(Y: np.ndarray, transform: str) -> np.ndarray:
        if transform == "raw":
            return Y
        if transform == "log1p":
            return np.log1p(np.clip(Y, a_min=0.0, a_max=None))
        if transform == "log10":
            return np.log10(np.clip(Y, a_min=0.0, a_max=None) + 5e-6)
        if transform == "zscore":
            mean = Y.mean(axis=0, keepdims=True)
            std = Y.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            return (Y - mean) / std
        raise ValueError(f"Unknown signal_transform: {transform}")
