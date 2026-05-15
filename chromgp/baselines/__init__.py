"""Baseline 3D-recon methods for head-to-head comparison with ChromGP.

Each baseline (PoisMS, Pastis) is implemented as a self-contained module that
takes the same config YAML as the ChromGP pipeline, fits the method on the
matched Hi-C inputs, and writes outputs to ``outputs/<dataset>/<region>/<method>/``
mirroring the ChromGP layout. The FISH validation block is the same one used
for ChromGP, so the resulting table is apples-to-apples.
"""

from .common import (
    load_raw_counts,
    save_baseline_outputs,
    fish_evaluate_positions,
)

__all__ = [
    "load_raw_counts",
    "save_baseline_outputs",
    "fish_evaluate_positions",
]
