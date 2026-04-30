"""Configuration system for ChromGP experiments.

Adapted from Spatial-Factorization with ChromGP-specific fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """Experiment configuration.

    Attributes:
        name: Experiment name
        seed: Random seed
        dataset: Dataset name (e.g., '4DNFIXP4QG5B')
        preprocessing: Dataset preprocessing parameters (mcool path, region, resolution, etc.)
        model: Model parameters dict (prior, groups, kernel, etc.)
        training: Training parameters dict
        output_dir: Output directory path
    """

    name: str
    seed: int = 67
    dataset: str = "4DNFIXP4QG5B"

    # Preprocessing config (Hi-C specific)
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        "mcool_path": None,
        "resolution": 25000,
        "region": None,
        "balance": True,
        "contact_transform": "log1p",
        "num_replicates": 1,
        "noise_level": 0.15,
        "groups_by": None,  # 'chromosome' or 'chromhmm_state'
        "chromhmm_bed": None,
        "chromhmm_states": None,
    })

    # Model config (passed to ChromGP)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "prior": "SVGP",  # SVGP, LCGP
        "groups": False,  # MGGP if true
        "E": 1,
        "n_components": 3,  # 3D output
        "kernel": "RBF",
        "lengthscale": 8.0,
        "output_lengthscale": 1.0,
        "sigma": 1.0,
        "train_lengthscale": False,
        "num_inducing": 800,
        "cholesky_mode": "exp",
        "noise": 0.1,
        "jitter": 1e-5,
        "scale": 10000,
        "integrated_force": False,
        "scale_kl_NM": True,
        "K": 50,
        "neighbors": "probabilistic",
        "precompute_knn": True,
    })

    # Training config
    training: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 20000,
        "learning_rate": 2e-3,
        "optimizer": "Adam",
        "device": "gpu",
        "batch_size": None,
        "y_batch_size": None,
        "shuffle": True,
    })

    # Output config
    output_dir: str = "outputs"

    @property
    def prior(self) -> str:
        """Return the prior class name."""
        return self.model.get("prior", "SVGP")

    @property
    def groups(self) -> bool:
        """Return whether multi-group (MGGP) mode is enabled."""
        return self.model.get("groups", False)

    @property
    def local(self) -> bool:
        """Return whether LCGP (local conditioning) mode is enabled."""
        return self.model.get("prior", "SVGP") == "LCGP"

    @property
    def model_name(self) -> str:
        """Return model directory name based on groups/local config.

        - SVGP, no groups: "svgp"
        - SVGP, with groups: "mggp_svgp"
        - LCGP, no groups: "lcgp"
        - LCGP, with groups: "mggp_lcgp"

        model.model_name_override bypasses the above and returns the
        override value directly.
        """
        override = self.model.get("model_name_override")
        if override:
            return override

        prior = self.model.get("prior", "SVGP").lower()
        if self.groups:
            return f"mggp_{prior}"
        return prior

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            name=data["name"],
            seed=data.get("seed", 67),
            dataset=data.get("dataset", "4DNFIXP4QG5B"),
            preprocessing=data.get("preprocessing", {}),
            model=data.get("model", {}),
            training=data.get("training", {}),
            output_dir=data.get("output_dir", "outputs"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "dataset": self.dataset,
            "preprocessing": self.preprocessing,
            "model": self.model,
            "training": self.training,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file with blank lines between sections."""
        d = self.to_dict()
        with open(path, "w") as f:
            # Write simple fields in order
            f.write(f"name: {d['name']}\n")
            f.write(f"seed: {d['seed']}\n")
            f.write(f"dataset: {d['dataset']}\n")
            f.write(f"output_dir: {d['output_dir']}\n")

            # Write sections with blank lines before each
            for key in ["preprocessing", "model", "training"]:
                f.write(f"\n{key}:\n")
                # Get yaml content without trailing newline from dump
                content = yaml.dump(d[key], default_flow_style=False, sort_keys=False)
                # Indent each line by 2 spaces
                for line in content.strip().split('\n'):
                    f.write(f"  {line}\n")

    @classmethod
    def is_general_config(cls, path: str | Path) -> bool:
        """Check if a config is a general config (no model.prior key).

        A general config is a superset of all model params and will be used
        to generate per-model configs (svgp.yaml, mggp_svgp.yaml, etc.).

        Args:
            path: Path to the YAML config file.

        Returns:
            True if the config is general (no model.prior key), False otherwise.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        model_section = data.get("model", {})
        return "prior" not in model_section
