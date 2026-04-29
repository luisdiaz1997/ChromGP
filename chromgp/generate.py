"""Config generation from general.yaml to per-model YAMLs.

Reads a general config (superset of all model params) and generates
per-model configs (svgp.yaml, mggp_svgp.yaml, lcgp.yaml, mggp_lcgp.yaml)
with proper field filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import Config


# Model variants to generate from a general config
MODEL_VARIANTS = [
    {"name": "svgp", "prior": "SVGP", "groups": False, "local": False},
    {"name": "mggp_svgp", "prior": "SVGP", "groups": True, "local": False},
    {"name": "lcgp", "prior": "LCGP", "groups": False, "local": True},
    {"name": "mggp_lcgp", "prior": "LCGP", "groups": True, "local": True},
]

# Field categories for filtering
COMMON_MODEL_FIELDS = {
    "n_components",
    "E",
}

SPATIAL_MODEL_FIELDS = {
    "kernel",
    "lengthscale",
    "sigma",
    "train_lengthscale",
    "noise",
    "jitter",
    "integrated_force",
}

SVGP_ONLY_FIELDS = {
    "num_inducing",
    "cholesky_mode",
}

LCGP_MODEL_FIELDS = {
    # LCGP-specific fields (placeholders for future implementation)
    # These will be added when GPzoo has LCGP support
}

GROUPS_MODEL_FIELDS = {
    "groups",
}


def generate_configs(general_path: str | Path) -> Dict[str, Path]:
    """Generate per-model configs from a general.yaml.

    Args:
        general_path: Path to the general config YAML.

    Returns:
        Dictionary mapping model name (e.g., 'svgp', 'mggp_svgp')
        to the generated config file path.

    Raises:
        ValueError: If general_path is not a general config.
    """
    general_path = Path(general_path)

    # Verify it's a general config
    if not Config.is_general_config(general_path):
        raise ValueError(f"{general_path} is not a general config (must lack model.prior)")

    # Load general config
    config = Config.from_yaml(general_path)

    # Always generate per-model configs into the same directory as general.yaml
    output_dir = general_path.parent

    generated = {}

    for variant in MODEL_VARIANTS:
        model_config = _generate_model_config(config, variant)

        # Set name to {dataset}_{model_name}
        model_config.name = f"{config.dataset}_{variant['name']}"

        # Determine output filename
        output_path = output_dir / f"{variant['name']}.yaml"

        # Save config
        model_config.save_yaml(output_path)
        generated[variant["name"]] = output_path

    return generated


def _generate_model_config(
    config: Config, variant: Dict[str, any]
) -> Config:
    """Generate a Config for a specific model variant.

    Filters the model section to only include relevant fields for the variant.

    Args:
        config: The general config.
        variant: Dict with 'name', 'prior', 'groups', 'local' keys.

    Returns:
        A new Config with filtered model section.
    """
    is_groups = variant["groups"]

    # Start with common model fields
    model_dict = {}

    for key in COMMON_MODEL_FIELDS:
        if key in config.model:
            model_dict[key] = config.model[key]

    # Add spatial fields (all ChromGP models are spatial)
    model_dict["prior"] = variant["prior"]
    model_dict["groups"] = is_groups

    # Add core spatial fields
    for key in SPATIAL_MODEL_FIELDS:
        if key in config.model:
            model_dict[key] = config.model[key]

    # Add SVGP-only or LCGP-specific fields
    is_local = variant.get("local", False)
    if is_local:
        model_dict["local"] = True
        for key in LCGP_MODEL_FIELDS:
            if key in config.model and key != "local":
                model_dict[key] = config.model[key]
    else:
        for key in SVGP_ONLY_FIELDS:
            if key in config.model:
                model_dict[key] = config.model[key]

    # Add group fields if applicable
    if is_groups:
        for key in GROUPS_MODEL_FIELDS:
            if key in config.model and key != "groups":
                model_dict[key] = config.model[key]

    # Build new config dict
    config_dict = {
        "name": config.name,
        "seed": config.seed,
        "dataset": config.dataset,
        "preprocessing": config.preprocessing,
        "model": model_dict,
        "training": config.training,
        "output_dir": config.output_dir,
    }

    return Config.from_dict(config_dict)
