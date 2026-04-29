"""Generate command: create per-model configs from general.yaml."""

import click


def run(config_path: str):
    """Generate per-model configs from a general.yaml.

    Args:
        config_path: Path to the general config YAML.
    """
    from ..generate import generate_configs

    generated = generate_configs(config_path)
    click.echo(f"Generated {len(generated)} model configs:")
    for name, path in generated.items():
        click.echo(f"  {name}: {path}")
