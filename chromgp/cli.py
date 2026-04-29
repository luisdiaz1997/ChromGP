"""CLI entry point for ChromGP using Click."""

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Chromatin structure inference with Gaussian Processes."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to general config YAML")
def generate(config):
    """Generate per-model configs from a general.yaml.

    \b
    Example:
        chromgp generate -c configs/4DNFIXP4QG5B/general.yaml
    """
    from .commands import generate as gen_cmd
    gen_cmd.run(config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def preprocess(config):
    """Preprocess dataset into standardized format."""
    click.echo(f"Preprocess: {config} (not implemented yet)")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--resume", is_flag=True, default=False, help="Resume training from saved checkpoint")
@click.option("--video", is_flag=True, default=False, help="Capture trajectory snapshots and save as MP4")
def train(config, resume, video):
    """Train a ChromGP model."""
    click.echo(f"Train: {config} (not implemented yet)")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def analyze(config):
    """Analyze a trained model (extract coords, metrics)."""
    click.echo(f"Analyze: {config} (not implemented yet)")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def figures(config):
    """Generate publication figures."""
    click.echo(f"Figures: {config} (not implemented yet)")
