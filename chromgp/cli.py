"""CLI entry point for ChromGP using Click."""

import click

STAGE_ORDER = ["preprocess", "train", "analyze", "figures"]


def _run_stage(stage, config, **kwargs):
    """Import and run a single pipeline stage."""
    if stage == "preprocess":
        from .commands import preprocess as cmd
        cmd.run(config)
    elif stage == "train":
        from .commands import train as cmd
        cmd.run(config, resume=kwargs.get("resume", False), video=kwargs.get("video", False))
    elif stage == "analyze":
        from .commands import analyze as cmd
        cmd.run(config)
    elif stage == "figures":
        from .commands import figures as cmd
        cmd.run(config)
    else:
        raise click.BadParameter(f"Unknown stage: {stage}")


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
    from .generate import generate_configs

    generated = generate_configs(config)
    click.echo(f"Generated {len(generated)} model configs:")
    for name, path in generated.items():
        click.echo(f"  {name}: {path}")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def preprocess(config):
    """Preprocess dataset into standardized format."""
    from .commands import preprocess as prep_cmd
    prep_cmd.run(config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--resume", is_flag=True, default=False, help="Resume training from saved checkpoint")
@click.option("--video", is_flag=True, default=False, help="Capture trajectory snapshots and save as MP4")
def train(config, resume, video):
    """Train a ChromGP model (SVGP, optionally with groups/batching)."""
    from .commands import train as train_cmd
    train_cmd.run(config, resume=resume, video=video)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def analyze(config):
    """Analyze a trained model (extract coords, metrics)."""
    click.echo(f"Analyze: {config} (not implemented yet)")


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
def figures(config):
    """Generate publication figures (ELBO curve, reconstruction, training animation)."""
    from .commands import figures as fig_cmd
    fig_cmd.run(config)


@cli.command()
@click.argument("stages", nargs=-1, required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to config YAML")
@click.option("--resume", is_flag=True, default=False,
              help="Resume training from saved checkpoint")
@click.option("--video", is_flag=True, default=False,
              help="Capture trajectory snapshots and save as MP4")
def run(stages, config, resume, video):
    """Run pipeline stages sequentially.

    \b
    STAGES:
        preprocess, train, analyze, figures  - Run specific stages in order

    \b
    EXAMPLES:
        chromgp run preprocess train -c configs/4DNFIXP4QG5B/svgp.yaml
        chromgp run train -c configs/4DNFIXP4QG5B/svgp.yaml --resume
        chromgp run preprocess train -c configs/4DNFIXP4QG5B/svgp.yaml --video
    """
    stages = [s.lower() for s in stages]

    # Validate
    valid = set(STAGE_ORDER)
    invalid = set(stages) - valid
    if invalid:
        raise click.BadParameter(
            f"Unknown stage(s): {', '.join(invalid)}. Valid: {', '.join(STAGE_ORDER)}"
        )

    # Sort into pipeline order
    ordered = [s for s in STAGE_ORDER if s in stages]
    click.echo(f"Running stages: {' → '.join(ordered)}")

    for stage in ordered:
        click.echo(f"\n{'='*60}")
        click.echo(f"  Stage: {stage}")
        click.echo(f"{'='*60}\n")
        _run_stage(stage, config, resume=resume, video=video)

    click.echo(f"\nAll stages complete.")
