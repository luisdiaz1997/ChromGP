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
        cmd.run(config, animation=kwargs.get("animation", False))
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
        chromgp generate -c configs/4DNFIXP4QG5B/chr14/general.yaml
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
    from .commands import analyze as analyze_cmd
    analyze_cmd.run(config)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--animation", is_flag=True, default=False, help="Generate training animation GIF (slow)")
def figures(config, animation):
    """Generate publication figures (ELBO curve, reconstruction, training animation)."""
    from .commands import figures as fig_cmd
    fig_cmd.run(config, animation=animation)


@cli.group()
def baseline():
    """Run head-to-head baseline 3D-recon methods (PoisMS, Pastis)."""
    pass


@baseline.command("poisms")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML")
@click.option("--df", type=int, default=5, show_default=True,
              help="B-spline degrees of freedom (PoisMS paper default 5)")
@click.option("--maxepoch", type=int, default=100, show_default=True,
              help="PoisMS outer-loop epoch cap")
def baseline_poisms(config, df, maxepoch):
    """Fit official PoisMS (R) on the Hi-C matrix and FISH-validate."""
    from .baselines import poisms as poisms_cmd
    poisms_cmd.run(config, df=df, maxepoch=maxepoch)


@baseline.command("summary")
@click.option("--dataset-dir", type=click.Path(exists=True), required=True,
              help="e.g. outputs/4DNFIJTOIGOI")
@click.option("--regions", default="chr20,chr21,chr22", show_default=True,
              help="Comma-separated region slugs")
@click.option("--methods", default="svgp,poisms", show_default=True,
              help="Comma-separated method slugs (subdirs under each region)")
def baseline_summary(dataset_dir, regions, methods):
    """Print a comparison table of FISH metrics across methods × regions."""
    from .baselines.summary import render_summary
    render_summary(dataset_dir,
                   [r.strip() for r in regions.split(",") if r.strip()],
                   [m.strip() for m in methods.split(",") if m.strip()])


@cli.command()
@click.argument("stages", nargs=-1, required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to config YAML")
@click.option("--resume", is_flag=True, default=False,
              help="Resume training from saved checkpoint")
@click.option("--video", is_flag=True, default=False,
              help="Capture trajectory snapshots and save as MP4")
@click.option("--animation", is_flag=True, default=False,
              help="Generate training animation GIF (slow)")
def run(stages, config, resume, video, animation):
    """Run pipeline stages sequentially.

    \b
    STAGES:
        preprocess, train, analyze, figures  - Run specific stages in order

    \b
    EXAMPLES:
        chromgp run preprocess train -c configs/4DNFIXP4QG5B/chr14/svgp.yaml
        chromgp run train -c configs/4DNFIXP4QG5B/chr14/svgp.yaml --resume
        chromgp run preprocess train -c configs/4DNFIXP4QG5B/chr14/svgp.yaml --video
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
        _run_stage(stage, config, resume=resume, video=video, animation=animation)

    click.echo(f"\nAll stages complete.")
