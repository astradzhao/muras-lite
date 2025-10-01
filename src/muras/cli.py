import json
import sys
import click
from pathlib import Path
from .metrics import Sample, evaluate


@click.group()
@click.version_option(package_name="muras")
def main():
    """Muras: Multimodal RAG Assessment Suite"""
    pass


@main.command(name="evaluate")
@click.option('--input', '-i', type=click.File('r'), default="/workspace/muras/sample/sample_input.json",
              help='Input JSON file (defaults to stdin)')
@click.option('--output', '-o', type=click.File('w'), default=sys.stdout,
              help='Output JSON file (defaults to stdout)')
def evaluate_cmd(input, output):
    """Evaluate multimodal RAG samples from JSON input."""
    data = json.load(input)
    samples = [Sample(**d) for d in data["samples"]]
    result = evaluate(samples)
    output.write(json.dumps(result, indent=2))
    output.write('\n')


@main.command()
@click.argument('sample_file', type=click.Path(exists=True))
def benchmark(sample_file):
    """Run benchmark suite on a dataset file."""
    click.echo(f"Running benchmark on: {sample_file}")
    # TODO: Implement benchmark logic
    click.secho("Not implemented yet!", fg="yellow")
