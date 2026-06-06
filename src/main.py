"""
main.py — CLI de entrada del sistema de segmentación óptima.

Uso:
    python src/main.py --help
    python src/main.py --generate-instances
    python src/main.py --instance data/instances/small_01.json --config baseline_cosine
    python src/main.py --instance data/instances/medium_01.json --config two_phase
    python src/main.py --run-all-experiments
    python src/main.py --summary
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import print as rprint

load_dotenv()
console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Sistema de Segmentación Óptima de Contenido — Proyecto Final IA 2025-2026."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ──────────────────────────────────────────────────────────────────────────────
# Comando: generar instancias
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("generate-instances")
@click.option("--output-dir", default="data/instances", help="Directorio de salida.")
@click.option("--seed", default=42, type=int, help="Semilla aleatoria.")
def generate_instances(output_dir: str, seed: int):
    """Genera el conjunto estándar de instancias sintéticas para los experimentos."""
    from src.instance import generate_standard_instances
    console.print("[bold green]Generando instancias...[/bold green]")
    generate_standard_instances(output_dir=output_dir, seed=seed)
    console.print(f"[bold green]✓ Instancias guardadas en '{output_dir}'[/bold green]")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: ejecutar solver en una instancia
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("run")
@click.option("--instance", required=True, help="Ruta al JSON de la instancia.")
@click.option(
    "--config",
    default="two_phase",
    type=click.Choice(["baseline_cosine", "dp_llm_full", "two_phase"]),
    show_default=True,
    help="Configuración del solver.",
)
@click.option("--verbose", is_flag=True, default=False, help="Modo verbose.")
def run_solver(instance: str, config: str, verbose: bool):
    """Ejecuta el solver en una instancia y muestra los resultados."""
    from src.instance import load_instance
    from src.evaluation.metrics import compute_all_metrics
    from src.solver.baseline import compute_embeddings, build_coherence_matrix_cosine
    import src.solver.baseline as solver_baseline
    import src.solver.dp_llm as solver_dp_llm
    import src.solver.two_phase as solver_two_phase

    console.print(f"\n[bold]Cargando instancia:[/bold] {instance}")
    problem = load_instance(instance)
    console.print(f"  n={problem.n} | λ={problem.lambda_pen} | ground_truth={problem.true_cuts is not None}")

    from src.llm.client import get_default_client
    client = get_default_client(verbose=verbose)

    console.print(f"\n[bold]Ejecutando config:[/bold] {config}")

    if config == "baseline_cosine":
        seg, stats = solver_baseline.solve(problem)
        emb = compute_embeddings(problem)
        coh = build_coherence_matrix_cosine(emb)
    elif config == "dp_llm_full":
        seg, stats = solver_dp_llm.solve(problem, client=client)
        emb = compute_embeddings(problem)
        coh = [[1.0 if i == j else 0.5 for j in range(problem.n)] for i in range(problem.n)]
    elif config == "two_phase":
        seg, stats = solver_two_phase.solve(problem, client=client)
        emb = compute_embeddings(problem)
        coh = build_coherence_matrix_cosine(emb)

    metrics = compute_all_metrics(seg, problem, coh, emb, stats)

    # Mostrar resultados
    console.print("\n[bold green]═══ RESULTADO ═══[/bold green]")
    console.print(f"  Segmentos:         {seg.num_segments()}")
    console.print(f"  Cortes:            {seg.cuts}")
    console.print(f"  Función objetivo:  {seg.score:.6f}")
    console.print(f"  Coh. interna:      {metrics['intra_coherence']:.6f}")
    console.print(f"  Separación inter:  {metrics['inter_separation']:.6f}")
    if problem.true_cuts is not None:
        console.print(f"  Ground truth:      {problem.true_cuts}")
        console.print(f"  Boundary F1:       {metrics['boundary_f1']}")
    console.print(f"  Tiempo (s):        {stats['time_total_s']}")
    console.print(f"  LLM calls:         {stats['llm_calls']}")

    # Mostrar segmentos
    console.print("\n[bold]Segmentos encontrados:[/bold]")
    for i, (start, end) in enumerate(seg.segments()):
        snippet = problem.get_segment_text(start, min(start, end))[:80].replace("\n", " ")
        console.print(f"  [{i+1}] E[{start}..{end}] → \"{snippet}...\"")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: ejecutar todos los experimentos
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("run-experiments")
@click.option("--instances-dir", default="data/instances", help="Directorio de instancias.")
@click.option("--results-dir", default="data/results", help="Directorio de resultados.")
@click.option("--skip-large-llm/--no-skip-large-llm", default=True, help="Omitir dp_llm_full en instancias grandes.")
def run_experiments(instances_dir: str, results_dir: str, skip_large_llm: bool):
    """Ejecuta el diseño experimental completo (45 corridas)."""
    from src.evaluation.experiments import run_all_experiments
    console.print("[bold green]Iniciando experimentos...[/bold green]")
    csv_path = run_all_experiments(
        instances_dir=instances_dir,
        results_dir=results_dir,
        skip_large_dp_llm=skip_large_llm,
    )
    console.print(f"[bold green]✓ Resultados guardados en: {csv_path}[/bold green]")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: resumen de resultados
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("summary")
@click.option("--results-csv", default="data/results/results_all.csv", help="CSV de resultados.")
def summary(results_csv: str):
    """Imprime un resumen de los experimentos completados."""
    from src.evaluation.experiments import print_summary
    print_summary(results_csv)


if __name__ == "__main__":
    cli()
