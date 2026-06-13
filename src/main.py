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

load_dotenv(override=True)
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
@click.option("--noise-ratio", default=0.1, type=float, help="Fracción de elementos con ruido.")
@click.option(
    "--noise-type",
    default="topic_swap",
    type=click.Choice(["topic_swap", "filler_injection"]),
    help="Tipo de ruido a inyectar.",
)
def generate_instances(output_dir: str, seed: int, noise_ratio: float, noise_type: str):
    """Genera el conjunto estándar de instancias sintéticas para los experimentos."""
    from src.instance import generate_standard_instances
    console.print(f"[bold green]Generando instancias (ruido={noise_ratio}, tipo={noise_type})...[/bold green]")
    generate_standard_instances(output_dir=output_dir, seed=seed, noise_ratio=noise_ratio, noise_type=noise_type)
    console.print(f"[bold green][OK] Instancias guardadas en '{output_dir}'[/bold green]")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: descargar e importar artículo de Wikipedia
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("fetch-wikipedia")
@click.option("--title", required=True, help="Título del artículo de Wikipedia (e.g. 'Revolución Francesa').")
@click.option("--lang", default="es", help="Idioma de Wikipedia ('es', 'en', etc.).")
@click.option("--output-dir", default="data/instances", help="Directorio donde guardar el JSON.")
def fetch_wikipedia(title: str, lang: str, output_dir: str):
    """Descarga un artículo de Wikipedia y lo guarda como una instancia del problema."""
    from src.instance import load_wikipedia_as_instance, save_instance
    console.print(f"[bold green]Descargando artículo '{title}' en {lang}...[/bold green]")
    try:
        problem = load_wikipedia_as_instance(page_title=title, language=lang)
        output_path = Path(output_dir) / f"{problem.name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_instance(problem, str(output_path))
        console.print(f"[bold green][OK] Artículo guardado correctamente en: {output_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error al descargar artículo: {e}[/bold red]")


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
@click.option(
    "--refinement-method",
    default="batch",
    type=click.Choice(["batch", "pairwise"]),
    show_default=True,
    help="Método de refinamiento para Two-Phase ('batch' o 'pairwise').",
)
@click.option("--verbose", is_flag=True, default=False, help="Modo verbose.")
def run_solver(instance: str, config: str, refinement_method: str, verbose: bool):
    """Ejecuta el solver en una instancia y muestra los resultados."""
    from src.instance import load_instance
    from src.evaluation.metrics import compute_all_metrics
    from src.solver.baseline import compute_embeddings, build_coherence_matrix_cosine
    import src.solver.baseline as solver_baseline
    import src.solver.dp_llm as solver_dp_llm
    import src.solver.two_phase as solver_two_phase

    console.print(f"\n[bold]Cargando instancia:[/bold] {instance}")
    problem = load_instance(instance)
    console.print(f"  n={problem.n} | lambda={problem.lambda_pen} | ground_truth={problem.true_cuts is not None}")

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
        seg, stats = solver_two_phase.solve(problem, client=client, refinement_method=refinement_method)
        emb = compute_embeddings(problem)
        coh = build_coherence_matrix_cosine(emb)

    metrics = compute_all_metrics(seg, problem, coh, emb, stats)

    # Mostrar resultados
    console.print("\n[bold green]=== RESULTADO ===[/bold green]")
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
        console.print(f"  [{i+1}] E[{start}..{end}] -> \"{snippet}...\"")


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
    console.print(f"[bold green][OK] Resultados guardados en: {csv_path}[/bold green]")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: resumen de resultados
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("summary")
@click.option("--results-csv", default="data/results/results_all.csv", help="CSV de resultados.")
def summary(results_csv: str):
    """Imprime un resumen de los experimentos completados."""
    from src.evaluation.experiments import print_summary
    print_summary(results_csv)


# ──────────────────────────────────────────────────────────────────────────────
# Comandos de Transcripciones e Importación
# ──────────────────────────────────────────────────────────────────────────────

@cli.command("import-transcript")
@click.option("--file", required=True, help="Ruta al archivo .txt de la transcripción.")
@click.option("--output-dir", default="data/instances", help="Directorio de salida para la instancia JSON.")
@click.option("--words", default=120, type=int, help="Número de palabras promedio por párrafo.")
@click.option("--lambda-pen", default=0.50, type=float, help="Penalización de cortes (λ) por defecto.")
@click.option("--min-seg", default=2, type=int, help="Longitud mínima de segmento por defecto.")
def import_transcript(file: str, output_dir: str, words: int, lambda_pen: float, min_seg: int):
    """Importa y normaliza un archivo local .txt de transcripción."""
    from src.utils.batch_processor import process_single_transcript_file
    console.print(f"[bold green]Importando transcripción '{file}'...[/bold green]")
    try:
        out_path = process_single_transcript_file(
            file_path=file,
            output_dir=output_dir,
            target_word_count=words,
            lambda_pen=lambda_pen,
            min_seg=min_seg
        )
        console.print(f"[bold green][OK] Transcripción normalizada e importada en: {out_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error al importar transcripción: {e}[/bold red]")


@cli.command("import-youtube")
@click.option("--url", required=True, help="Enlace de YouTube o ID de video de 11 caracteres.")
@click.option("--output-dir", default="data/instances", help="Directorio de salida para la instancia JSON.")
@click.option("--words", default=120, type=int, help="Número de palabras promedio por párrafo.")
@click.option("--lang", default="es", help="Idioma preferido para la transcripción.")
@click.option("--lambda-pen", default=0.50, type=float, help="Penalización de cortes (λ) por defecto.")
@click.option("--min-seg", default=2, type=int, help="Longitud mínima de segmento por defecto.")
def import_youtube(url: str, output_dir: str, words: int, lang: str, lambda_pen: float, min_seg: int):
    """Descarga e importa una transcripción directamente desde un video de YouTube."""
    from src.utils.youtube_downloader import download_and_convert_youtube
    from src.instance import save_instance
    console.print(f"[bold green]Descargando transcripción de YouTube '{url}'...[/bold green]")
    try:
        problem = download_and_convert_youtube(
            url=url,
            target_word_count=words,
            languages=[lang, "en"],
            lambda_pen=lambda_pen,
            min_seg=min_seg
        )
        out_path = Path(output_dir) / f"{problem.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_instance(problem, str(out_path))
        console.print(f"[bold green][OK] Transcripción descargada e importada en: {out_path}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error al descargar de YouTube: {e}[/bold red]")


@cli.command("batch-transcripts")
@click.option("--input-dir", required=True, help="Directorio con archivos .txt de transcripciones.")
@click.option("--output-dir", default="data/instances", help="Directorio de salida para las instancias JSON.")
@click.option("--words", default=120, type=int, help="Número de palabras promedio por párrafo.")
@click.option("--lambda-pen", default=0.50, type=float, help="Penalización de cortes (λ) por defecto.")
@click.option("--min-seg", default=2, type=int, help="Longitud mínima de segmento por defecto.")
def batch_transcripts(input_dir: str, output_dir: str, words: int, lambda_pen: float, min_seg: int):
    """Procesa en lote múltiples archivos .txt de transcripciones."""
    from src.utils.batch_processor import batch_process_transcripts
    console.print(f"[bold green]Procesando transcripciones en lote desde '{input_dir}'...[/bold green]")
    try:
        out_files = batch_process_transcripts(
            input_dir=input_dir,
            output_dir=output_dir,
            target_word_count=words,
            lambda_pen=lambda_pen,
            min_seg=min_seg
        )
        console.print(f"[bold green][OK] Se procesaron {len(out_files)} archivos con éxito.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error en procesamiento por lote: {e}[/bold red]")


if __name__ == "__main__":
    cli()
