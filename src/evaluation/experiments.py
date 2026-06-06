"""
evaluation/experiments.py — Orquestador del diseño experimental.

Diseño experimental (según orientacion.md):
─────────────────────────────────────────────────────────────────────────────

Configuraciones comparadas:
    A — baseline_cosine : DP con similitud coseno (sin LLM)
    B — dp_llm_full     : DP con coherencia LLM en todos los rangos
    C — two_phase       : Embeddings + LLM solo en cortes ambiguos

Instancias:
    - 3 tamaños: small (n≈15), medium (n≈60), large (n≈250)
    - 5 instancias por tamaño
    - Total: 15 instancias

Corridas totales: 3 configs × 15 instancias = 45 corridas.

Salida:
    data/results/results_all.csv — todas las métricas de todas las corridas.

Notas:
    - La configuración B (dp_llm_full) se omite por defecto para instancias
      large para evitar ~31,250 llamadas al LLM. Configurable con --skip-large-llm.
    - Las corridas se guardan incrementalmente para no perder resultados
      si el proceso se interrumpe.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.evaluation.metrics import compute_all_metrics
from src.instance import list_instances, load_instance
from src.llm.client import LLMClient, get_default_client
from src.problem import SegmentationProblem
import src.solver.baseline as solver_baseline
import src.solver.dp_llm as solver_dp_llm
import src.solver.two_phase as solver_two_phase
from src.solver.baseline import compute_embeddings, build_coherence_matrix_cosine


# ──────────────────────────────────────────────────────────────────────────────
# Configuraciones de solvers
# ──────────────────────────────────────────────────────────────────────────────

CONFIGS = ["baseline_cosine", "dp_llm_full", "two_phase"]


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución de una sola corrida
# ──────────────────────────────────────────────────────────────────────────────

def run_single(
    problem: SegmentationProblem,
    config: str,
    client: Optional[LLMClient] = None,
) -> dict:
    """
    Ejecuta una configuración en una instancia y devuelve las métricas.

    Args:
        problem : instancia del problema
        config  : "baseline_cosine" | "dp_llm_full" | "two_phase"
        client  : cliente LLM compartido (para aprovechar el caché)

    Returns:
        dict con todas las métricas (listo para escribir en CSV).
    """
    print(f"\n  → Ejecutando config='{config}' | instancia='{problem.name}' | n={problem.n}")

    if config == "baseline_cosine":
        seg, stats = solver_baseline.solve(problem)
        # Para métricas, necesitamos la matriz de coseno y embeddings
        emb = compute_embeddings(problem)
        coh = build_coherence_matrix_cosine(emb)

    elif config == "dp_llm_full":
        if client is None:
            client = get_default_client()
        seg, stats = solver_dp_llm.solve(problem, client=client, show_progress=False)
        # Para métricas de separación, necesitamos embeddings
        emb = compute_embeddings(problem)
        coh = [[0.0] * problem.n for _ in range(problem.n)]  # placeholder
        # Reconstruir la coh matrix desde el caché del LLM
        for i in range(problem.n):
            coh[i][i] = 1.0
            for j in range(i + 1, problem.n):
                coh[i][j] = client.ask_coherence_range(problem, i, j)

    elif config == "two_phase":
        if client is None:
            client = get_default_client()
        seg, stats = solver_two_phase.solve(problem, client=client, show_progress=False)
        emb = compute_embeddings(problem)
        coh = build_coherence_matrix_cosine(emb)

    else:
        raise ValueError(f"Configuración desconocida: '{config}'")

    metrics = compute_all_metrics(
        segmentation=seg,
        problem=problem,
        coherence_matrix=coh,
        embeddings=emb,
        stats=stats,
    )
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Ejecución del experimento completo
# ──────────────────────────────────────────────────────────────────────────────

def run_all_experiments(
    instances_dir: str = "data/instances",
    results_dir: str = "data/results",
    configs: Optional[List[str]] = None,
    skip_large_dp_llm: bool = True,
    large_threshold_n: int = 100,
) -> str:
    """
    Ejecuta todas las corridas del diseño experimental y guarda los resultados.

    Args:
        instances_dir      : directorio con las instancias JSON
        results_dir        : directorio donde guardar los resultados
        configs            : lista de configs a ejecutar (default: todas)
        skip_large_dp_llm  : si True, omite dp_llm_full para instancias grandes
        large_threshold_n  : umbral de n para considerar instancia "grande"

    Returns:
        Ruta al archivo CSV de resultados.
    """
    if configs is None:
        configs = CONFIGS

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    output_csv = results_path / "results_all.csv"

    instance_paths = list_instances(instances_dir)
    if not instance_paths:
        raise FileNotFoundError(
            f"No se encontraron instancias en '{instances_dir}'. "
            "Ejecuta primero: python src/main.py --generate-instances"
        )

    # Crear cliente LLM compartido (para maximizar hits de caché entre configs)
    client = get_default_client(verbose=False)

    # Determinar si ya existe el CSV (para no re-escribir el header)
    existing_rows = set()
    write_header = True
    if output_csv.exists():
        write_header = False
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.add((row["instance"], row["config"]))

    total = len(instance_paths) * len(configs)
    done = 0
    t_global = time.perf_counter()

    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = None

        for inst_path in instance_paths:
            problem = load_instance(inst_path)
            print(f"\n[experiments] Instancia: {problem.name} (n={problem.n})")

            for config in configs:
                done += 1
                key = (problem.name, config)

                # Skip si ya está en el CSV
                if key in existing_rows:
                    print(f"  → SKIP (ya existe) | config='{config}'")
                    continue

                # Skip dp_llm_full para instancias grandes
                if (
                    skip_large_dp_llm
                    and config == "dp_llm_full"
                    and problem.n > large_threshold_n
                ):
                    print(
                        f"  → SKIP dp_llm_full (n={problem.n} > {large_threshold_n}) "
                        "para evitar demasiadas llamadas al LLM."
                    )
                    continue

                try:
                    metrics = run_single(problem, config, client=client)
                except Exception as e:
                    print(f"  ⚠️  ERROR en config='{config}' instancia='{problem.name}': {e}")
                    continue

                # Inicializar el DictWriter con el header de la primera fila
                if writer is None:
                    fieldnames = list(metrics.keys())
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()
                        write_header = False

                writer.writerow(metrics)
                csvfile.flush()  # guardar inmediatamente

                elapsed = time.perf_counter() - t_global
                print(
                    f"  ✓ [{done}/{total}] score={metrics['objective_score']:.4f} | "
                    f"segs={metrics['num_segments']} | "
                    f"llm_calls={metrics['llm_calls']} | "
                    f"t={metrics['time_total_s']}s | "
                    f"elapsed={elapsed:.1f}s"
                )

    print(f"\n[experiments] ✅ Resultados guardados en: {output_csv}")
    return str(output_csv)


# ──────────────────────────────────────────────────────────────────────────────
# Script de resumen rápido
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results_csv: str = "data/results/results_all.csv") -> None:
    """
    Imprime un resumen de los resultados por configuración.
    """
    import pandas as pd

    df = pd.read_csv(results_csv)
    print("\n" + "=" * 70)
    print("RESUMEN DE EXPERIMENTOS")
    print("=" * 70)

    for config in df["config"].unique():
        subset = df[df["config"] == config]
        print(f"\n  Configuración: {config}")
        print(f"    Instancias:         {len(subset)}")
        print(f"    Coherencia interna: {subset['intra_coherence'].mean():.4f} ± {subset['intra_coherence'].std():.4f}")
        print(f"    Separación inter:   {subset['inter_separation'].mean():.4f} ± {subset['inter_separation'].std():.4f}")
        print(f"    Tiempo medio (s):   {subset['time_total_s'].mean():.3f}")
        print(f"    LLM calls (total):  {subset['llm_calls'].sum()}")
        if subset["has_ground_truth"].any():
            gt = subset[subset["has_ground_truth"] == True]
            print(f"    Boundary F1:        {gt['boundary_f1'].mean():.4f}")
    print("=" * 70)
