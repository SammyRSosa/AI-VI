"""
solver/dp_llm.py — Solver DP con evaluación de coherencia por LLM.

CONFIGURACIÓN B: Programación Dinámica + LLM completo (todas las celdas).

Diferencias con baseline.py:
    - La matriz de coherencias se construye llamando al LLM para CADA rango [i,j].
    - Esto produce O(n²/2) llamadas al LLM en el peor caso.
    - El caché en disco convierte las re-ejecuciones en O(1) operaciones de disco.
    - La DP en sí es idéntica al baseline (mismo algoritmo, distinta métrica).

Recomendación de uso:
    - Solo para instancias pequeñas (n ≤ 20) en la primera ejecución.
    - Para n mediano/grande, usar two_phase.py que limita las llamadas al LLM.

Advertencia de costo:
    Para n=50 → ~1225 llamadas al LLM.
    Para n=100 → ~4950 llamadas al LLM.
    Usar caché para re-ejecuciones.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from tqdm import tqdm

from src.llm.client import LLMClient, get_default_client
from src.problem import Segmentation, SegmentationProblem
from src.solver.baseline import dp_segmentation


# ──────────────────────────────────────────────────────────────────────────────
# Construcción de la matriz de coherencias con LLM
# ──────────────────────────────────────────────────────────────────────────────

def build_coherence_matrix_llm(
    problem: SegmentationProblem,
    client: LLMClient,
    show_progress: bool = True,
) -> list:
    """
    Construye la matriz de coherencias evaluando CADA rango [i, j] con el LLM.

    Nota: coh[i][i] = 1.0 por definición (un solo elemento es trivialmente coherente).

    Complejidad: O(n²) llamadas al LLM.
    Con caché: la primera ejecución es lenta; las siguientes son O(n²) de disco.

    Args:
        problem       : instancia del problema
        client        : cliente LLM con caché integrado
        show_progress : si True, muestra barra de progreso

    Returns:
        Matriz n×n de coherencias (triangular superior).
    """
    n = problem.n
    coh = [[0.0] * n for _ in range(n)]

    total_pairs = n * (n + 1) // 2  # diagonal + triangular superior

    pbar = tqdm(total=total_pairs, desc="[dp_llm] Calculando coherencias", disable=not show_progress)

    for i in range(n):
        # Segmento de un solo elemento: coherencia = 1.0
        coh[i][i] = 1.0
        pbar.update(1)

        for j in range(i + 1, n):
            score = client.ask_coherence_range(problem, i, j)
            coh[i][j] = score
            pbar.update(1)

    pbar.close()
    return coh


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada del solver DP+LLM completo
# ──────────────────────────────────────────────────────────────────────────────

def solve(
    problem: SegmentationProblem,
    client: Optional[LLMClient] = None,
    show_progress: bool = True,
) -> Tuple[Segmentation, dict]:
    """
    Ejecuta el solver DP+LLM completo.

    Args:
        problem       : instancia del problema
        client        : cliente LLM (si None, se crea uno con get_default_client)
        show_progress : mostrar barra de progreso

    Returns:
        (segmentation, stats)
    """
    if client is None:
        client = get_default_client()

    client.reset_stats()
    t0 = time.perf_counter()

    # Paso 1: Matriz de coherencias con LLM
    t_llm0 = time.perf_counter()
    coh_matrix = build_coherence_matrix_llm(problem, client, show_progress=show_progress)
    t_llm = time.perf_counter() - t_llm0

    # Paso 2: DP (idéntica al baseline)
    t_dp0 = time.perf_counter()
    seg = dp_segmentation(problem, coh_matrix)
    seg.config_name = "dp_llm_full"
    t_dp = time.perf_counter() - t_dp0

    total_time = time.perf_counter() - t0
    llm_stats = client.stats()

    stats = {
        "config": "dp_llm_full",
        "instance": problem.name,
        "n": problem.n,
        "num_segments": seg.num_segments(),
        "score": seg.score,
        "time_total_s": round(total_time, 4),
        "time_llm_calls_s": round(t_llm, 4),
        "time_dp_s": round(t_dp, 4),
        "llm_calls": llm_stats["real_llm_calls"],
        "cache_hits": llm_stats["cache_stats"]["hits"],
        "model": llm_stats["model"],
    }

    return seg, stats
