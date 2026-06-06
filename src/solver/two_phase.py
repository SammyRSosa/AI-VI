"""
solver/two_phase.py — Solver de dos fases: embeddings + LLM solo en zonas ambiguas.

CONFIGURACIÓN C (recomendada): Mejor balance calidad/costo de LLM.

Algoritmo de dos fases:
─────────────────────────────────────────────────────────────────────────────
Fase 1 — Segmentación inicial barata (coseno de embeddings):
    Igual que baseline.py. Produce una segmentación S₀ con los cortes C₀.
    Identifica los cortes "borderline": puntos de corte donde la similitud
    coseno entre el último elemento del segmento izquierdo y el primero del
    segmento derecho es > umbral_ambigüedad (los temas son similares → el
    corte es dudoso).

Fase 2 — Refinamiento con LLM (solo en zonas ambiguas):
    Para cada zona ambigua [c - ventana, c + ventana]:
        - Calcular coherencias LLM de los rangos candidatos alternativos.
        - Ejecutar DP local en la ventana.
        - Sustituir el corte original por el corte refinado.

Resultado: segmentación S₁ con llamadas al LLM ≈ O(k · w²) donde:
    k = número de cortes ambiguos ≤ k_total
    w = tamaño de la ventana de revisión

Parámetros clave:
    ambiguity_threshold : similitud coseno [0, 1] por encima de la cual
                          un corte se considera ambiguo (default 0.6).
    window_size         : número de elementos a cada lado del corte ambiguo
                          que se incluyen en la revisión local (default 3).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from src.llm.client import LLMClient, get_default_client
from src.problem import Segmentation, SegmentationProblem
from src.solver.baseline import (
    build_coherence_matrix_cosine,
    compute_embeddings,
    dp_segmentation,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fase 1: segmentación inicial con coseno
# ──────────────────────────────────────────────────────────────────────────────

def _phase1(problem: SegmentationProblem, embedding_model: str) -> Tuple[Segmentation, np.ndarray, list]:
    """
    Ejecuta la fase 1: baseline coseno.

    Returns:
        (segmentacion_inicial, embeddings, coherence_matrix_cosine)
    """
    embeddings = compute_embeddings(problem, model_name=embedding_model)
    coh_matrix = build_coherence_matrix_cosine(embeddings)
    seg0 = dp_segmentation(problem, coh_matrix)
    return seg0, embeddings, coh_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Detección de cortes ambiguos
# ──────────────────────────────────────────────────────────────────────────────

def _find_ambiguous_cuts(
    cuts: List[int],
    embeddings: np.ndarray,
    ambiguity_threshold: float,
) -> List[int]:
    """
    Identifica cortes donde la frontera es temáticamente ambigua.

    Un corte en c es ambiguo si la similitud coseno entre E[c] y E[c+1]
    (los elementos a ambos lados del corte) supera el umbral.

    Args:
        cuts               : lista de cortes de la segmentación inicial
        embeddings         : embeddings normalizados (n, d)
        ambiguity_threshold: umbral de similitud coseno

    Returns:
        Lista de cortes ambiguos.
    """
    ambiguous = []
    for c in cuts:
        # c es el último índice del segmento izquierdo
        # c+1 es el primer índice del segmento derecho
        if c + 1 >= len(embeddings):
            continue
        sim = float(embeddings[c] @ embeddings[c + 1])  # coseno (normalizado)
        if sim > ambiguity_threshold:
            ambiguous.append(c)
    return ambiguous


# ──────────────────────────────────────────────────────────────────────────────
# Fase 2: refinamiento local con LLM
# ──────────────────────────────────────────────────────────────────────────────

def _refine_cut_with_llm(
    problem: SegmentationProblem,
    cut: int,
    window_size: int,
    client: LLMClient,
    coh_matrix_cosine: list,
) -> int:
    """
    Refina un corte ambiguo evaluando alternativas con el LLM.

    Considera todos los cortes posibles en la ventana [cut-window, cut+window]
    y elige el que maximiza:
        coherencia_llm(segmento_izq) + coherencia_llm(segmento_der)

    Args:
        problem          : instancia del problema
        cut              : corte actual a refinar
        window_size      : tamaño de la ventana a cada lado
        client           : cliente LLM
        coh_matrix_cosine: matriz de coherencias coseno (como fallback)

    Returns:
        El mejor corte alternativo encontrado.
    """
    n = problem.n
    lo = max(0, cut - window_size)
    hi = min(n - 2, cut + window_size)  # el corte no puede estar en n-1

    best_cut = cut
    best_score = float("-inf")

    for c in range(lo, hi + 1):
        # Evaluar coherencia del segmento izquierdo (lo..c) y derecho (c+1..hi+window)
        # Usamos rangos extendidos para capturar contexto
        left_end = c
        right_start = c + 1

        # Calcular la coherencia de ambos segmentos
        # Para el segmento izquierdo tomamos desde el inicio de la ventana
        score_left = client.ask_coherence_range(problem, lo, left_end)

        if right_start <= hi + window_size and right_start < n:
            score_right = client.ask_coherence_range(
                problem, right_start, min(hi + window_size, n - 1)
            )
        else:
            score_right = 0.5  # fallback si el segmento derecho está fuera de rango

        combined = score_left + score_right
        if combined > best_score:
            best_score = combined
            best_cut = c

    return best_cut


def _phase2(
    problem: SegmentationProblem,
    seg0: Segmentation,
    embeddings: np.ndarray,
    coh_matrix_cosine: list,
    client: LLMClient,
    ambiguity_threshold: float,
    window_size: int,
) -> Segmentation:
    """
    Ejecuta la fase 2: refinamiento de cortes ambiguos con LLM.

    Returns:
        Segmentación refinada.
    """
    cuts0 = list(seg0.cuts)
    ambiguous = _find_ambiguous_cuts(cuts0, embeddings, ambiguity_threshold)

    if not ambiguous:
        # Sin cortes ambiguos → la fase 1 ya es óptima
        seg0.config_name = "two_phase"
        return seg0

    refined_cuts = list(cuts0)
    for c in ambiguous:
        new_c = _refine_cut_with_llm(
            problem, c, window_size, client, coh_matrix_cosine
        )
        idx = refined_cuts.index(c)
        refined_cuts[idx] = new_c

    # Eliminar duplicados y ordenar
    refined_cuts = sorted(set(refined_cuts))

    # Calcular el nuevo score usando LLM para los segmentos afectados
    # (usamos coherencia coseno para el score de la segmentación final, por consistencia)
    seg1 = Segmentation(
        cuts=refined_cuts,
        n=problem.n,
        config_name="two_phase",
    )
    seg1.score = problem.score_segmentation(seg1, coh_matrix_cosine)

    return seg1


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada del solver two-phase
# ──────────────────────────────────────────────────────────────────────────────

def solve(
    problem: SegmentationProblem,
    client: Optional[LLMClient] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    ambiguity_threshold: float = 0.60,
    window_size: int = 3,
    show_progress: bool = True,
) -> Tuple[Segmentation, dict]:
    """
    Ejecuta el solver two-phase.

    Args:
        problem              : instancia del problema
        client               : cliente LLM (si None, se crea uno)
        embedding_model      : modelo de sentence-transformers para fase 1
        ambiguity_threshold  : umbral coseno para marcar un corte como ambiguo
        window_size          : elementos a cada lado del corte para la fase 2
        show_progress        : mostrar progreso

    Returns:
        (segmentation, stats)
    """
    if client is None:
        client = get_default_client(verbose=show_progress)

    client.reset_stats()
    t0 = time.perf_counter()

    # Fase 1
    t1_start = time.perf_counter()
    seg0, embeddings, coh_cosine = _phase1(problem, embedding_model)
    t1 = time.perf_counter() - t1_start

    # Fase 2
    t2_start = time.perf_counter()
    seg1 = _phase2(
        problem, seg0, embeddings, coh_cosine, client,
        ambiguity_threshold, window_size
    )
    t2 = time.perf_counter() - t2_start

    total_time = time.perf_counter() - t0
    llm_st = client.stats()

    stats = {
        "config": "two_phase",
        "instance": problem.name,
        "n": problem.n,
        "num_segments": seg1.num_segments(),
        "score": seg1.score,
        "time_total_s": round(total_time, 4),
        "time_phase1_s": round(t1, 4),
        "time_phase2_llm_s": round(t2, 4),
        "llm_calls": llm_st["real_llm_calls"],
        "cache_hits": llm_st["cache_stats"]["hits"],
        "model": llm_st["model"],
        "ambiguity_threshold": ambiguity_threshold,
        "window_size": window_size,
        "initial_cuts": seg0.cuts,
        "refined_cuts": seg1.cuts,
    }

    return seg1, stats
