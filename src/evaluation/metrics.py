"""
evaluation/metrics.py — Métricas de evaluación de segmentaciones.

Métricas implementadas:
─────────────────────────────────────────────────────────────────────────────

1. intra_coherence(segmentation, coherence_matrix)
   Calidad interna: promedio de coherencias de cada segmento.
   ↑ Mayor es mejor.

2. inter_separation(segmentation, embeddings)
   Separación entre segmentos: distancia coseno promedio entre centroides
   de segmentos consecutivos.
   ↑ Mayor separación = segmentos más distintos = mejor.

3. boundary_f1(predicted_cuts, true_cuts, tolerance)
   Precisión y recall de la detección de fronteras vs. ground truth.
   Solo aplicable cuando true_cuts está disponible (instancias sintéticas).
   ↑ Mayor F1 = mejor alineación con el ground truth.

4. Métricas de costo:
   - llm_calls: número de llamadas reales al LLM.
   - time_total_s: tiempo de ejecución.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# 1. Coherencia interna
# ──────────────────────────────────────────────────────────────────────────────

def intra_coherence(
    segmentation,  # Segmentation
    coherence_matrix: List[List[float]],
) -> float:
    """
    Calcula la coherencia interna media de la segmentación.

    intra_coherence = (1/K) · Σ_j coherence_matrix[start_j][end_j]

    donde K = número de segmentos.

    Args:
        segmentation     : objeto Segmentation
        coherence_matrix : matriz n×n de coherencias

    Returns:
        float en [0, 1]. Mayor = mejor.
    """
    segments = segmentation.segments()
    if not segments:
        return 0.0
    scores = [coherence_matrix[start][end] for start, end in segments]
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Separación entre segmentos
# ──────────────────────────────────────────────────────────────────────────────

def inter_separation(
    segmentation,  # Segmentation
    embeddings: np.ndarray,
) -> float:
    """
    Calcula la separación entre segmentos consecutivos como distancia coseno
    entre sus centroides.

    separacion = (1/(K-1)) · Σ_j (1 - sim_coseno(centroide_j, centroide_{j+1}))

    Para K=1 (sin cortes) devuelve 0.0 (sin separación).

    Args:
        segmentation : objeto Segmentation
        embeddings   : numpy array (n, d) normalizado

    Returns:
        float en [0, 1]. Mayor = mayor separación entre temas = mejor.
    """
    segments = segmentation.segments()
    if len(segments) < 2:
        return 0.0

    centroids = []
    for start, end in segments:
        seg_embeddings = embeddings[start : end + 1]
        centroid = seg_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    separations = []
    for i in range(len(centroids) - 1):
        sim = float(centroids[i] @ centroids[i + 1])
        separations.append(1.0 - sim)  # distancia coseno = 1 - similitud

    return float(np.mean(separations))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Boundary F1 (vs. ground truth)
# ──────────────────────────────────────────────────────────────────────────────

def boundary_f1(
    predicted_cuts: List[int],
    true_cuts: List[int],
    tolerance: int = 1,
) -> Tuple[float, float, float]:
    """
    Calcula Precision, Recall y F1 de los cortes predichos vs. cortes reales.

    Un corte predicho se considera correcto (true positive) si existe un
    corte verdadero dentro de la ventana de tolerancia (±tolerance elementos).

    Args:
        predicted_cuts : lista de cortes predichos
        true_cuts      : lista de cortes del ground truth
        tolerance      : ventana en elementos (default 1)

    Returns:
        (precision, recall, f1) — todos en [0, 1].
    """
    if not predicted_cuts and not true_cuts:
        return 1.0, 1.0, 1.0
    if not predicted_cuts:
        return 0.0, 0.0, 0.0
    if not true_cuts:
        return 0.0, 0.0, 0.0

    true_set = set(true_cuts)
    matched_true = set()
    tp = 0

    for pc in predicted_cuts:
        # Buscar si hay un corte verdadero en la ventana [pc-tol, pc+tol]
        for delta in range(-tolerance, tolerance + 1):
            candidate = pc + delta
            if candidate in true_set and candidate not in matched_true:
                tp += 1
                matched_true.add(candidate)
                break

    precision = tp / len(predicted_cuts) if predicted_cuts else 0.0
    recall = tp / len(true_cuts) if true_cuts else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return round(precision, 4), round(recall, 4), round(f1, 4)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Informe completo de métricas
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    segmentation,          # Segmentation
    problem,               # SegmentationProblem
    coherence_matrix: List[List[float]],
    embeddings: np.ndarray,
    stats: dict,
    tolerance: int = 1,
) -> dict:
    """
    Calcula todas las métricas para una segmentación y las agrupa en un dict.

    Args:
        segmentation     : la segmentación a evaluar
        problem          : instancia del problema
        coherence_matrix : usada para intra_coherence
        embeddings       : usados para inter_separation
        stats            : dict de stats del solver (tiempo, llm_calls, etc.)
        tolerance        : para boundary_f1

    Returns:
        dict con todas las métricas, listo para guardarse como fila de CSV.
    """
    ic = intra_coherence(segmentation, coherence_matrix)
    sep = inter_separation(segmentation, embeddings)

    # Boundary F1 solo si hay ground truth
    prec, rec, f1 = 0.0, 0.0, 0.0
    has_gt = problem.true_cuts is not None
    if has_gt:
        prec, rec, f1 = boundary_f1(
            segmentation.cuts, problem.true_cuts, tolerance=tolerance
        )

    return {
        # Identificación
        "instance": problem.name,
        "config": segmentation.config_name,
        "n": problem.n,
        "lambda_pen": problem.lambda_pen,
        # Solución
        "num_segments": segmentation.num_segments(),
        "cuts": str(segmentation.cuts),
        "objective_score": round(segmentation.score, 6),
        # Calidad
        "intra_coherence": round(ic, 6),
        "inter_separation": round(sep, 6),
        # Ground truth (solo instancias sintéticas)
        "has_ground_truth": has_gt,
        "true_cuts": str(problem.true_cuts) if has_gt else None,
        "boundary_precision": prec if has_gt else None,
        "boundary_recall": rec if has_gt else None,
        "boundary_f1": f1 if has_gt else None,
        # Costo
        "time_total_s": stats.get("time_total_s"),
        "llm_calls": stats.get("llm_calls", 0),
        "cache_hits": stats.get("cache_hits", 0),
        "model": stats.get("model", "none"),
    }
