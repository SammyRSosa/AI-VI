"""
solver/baseline.py — Solver baseline: Programación Dinámica con similitud coseno.

CONFIGURACIÓN A (sin LLM): Control experimental.

Algoritmo:
    1. Calcular embeddings de cada elemento con sentence-transformers.
    2. Pre-computar la matriz de coherencias coseno coherence[i][j] para todos
       los rangos [i, j].  Complejidad: O(n²).
    3. Ejecutar DP:
           dp[i] = max_{j ≤ i} { dp[j-1] + coherence[j][i] - λ }
       con dp[-1] = 0 como caso base.
       Complejidad: O(n²) tiempo y espacio.
    4. Reconstruir los cortes con backtracking sobre parent[i].

Complejidad total: O(n²) tiempo, O(n²) espacio (por la matriz de coherencias).

La coherencia de un rango [i, j] se define como el promedio de similitudes
coseno entre todos los pares de embeddings dentro del rango:
    coherence(i, j) = mean { cos_sim(e_a, e_b) : i ≤ a < b ≤ j }

Para j == i (segmento de un solo elemento) la coherencia es 1.0 por definición
(un solo elemento siempre es perfectamente coherente consigo mismo).
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.problem import Segmentation, SegmentationProblem


# ──────────────────────────────────────────────────────────────────────────────
# Cálculo de embeddings
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_CACHE = {}


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Carga el modelo de embeddings (con caché en memoria)."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"[baseline] Cargando modelo de embeddings '{model_name}' ...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def compute_embeddings(
    problem: SegmentationProblem,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Calcula los embeddings de todos los elementos del problema.

    Args:
        problem    : instancia del problema
        model_name : nombre del modelo de sentence-transformers
        batch_size : tamaño del batch para la inferencia

    Returns:
        numpy array de shape (n, embedding_dim)
    """
    # Si los embeddings ya están calculados y guardados en el problema, usarlos
    if all(e.embedding is not None for e in problem.elements):
        return np.array([e.embedding for e in problem.elements])

    model = _get_embedding_model(model_name)
    texts = [e.text for e in problem.elements]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # normalizar para que coseno = producto punto
    )
    # Guardar en el problema para reutilizar
    for elem, emb in zip(problem.elements, embeddings):
        elem.embedding = emb.tolist()

    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# Construcción de la matriz de coherencias
# ──────────────────────────────────────────────────────────────────────────────

def build_coherence_matrix_cosine(embeddings: np.ndarray) -> List[List[float]]:
    """
    Pre-computa la matriz de coherencias usando similitud coseno.

    coherence[i][j] = promedio de cosine_similarity(e_a, e_b) para i ≤ a < b ≤ j.
    Para i == j: coherence[i][j] = 1.0.

    Complejidad: O(n²) llamadas a similitud coseno, cada una O(d).
    Total: O(n² · d) donde d = dimensión del embedding.

    Args:
        embeddings : numpy array (n, d), normalizado.

    Returns:
        Matriz n×n como lista de listas (triangular superior).
    """
    n = len(embeddings)
    # Calcular la matriz de similitud coseno completa O(n²) de golpe
    sim_matrix = embeddings @ embeddings.T  # equivalente a cosine_similarity cuando normalizado

    coh = [[0.0] * n for _ in range(n)]

    for i in range(n):
        coh[i][i] = 1.0
        for j in range(i + 1, n):
            # Promedio de similitudes de todos los pares (a, b) con i ≤ a < b ≤ j
            # Extraemos la sub-matriz sim_matrix[i:j+1, i:j+1] y promediamos
            # solo la parte triangular superior.
            sub = sim_matrix[i : j + 1, i : j + 1]
            size = j - i + 1
            if size == 1:
                coh[i][j] = 1.0
            else:
                # Suma de la parte triangular superior (excluyendo diagonal)
                upper_sum = (sub.sum() - sub.trace()) / 2.0
                num_pairs = size * (size - 1) / 2
                coh[i][j] = float(upper_sum / num_pairs) if num_pairs > 0 else 1.0

    return coh


# ──────────────────────────────────────────────────────────────────────────────
# Algoritmo DP
# ──────────────────────────────────────────────────────────────────────────────

def dp_segmentation(
    problem: SegmentationProblem,
    coherence_matrix: List[List[float]],
) -> Segmentation:
    """
    Programación Dinámica para encontrar la segmentación óptima.

    Estado:
        dp[i] = mejor valor acumulado considerando los elementos 0..i

    Transición:
        dp[i] = max_{j : 0 ≤ j ≤ i} {
            (dp[j-1] if j > 0 else 0)
            + coherence_matrix[j][i]
            - (lambda_pen if j > 0 else 0)
        }

    La penalización λ se aplica por cada corte (= inicio de un nuevo segmento
    excepto el primero). Así, abrir un nuevo segmento en j > 0 cuesta λ.

    Condición k_max: si problem.k_max no es None, se rechaza cualquier
    transición que exceda el número máximo de cortes.

    Args:
        problem          : instancia con lambda_pen y k_max
        coherence_matrix : matriz pre-computada de coherencias

    Returns:
        Segmentation óptima.
    """
    n = problem.n
    lam = problem.lambda_pen
    k_max = problem.k_max  # None = sin límite
    min_seg = problem.min_seg

    INF = float("-inf")

    # dp[i]     : mejor valor acumulado para E[0..i]
    # parent[i] : j tal que el segmento actual empieza en j (o sea, j-1 es el corte)
    # ncuts[i]  : número de cortes en la solución que alcanza dp[i]

    dp = [INF] * n
    parent = [-1] * n
    ncuts = [0] * n

    for i in range(n):
        for j in range(i + 1):  # j = inicio del segmento actual
            seg_size = i - j + 1
            if seg_size < min_seg:
                continue

            # Número de cortes si el segmento empieza en j
            cuts_if_j = 0 if j == 0 else (ncuts[j - 1] + 1)

            # Verificar k_max
            if k_max is not None and cuts_if_j > k_max:
                continue

            # Valor acumulado hasta j-1
            prev = 0.0 if j == 0 else dp[j - 1]
            if prev == INF:
                continue

            # Penalización por añadir corte (si no es el primer segmento)
            penalty = lam if j > 0 else 0.0

            val = prev + coherence_matrix[j][i] - penalty

            if val > dp[i]:
                dp[i] = val
                parent[i] = j
                ncuts[i] = cuts_if_j

    # Reconstruir los cortes con backtracking
    cuts = []
    pos = n - 1
    while True:
        j = parent[pos]
        if j == 0:
            break
        cuts.append(j - 1)  # el corte es el ÚLTIMO índice del segmento anterior
        pos = j - 1

    cuts.reverse()

    return Segmentation(
        cuts=cuts,
        n=n,
        score=dp[n - 1],
        config_name="baseline_cosine",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada del solver baseline
# ──────────────────────────────────────────────────────────────────────────────

def solve(
    problem: SegmentationProblem,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[Segmentation, dict]:
    """
    Ejecuta el solver baseline completo.

    Returns:
        (segmentation, stats)
        stats contiene: tiempo total, tiempo de embeddings, tiempo de DP,
                        número de llamadas al LLM (siempre 0 en baseline).
    """
    t0 = time.perf_counter()

    # Paso 1: Embeddings
    t_emb0 = time.perf_counter()
    embeddings = compute_embeddings(problem, model_name=embedding_model)
    t_emb = time.perf_counter() - t_emb0

    # Paso 2: Matriz de coherencias
    t_coh0 = time.perf_counter()
    coh_matrix = build_coherence_matrix_cosine(embeddings)
    t_coh = time.perf_counter() - t_coh0

    # Paso 3: DP
    t_dp0 = time.perf_counter()
    seg = dp_segmentation(problem, coh_matrix)
    t_dp = time.perf_counter() - t_dp0

    total_time = time.perf_counter() - t0

    stats = {
        "config": "baseline_cosine",
        "instance": problem.name,
        "n": problem.n,
        "num_segments": seg.num_segments(),
        "score": seg.score,
        "time_total_s": round(total_time, 4),
        "time_embeddings_s": round(t_emb, 4),
        "time_coherence_matrix_s": round(t_coh, 4),
        "time_dp_s": round(t_dp, 4),
        "llm_calls": 0,
    }

    return seg, stats
