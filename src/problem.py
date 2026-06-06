"""
problem.py — Definición formal del problema de Segmentación Óptima de Contenido.

PROBLEMA FORMAL (Tema 6):
    Dada una secuencia E = [e_1, ..., e_n] de elementos de texto, encontrar un
    conjunto de cortes C = {c_1 < c_2 < ... < c_k} con c_i ∈ {1, ..., n-1}
    que induce la partición:

        S_0 = E[0 .. c_1]
        S_1 = E[c_1+1 .. c_2]
        ...
        S_k = E[c_k+1 .. n-1]

    tal que se maximiza la función objetivo:

        f(C) = Σ_j coherencia(S_j) - λ · |C|

    donde:
        coherencia(S_j) ∈ [0, 1]  — unidad temática del segmento j
        λ ≥ 0                      — penalización por segmento adicional (hiperparámetro)
        |C|                        — número de cortes (= número de segmentos - 1)

RESTRICCIONES:
    - Los segmentos son contiguos y no vacíos.
    - |C| ≤ k_max  (número máximo de cortes, por defecto sin límite)
    - Cada segmento tiene al menos min_seg_size elementos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import json


# ──────────────────────────────────────────────────────────────────────────────
# Tipos base
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextElement:
    """Un elemento de la secuencia (e.g. un párrafo)."""
    idx: int           # índice 0-based en la secuencia
    text: str          # texto del elemento
    # Embedding pre-computado (numpy array serializado como lista); puede ser None
    embedding: Optional[List[float]] = None

    def __repr__(self) -> str:
        snippet = self.text[:60].replace("\n", " ")
        return f"TextElement(idx={self.idx}, text='{snippet}...')"


@dataclass
class Segmentation:
    """
    Una solución: conjunto de cortes sobre la secuencia E.

    cuts = [c_1, c_2, ..., c_k]  con c_i ∈ {0, ..., n-2}
    El segmento j va desde cuts[j-1]+1 hasta cuts[j] (inclusive),
    usando centinelas cuts[-1]=-1 y cuts[k]=n-1.

    Ejemplo con n=6 y cuts=[1, 3]:
        Segmento 0: E[0..1]
        Segmento 1: E[2..3]
        Segmento 2: E[4..5]
    """
    cuts: List[int]             # índices de corte (último elemento de cada segmento)
    n: int                      # longitud total de la secuencia
    score: float = 0.0          # valor de la función objetivo
    config_name: str = ""       # nombre de la configuración que la produjo

    def segments(self) -> List[Tuple[int, int]]:
        """
        Devuelve lista de (inicio, fin) para cada segmento (ambos inclusive).
        """
        boundaries = [-1] + self.cuts + [self.n - 1]
        result = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1
            end = boundaries[i + 1]
            if start <= end:
                result.append((start, end))
        return result

    def num_segments(self) -> int:
        return len(self.cuts) + 1

    def is_valid(self, min_seg_size: int = 1) -> bool:
        """Verifica que la segmentación sea válida."""
        if not self.cuts:
            return True  # sin cortes = un solo segmento, válido
        # cortes estrictamente crecientes y dentro de rango
        if self.cuts != sorted(self.cuts):
            return False
        if self.cuts[0] < 0 or self.cuts[-1] >= self.n - 1:
            return False
        # tamaño mínimo de segmento
        for start, end in self.segments():
            if (end - start + 1) < min_seg_size:
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "cuts": self.cuts,
            "n": self.n,
            "score": self.score,
            "config_name": self.config_name,
            "segments": [{"start": s, "end": e} for s, e in self.segments()],
            "num_segments": self.num_segments(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Problema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentationProblem:
    """
    Encapsula una instancia del problema de segmentación óptima.

    Atributos:
        elements   : secuencia de TextElement
        lambda_pen : λ — penalización por corte adicional
        k_max      : número máximo de cortes (None = sin límite)
        min_seg    : tamaño mínimo de un segmento en elementos
        name       : identificador de la instancia
        true_cuts  : cortes de referencia (ground truth), si existen
    """
    elements: List[TextElement]
    lambda_pen: float = 0.1
    k_max: Optional[int] = None
    min_seg: int = 1
    name: str = "instancia"
    true_cuts: Optional[List[int]] = None  # para instancias sintéticas

    def __post_init__(self):
        # Validar que los índices son consecutivos
        for i, e in enumerate(self.elements):
            if e.idx != i:
                raise ValueError(
                    f"TextElement en posición {i} tiene idx={e.idx}. "
                    "Los índices deben ser consecutivos 0, 1, 2, ..."
                )

    @property
    def n(self) -> int:
        return len(self.elements)

    def get_segment_text(self, start: int, end: int, separator: str = " ") -> str:
        """Concatena el texto de E[start..end] (inclusive)."""
        return separator.join(e.text for e in self.elements[start : end + 1])

    def score_segmentation(self, seg: Segmentation, coherence_matrix: List[List[float]]) -> float:
        """
        Calcula f(C) = Σ_j coherencia(S_j) - λ · |C|

        Args:
            seg              : Segmentation a evaluar
            coherence_matrix : coherence_matrix[i][j] = coherencia de E[i..j]
                               (matriz triangular superior, i ≤ j)

        Returns:
            Valor de la función objetivo.
        """
        total = 0.0
        for start, end in seg.segments():
            total += coherence_matrix[start][end]
        total -= self.lambda_pen * len(seg.cuts)
        return total

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "lambda_pen": self.lambda_pen,
            "k_max": self.k_max,
            "min_seg": self.min_seg,
            "true_cuts": self.true_cuts,
            "elements": [
                {"idx": e.idx, "text": e.text, "embedding": e.embedding}
                for e in self.elements
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentationProblem":
        elements = [
            TextElement(idx=e["idx"], text=e["text"], embedding=e.get("embedding"))
            for e in d["elements"]
        ]
        return cls(
            elements=elements,
            lambda_pen=d.get("lambda_pen", 0.1),
            k_max=d.get("k_max"),
            min_seg=d.get("min_seg", 1),
            name=d.get("name", "instancia"),
            true_cuts=d.get("true_cuts"),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SegmentationProblem":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
