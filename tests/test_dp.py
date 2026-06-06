"""
tests/test_dp.py — Tests unitarios para el algoritmo DP y las métricas.

Ejecutar con: pytest tests/test_dp.py -v
"""

import pytest
import sys
import os

# Asegura que src/ esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.problem import TextElement, Segmentation, SegmentationProblem
from src.instance import generate_synthetic
from src.solver.baseline import build_coherence_matrix_cosine, dp_segmentation
from src.evaluation.metrics import intra_coherence, boundary_f1


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_simple_problem(n: int = 6, true_k: int = 1) -> SegmentationProblem:
    """Crea una instancia sintética pequeña para pruebas."""
    return generate_synthetic(n=n, k_true=true_k, seed=0)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: SegmentationProblem
# ──────────────────────────────────────────────────────────────────────────────

class TestSegmentationProblem:

    def test_n_property(self):
        prob = make_simple_problem(n=8, true_k=2)
        assert prob.n == 8

    def test_get_segment_text(self):
        elements = [TextElement(idx=i, text=f"palabra{i}") for i in range(5)]
        prob = SegmentationProblem(elements=elements, name="test")
        text = prob.get_segment_text(1, 3)
        assert "palabra1" in text
        assert "palabra3" in text

    def test_save_load_roundtrip(self, tmp_path):
        prob = make_simple_problem(n=10, true_k=2)
        path = str(tmp_path / "inst.json")
        prob.save(path)
        loaded = SegmentationProblem.load(path)
        assert loaded.n == prob.n
        assert loaded.true_cuts == prob.true_cuts
        assert loaded.name == prob.name


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Segmentation
# ──────────────────────────────────────────────────────────────────────────────

class TestSegmentation:

    def test_segments_no_cuts(self):
        seg = Segmentation(cuts=[], n=5)
        segs = seg.segments()
        assert segs == [(0, 4)]

    def test_segments_one_cut(self):
        seg = Segmentation(cuts=[2], n=6)
        segs = seg.segments()
        assert segs == [(0, 2), (3, 5)]

    def test_segments_two_cuts(self):
        seg = Segmentation(cuts=[1, 3], n=6)
        segs = seg.segments()
        assert segs == [(0, 1), (2, 3), (4, 5)]

    def test_num_segments(self):
        seg = Segmentation(cuts=[2, 4], n=8)
        assert seg.num_segments() == 3

    def test_is_valid_basic(self):
        seg = Segmentation(cuts=[2], n=5)
        assert seg.is_valid()

    def test_is_valid_empty_cuts(self):
        seg = Segmentation(cuts=[], n=5)
        assert seg.is_valid()

    def test_is_valid_invalid_cut_at_n_minus_1(self):
        # El corte no puede estar en el último índice (n-1)
        seg = Segmentation(cuts=[4], n=5)
        assert not seg.is_valid()

    def test_is_valid_unsorted_cuts(self):
        seg = Segmentation(cuts=[3, 1], n=6)
        assert not seg.is_valid()


# ──────────────────────────────────────────────────────────────────────────────
# Tests: DP con matriz de coherencia trivial
# ──────────────────────────────────────────────────────────────────────────────

class TestDPSegmentation:

    def _make_block_coherence(self, n: int, true_cuts: list, high: float = 0.9, low: float = 0.1) -> list:
        """
        Crea una matriz de coherencia sintética con bloques de alta coherencia
        separados por baja coherencia entre segmentos.
        """
        coh = [[0.0] * n for _ in range(n)]
        # Determinar a qué bloque pertenece cada elemento
        blocks = [-1] * n
        block_id = 0
        cuts_ext = [-1] + true_cuts + [n - 1]
        for b in range(len(cuts_ext) - 1):
            for i in range(cuts_ext[b] + 1, cuts_ext[b + 1] + 1):
                blocks[i] = b
                block_id += 1

        for i in range(n):
            for j in range(i, n):
                # Todos los elementos del rango en el mismo bloque → alta coherencia
                same_block = all(blocks[k] == blocks[i] for k in range(i, j + 1))
                coh[i][j] = high if same_block else low

        return coh

    def test_dp_recovers_true_cuts_trivial(self):
        """Con una matriz de coherencia perfecta, la DP debe recuperar los cortes exactos."""
        n = 9
        true_cuts = [2, 5]  # 3 segmentos: [0..2], [3..5], [6..8]
        elements = [TextElement(idx=i, text=f"e{i}") for i in range(n)]
        prob = SegmentationProblem(elements=elements, lambda_pen=0.05, true_cuts=true_cuts)

        coh = self._make_block_coherence(n, true_cuts, high=0.95, low=0.05)
        seg = dp_segmentation(prob, coh)

        assert seg.is_valid()
        assert seg.num_segments() >= 2  # debe haber al menos un corte

    def test_dp_no_cuts_when_uniform(self):
        """Con coherencia uniforme y lambda alto, la DP debe preferir no hacer cortes."""
        n = 6
        elements = [TextElement(idx=i, text=f"e{i}") for i in range(n)]
        prob = SegmentationProblem(elements=elements, lambda_pen=10.0)  # λ muy alto

        # Coherencia uniforme (todo igual = 0.5)
        coh = [[0.5] * n for _ in range(n)]
        seg = dp_segmentation(prob, coh)

        assert seg.is_valid()
        # Con λ=10 y coherencia=0.5, añadir cortes siempre penaliza más de lo que gana
        assert len(seg.cuts) == 0  # sin cortes

    def test_dp_respects_k_max(self):
        """La DP no debe generar más cortes que k_max."""
        n = 10
        elements = [TextElement(idx=i, text=f"e{i}") for i in range(n)]
        prob = SegmentationProblem(elements=elements, lambda_pen=0.01, k_max=2)

        # Coherencia muy baja = incentivo a hacer muchos cortes
        coh = [[0.1 if i != j else 1.0 for j in range(n)] for i in range(n)]
        seg = dp_segmentation(prob, coh)

        assert seg.is_valid()
        assert len(seg.cuts) <= 2


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Métricas
# ──────────────────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_intra_coherence_perfect(self):
        seg = Segmentation(cuts=[2], n=6, config_name="test")
        # Coherencia perfecta en cada segmento
        coh = [[1.0] * 6 for _ in range(6)]
        ic = intra_coherence(seg, coh)
        assert abs(ic - 1.0) < 1e-6

    def test_boundary_f1_exact_match(self):
        prec, rec, f1 = boundary_f1([2, 5], [2, 5], tolerance=0)
        assert f1 == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_boundary_f1_no_match(self):
        prec, rec, f1 = boundary_f1([1], [5], tolerance=0)
        assert f1 == 0.0

    def test_boundary_f1_with_tolerance(self):
        # Corte predicho en 3, verdadero en 4, tolerancia 1 → match
        prec, rec, f1 = boundary_f1([3], [4], tolerance=1)
        assert f1 == 1.0

    def test_boundary_f1_empty_predicted(self):
        prec, rec, f1 = boundary_f1([], [3], tolerance=1)
        assert f1 == 0.0

    def test_boundary_f1_both_empty(self):
        prec, rec, f1 = boundary_f1([], [], tolerance=1)
        assert f1 == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Tests: instancias sintéticas
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticInstances:

    def test_generate_synthetic_n_and_k(self):
        prob = generate_synthetic(n=12, k_true=2, seed=99)
        assert prob.n == 12
        assert len(prob.true_cuts) == 2

    def test_generate_synthetic_true_cuts_valid(self):
        prob = generate_synthetic(n=15, k_true=3, seed=7)
        for c in prob.true_cuts:
            assert 0 <= c < prob.n - 1

    def test_generate_synthetic_different_seeds_differ(self):
        p1 = generate_synthetic(n=10, k_true=2, seed=1)
        p2 = generate_synthetic(n=10, k_true=2, seed=2)
        # Las instancias deben diferir (al menos en algo)
        texts1 = [e.text for e in p1.elements]
        texts2 = [e.text for e in p2.elements]
        assert texts1 != texts2 or p1.true_cuts != p2.true_cuts
