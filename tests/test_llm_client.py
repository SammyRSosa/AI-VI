"""
tests/test_llm_client.py — Tests unitarios para el cliente LLM y el caché.

Estos tests usan MOCKS del LLM real para no requerir credenciales ni conexión.
Ejecutar con: pytest tests/test_llm_client.py -v
"""

import json
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.cache import LLMCache
from src.llm.client import LLMClient
from src.llm.prompts import format_coherence_prompt, COHERENCE_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# Tests: LLMCache
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMCache:

    @pytest.fixture
    def cache(self, tmp_path):
        return LLMCache(cache_dir=str(tmp_path / "cache"))

    def test_miss_on_empty_cache(self, cache):
        result = cache.get("prompt de prueba", "model-test")
        assert result is None

    def test_put_and_get(self, cache):
        data = {"score": 0.85, "razon": "coherente"}
        cache.put("mi prompt", "mi-modelo", data)
        retrieved = cache.get("mi prompt", "mi-modelo")
        assert retrieved is not None
        assert retrieved["score"] == 0.85

    def test_cache_hit_increments_hits(self, cache):
        cache.put("p", "m", {"score": 0.5, "razon": "ok"})
        cache.get("p", "m")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss_increments_misses(self, cache):
        cache.get("no existe", "modelo")
        stats = cache.stats()
        assert stats["misses"] == 1

    def test_different_models_different_keys(self, cache):
        cache.put("prompt", "modelo-A", {"score": 0.9, "razon": "A"})
        cache.put("prompt", "modelo-B", {"score": 0.1, "razon": "B"})
        a = cache.get("prompt", "modelo-A")
        b = cache.get("prompt", "modelo-B")
        assert a["score"] == 0.9
        assert b["score"] == 0.1

    def test_clear_removes_all_files(self, cache):
        cache.put("p1", "m", {"score": 0.5, "razon": "r"})
        cache.put("p2", "m", {"score": 0.7, "razon": "r"})
        removed = cache.clear()
        assert removed == 2
        assert cache.get("p1", "m") is None

    def test_hit_rate_calculation(self, cache):
        cache.put("p", "m", {"score": 0.5, "razon": "ok"})
        cache.get("p", "m")   # hit
        cache.get("p", "m")   # hit
        cache.get("no", "m")  # miss
        stats = cache.stats()
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_cached_files_count(self, cache):
        assert cache.stats()["cached_files"] == 0
        cache.put("p1", "m", {"score": 0.1, "razon": "x"})
        cache.put("p2", "m", {"score": 0.2, "razon": "x"})
        assert cache.stats()["cached_files"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Prompts
# ──────────────────────────────────────────────────────────────────────────────

class TestPrompts:

    def test_coherence_prompt_contains_segment(self):
        text = "Este es el texto del segmento."
        prompt = format_coherence_prompt(text)
        assert text in prompt

    def test_coherence_prompt_is_string(self):
        prompt = format_coherence_prompt("cualquier texto")
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_coherence_prompt_truncates_long_text(self):
        long_text = "palabra " * 2000  # 2000 palabras
        prompt = format_coherence_prompt(long_text)
        assert "truncado" in prompt.lower()

    def test_coherence_prompt_has_json_instruction(self):
        prompt = format_coherence_prompt("texto corto")
        assert "JSON" in prompt or "json" in prompt.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Tests: LLMClient (con mock de litellm)
# ──────────────────────────────────────────────────────────────────────────────

class TestLLMClient:

    @pytest.fixture
    def client(self, tmp_path):
        return LLMClient(
            model="test/mock-model",
            temperature=0.0,
            max_tokens=128,
            cache_dir=str(tmp_path / "cache"),
            verbose=False,
        )

    def _make_mock_response(self, score: float, razon: str = "mock"):
        """Crea una respuesta mock de litellm."""
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({"score": score, "razon": razon})
        return mock_resp

    def test_ask_coherence_returns_float(self, client):
        with patch("litellm.completion") as mock_llm:
            mock_llm.return_value = self._make_mock_response(0.75)
            score, razon = client.ask_coherence("Texto de prueba.")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_ask_coherence_uses_cache(self, client):
        with patch("litellm.completion") as mock_llm:
            mock_llm.return_value = self._make_mock_response(0.8)
            # Primera llamada: real
            s1, _ = client.ask_coherence("Texto igual.")
            # Segunda llamada: debe venir del caché
            s2, _ = client.ask_coherence("Texto igual.")
        # litellm solo debe haberse llamado UNA vez
        assert mock_llm.call_count == 1
        assert s1 == s2

    def test_ask_coherence_score_clipped(self, client):
        with patch("litellm.completion") as mock_llm:
            # El LLM devuelve un score fuera de rango
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = json.dumps({"score": 1.5, "razon": "over"})
            mock_llm.return_value = mock_resp
            score, _ = client.ask_coherence("Texto con score alto.")
        assert score <= 1.0

    def test_ask_coherence_handles_invalid_json(self, client):
        with patch("litellm.completion") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = 'Texto sin JSON válido. score: 0.6'
            mock_llm.return_value = mock_resp
            score, razon = client.ask_coherence("Texto con respuesta rota.")
        # Debe parsear el score con regex o devolver el neutro 0.5
        assert 0.0 <= score <= 1.0

    def test_stats_tracking(self, client):
        with patch("litellm.completion") as mock_llm:
            mock_llm.return_value = self._make_mock_response(0.7)
            client.ask_coherence("Prompt A")
            client.ask_coherence("Prompt B")
            client.ask_coherence("Prompt A")  # cache hit
        stats = client.stats()
        assert stats["real_llm_calls"] == 2  # A y B (A segunda vez → caché)
        assert stats["total_requests"] == 3

    def test_parse_json_with_code_block(self, client):
        """Respuesta envuelta en ```json ... ```."""
        with patch("litellm.completion") as mock_llm:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = (
                '```json\n{"score": 0.65, "razon": "con bloque"}\n```'
            )
            mock_llm.return_value = mock_resp
            score, razon = client.ask_coherence("Texto con bloque JSON.")
        assert abs(score - 0.65) < 1e-6
