"""
llm/client.py — Cliente LLM agnóstico de proveedor.

Soporta: Google Gemini, Groq (Llama), OpenAI, Anthropic.
Usa `litellm` como capa de abstracción.

Patrón de integración: #2 — LLM como Evaluador.
El método principal es `ask_coherence(segment_text)` que devuelve un float [0, 1].

Manejo de errores:
    - Retry con backoff exponencial (3 intentos).
    - Si el JSON de respuesta es inválido → re-parseo con extracción por regex.
    - Si todo falla → devuelve score neutro (0.5) con advertencia.

Configuración:
    Se lee de variables de entorno (.env):
        LLM_PROVIDER    : "gemini" | "groq" | "openai" | "anthropic"
        LLM_MODEL       : nombre del modelo (e.g. "gemini/gemini-1.5-flash")
        LLM_TEMPERATURE : float (default 0.0)
        LLM_MAX_TOKENS  : int (default 256)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Tuple

from dotenv import load_dotenv

from src.llm.cache import LLMCache
from src.llm.prompts import format_coherence_prompt, format_best_cut_prompt

load_dotenv(override=True)


# ──────────────────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini/gemini-3.1-flash-lite")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))


# ──────────────────────────────────────────────────────────────────────────────
# Cliente LLM
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Cliente LLM con caché, reintentos y parsing robusto.

    Args:
        model       : identificador del modelo (formato litellm)
        temperature : temperatura de muestreo
        max_tokens  : tokens máximos en la respuesta
        cache_dir   : directorio para el caché en disco (None = usar .env)
        verbose     : si True, imprime información de cada llamada
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.cache = LLMCache(cache_dir=cache_dir)
        self._call_count = 0  # llamadas reales al LLM (sin contar caché)
        self._total_requests = 0  # total incluyendo caché

    # ── Llamada de bajo nivel ─────────────────────────────────────────────────

    def _call_llm_raw(self, prompt: str) -> str:
        """
        Llama al LLM y devuelve el texto de respuesta.
        Implementa retry con backoff exponencial.
        """
        try:
            import litellm
        except ImportError as e:
            raise ImportError(
                "litellm no está instalado. Ejecuta: pip install litellm"
            ) from e

        last_exc = None
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    request_timeout=30,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                last_exc = exc
                # Si es un error de rate limit (429 / RESOURCE_EXHAUSTED), esperar más tiempo
                is_rate_limit = (
                    "429" in str(exc) 
                    or "RESOURCE_EXHAUSTED" in str(exc) 
                    or "rate_limit" in str(type(exc)).lower() 
                    or "ratelimit" in str(type(exc)).lower()
                )
                wait = 15 if is_rate_limit else (2 ** attempt)
                if self.verbose:
                    print(f"[llm] Intento {attempt+1}/{max_attempts} fallido: {exc}. Esperando {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"LLM falló después de {max_attempts} intentos. Último error: {last_exc}")

    # ── Parsing de la respuesta JSON ──────────────────────────────────────────

    @staticmethod
    def _parse_coherence_response(raw: str) -> Tuple[float, str]:
        """
        Parsea la respuesta JSON del LLM para el prompt de coherencia.

        Returns:
            (score: float, razon: str)

        Si el JSON es inválido, intenta extraer el score con regex.
        Si todo falla, devuelve (0.5, "parse_error").
        """
        # Intentar parsing JSON directo
        try:
            # Eliminar posibles bloques ```json ... ```
            clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            data = json.loads(clean)
            score = float(data["score"])
            score = max(0.0, min(1.0, score))  # clip a [0, 1]
            razon = str(data.get("razon", ""))
            return score, razon
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback: buscar el score con regex
        m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', raw)
        if m:
            score = float(m.group(1))
            score = max(0.0, min(1.0, score))
            return score, "parse_fallback_regex"

        # Si todo falla, score neutro
        return 0.5, "parse_error"

    # ── Método principal: evaluación de coherencia ────────────────────────────

    def ask_coherence(self, segment_text: str) -> Tuple[float, str]:
        """
        Evalúa la coherencia semántica de un segmento de texto.

        Este es el método central del patrón Evaluador (rol #2 del LLM).
        Primero consulta el caché; si no está, llama al LLM y cachea.

        Args:
            segment_text : texto concatenado del segmento a evaluar

        Returns:
            (score, razon)
            score ∈ [0.0, 1.0]  — mayor = más coherente
            razon                — justificación breve del LLM
        """
        prompt = format_coherence_prompt(segment_text)
        self._total_requests += 1

        # Consultar caché
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            if self.verbose:
                print(f"[llm] CACHE HIT | score={cached['score']:.3f}")
            return cached["score"], cached["razon"]

        # Llamada real al LLM
        self._call_count += 1
        if self.verbose:
            snippet = segment_text[:80].replace("\n", " ")
            print(f"[llm] CALL #{self._call_count} | '{snippet}...'")

        raw = self._call_llm_raw(prompt)
        score, razon = self._parse_coherence_response(raw)

        # Guardar en caché
        if razon != "parse_error":
            self.cache.put(prompt, self.model, {"score": score, "razon": razon, "raw": raw})

        if self.verbose:
            print(f"[llm] score={score:.3f} | {razon}")

        return score, razon

    def ask_coherence_range(
        self,
        problem,  # SegmentationProblem
        start: int,
        end: int,
        separator: str = "\n\n",
    ) -> float:
        """
        Evalúa la coherencia del rango E[start..end] del problema.

        Conveniencia que extrae el texto del problema y llama a ask_coherence.

        Returns:
            score ∈ [0.0, 1.0]
        """
        text = problem.get_segment_text(start, end, separator=separator)
        score, _ = self.ask_coherence(text)
        return score

    def _parse_best_cut_response(self, raw: str, lo: int, hi: int) -> Tuple[int, str]:
        """
        Parsea la respuesta JSON del LLM para el prompt de mejor corte.
        """
        try:
            # Eliminar posibles bloques ```json ... ```
            clean = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            data = json.loads(clean)
            best_cut = int(data["best_cut"])
            # Asegurar que el corte esté dentro del rango de opciones candidato
            best_cut = max(lo, min(hi, best_cut))
            razon = str(data.get("razon", ""))
            return best_cut, razon
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Fallback con regex
        m = re.search(r'"best_cut"\s*:\s*(\d+)', raw)
        if m:
            best_cut = int(m.group(1))
            best_cut = max(lo, min(hi, best_cut))
            return best_cut, "parse_fallback_regex"

        # Si todo falla, devolvemos el punto medio como fallback seguro
        return (lo + hi) // 2, "parse_error"

    def ask_best_cut(self, problem, lo: int, hi: int, window_size: int) -> Tuple[int, str]:
        """
        Pide al LLM encontrar el mejor corte en el rango [lo, hi] en una sola llamada.
        """
        prompt = format_best_cut_prompt(problem, lo, hi, window_size)
        self._total_requests += 1

        # Consultar caché
        cached = self.cache.get(prompt, self.model)
        if cached is not None:
            if self.verbose:
                print(f"[llm] CACHE HIT (best_cut) | cut={cached['best_cut']}")
            return int(cached["best_cut"]), cached["razon"]

        # Llamada real al LLM
        self._call_count += 1
        if self.verbose:
            print(f"[llm] CALL #{self._call_count} (best_cut) | rango=[{lo}..{hi}]")

        raw = self._call_llm_raw(prompt)
        best_cut, razon = self._parse_best_cut_response(raw, lo, hi)

        # Guardar en caché
        if razon != "parse_error":
            self.cache.put(prompt, self.model, {"best_cut": best_cut, "razon": razon, "raw": raw})

        if self.verbose:
            print(f"[llm] best_cut={best_cut} | {razon}")

        return best_cut, razon

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "model": self.model,
            "real_llm_calls": self._call_count,
            "total_requests": self._total_requests,
            "cache_stats": self.cache.stats(),
        }

    def reset_stats(self) -> None:
        self._call_count = 0
        self._total_requests = 0


# ──────────────────────────────────────────────────────────────────────────────
# Inicialización desde .env
# ──────────────────────────────────────────────────────────────────────────────

def get_default_client(verbose: bool = False) -> LLMClient:
    """
    Crea un LLMClient con la configuración del .env.
    Función de conveniencia para usar en los solvers.
    """
    return LLMClient(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        verbose=verbose,
    )
