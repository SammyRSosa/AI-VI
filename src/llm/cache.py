"""
llm/cache.py — Caché en disco para respuestas del LLM.

Motivación:
    Las llamadas al LLM son costosas (tiempo y dinero/cuota). Durante el
    desarrollo se repiten los mismos prompts al depurar o re-ejecutar
    experimentos. Un caché simple en disco (clave = SHA-256 del prompt + modelo)
    evita llamadas redundantes y garantiza la reproducibilidad de los resultados.

Diseño:
    - Cada entrada del caché es un archivo JSON.
    - El nombre del archivo es SHA-256(modelo::prompt).
    - El archivo contiene el objeto de respuesta completo (score, razon, raw).
    - El directorio de caché se configura via variable de entorno LLM_CACHE_DIR.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


class LLMCache:
    """
    Caché en disco para respuestas del LLM.

    Uso:
        cache = LLMCache()
        cached = cache.get(prompt, model)
        if cached is None:
            response = call_llm(prompt)
            cache.put(prompt, model, response)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            cache_dir = os.getenv("LLM_CACHE_DIR", "data/llm_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key(self, prompt: str, model: str) -> str:
        """Calcula la clave SHA-256 para el par (model, prompt)."""
        content = f"{model}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str, model: str) -> Optional[Any]:
        """
        Busca la respuesta en caché.

        Returns:
            El objeto guardado si existe, None si no.
        """
        path = self._path(self._key(prompt, model))
        if path.exists():
            self._hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        self._misses += 1
        return None

    def put(self, prompt: str, model: str, response: Any) -> None:
        """
        Guarda una respuesta en caché.

        Args:
            prompt   : el prompt completo enviado al LLM
            model    : identificador del modelo (e.g. "gemini/gemini-1.5-flash")
            response : el objeto de respuesta a cachear (serializable a JSON)
        """
        path = self._path(self._key(prompt, model))
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> dict:
        """Estadísticas de hits/misses del caché."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": round(hit_rate, 3),
            "cache_dir": str(self.cache_dir),
            "cached_files": len(list(self.cache_dir.glob("*.json"))),
        }

    def clear(self) -> int:
        """Elimina todos los archivos del caché. Devuelve el número eliminado."""
        files = list(self.cache_dir.glob("*.json"))
        for f in files:
            f.unlink()
        self._hits = 0
        self._misses = 0
        return len(files)
