"""
utils/youtube_downloader.py — Descarga de transcripciones directas desde enlaces de YouTube.

Utiliza `youtube-transcript-api` para descargar los subtítulos y el módulo
`transcript_parser` para agruparlos y formatearlos como un SegmentationProblem.
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.problem import SegmentationProblem
from src.utils.transcript_parser import convert_transcript_to_problem


def extract_youtube_video_id(url: str) -> Optional[str]:
    """
    Extrae el ID de 11 caracteres de un video de YouTube a partir de una URL.
    Soporta formatos estándar: youtube.com/watch?v=..., youtu.be/..., shorts, embeds, etc.
    """
    url = url.strip()
    # Si ya tiene exactamente 11 caracteres y no contiene diagonales o signos, asumimos que es el ID directo
    if len(url) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
        
    # Expresiones regulares comunes
    regexes = [
        r'(?:v=|\/embed\/|\/watch\?v=|\/\d{1,}\/|\/vi\/|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'^[a-zA-Z0-9_-]{11}$'
    ]
    
    for regex in regexes:
        m = re.search(regex, url)
        if m:
            return m.group(1)
            
    return None


import os
import requests
import http.cookiejar


def get_youtube_session() -> requests.Session:
    """
    Crea una sesión de requests configurada con cookies y proxies.
    Lee de .env o variables de entorno:
      - YOUTUBE_COOKIES_FILE (default: "cookies.txt" si existe)
      - HTTP_PROXY / HTTPS_PROXY
    """
    session = requests.Session()
    
    # 1. Configurar proxies
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    if proxies:
        session.proxies.update(proxies)
        
    # 2. Configurar cookies
    cookies_file = os.getenv("YOUTUBE_COOKIES_FILE") or "cookies.txt"
    if cookies_file and os.path.exists(cookies_file):
        try:
            cj = http.cookiejar.MozillaCookieJar(cookies_file)
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies = cj
        except Exception as e:
            # Imprimir error sutilmente
            pass
            
    return session


def download_youtube_transcript(video_id: str, languages: List[str] = ['es', 'en']) -> str:
    """
    Descarga la transcripción de un video de YouTube en los idiomas preferidos.
    
    Args:
        video_id  : ID de 11 caracteres del video
        languages : Lista ordenada de códigos de idioma preferidos (e.g. ['es', 'en'])
        
    Returns:
        Texto completo de la transcripción (con marcas de tiempo removidas).
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise ImportError(
            "youtube-transcript-api no está instalado. Ejecuta: pip install youtube-transcript-api"
        ) from e
        
    session = get_youtube_session()
    api = YouTubeTranscriptApi(http_client=session)
    
    try:
        # Intentar obtener la transcripción en los idiomas indicados
        entries = api.fetch(video_id, languages=languages)
    except Exception as e:
        # Si falla, intentar listar las transcripciones para buscar traducción
        try:
            transcript_list = api.list(video_id)
            # Buscar si se puede traducir al primer idioma de la lista
            for trans in transcript_list:
                if trans.is_translatable:
                    # Traducir al primer idioma de la lista
                    translated = trans.translate(languages[0])
                    entries = translated.fetch()
                    break
            else:
                raise e
        except Exception as inner_e:
            raise RuntimeError(
                f"No se pudo descargar la transcripción para el video '{video_id}'. "
                f"Es posible que no tenga subtítulos disponibles o habilitados. Error: {inner_e}"
            ) from e

    # Extraer el texto de las entradas y unirlos
    lines = [entry['text'] for entry in entries]
    return '\n'.join(lines)


def download_and_convert_youtube(
    url: str,
    target_word_count: int = 120,
    languages: List[str] = ['es', 'en'],
    lambda_pen: float = 0.50,
    min_seg: int = 2,
) -> SegmentationProblem:
    """
    Flujo de alto nivel para descargar una URL de YouTube y crear un SegmentationProblem.
    
    Args:
        url               : URL completa del video de YouTube o ID directo
        target_word_count : Cantidad aproximada de palabras por párrafo
        languages         : Idiomas a intentar descargar
        lambda_pen        : Penalización de corte
        min_seg           : Mínimo tamaño de segmento
        
    Returns:
        Instancia de SegmentationProblem.
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"URL de YouTube inválida o no se pudo extraer el ID de video de: {url}")
        
    raw_text = download_youtube_transcript(video_id, languages=languages)
    name = f"youtube_{video_id}"
    
    return convert_transcript_to_problem(
        text=raw_text,
        name=name,
        target_word_count=target_word_count,
        lambda_pen=lambda_pen,
        min_seg=min_seg
    )
