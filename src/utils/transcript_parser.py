"""
utils/transcript_parser.py — Parser y normalizador de archivos de transcripción.

Limpia marcas de tiempo, metadatos y comentarios, y agrupa líneas cortas
en párrafos uniformes adecuados para el segmentador temático.
"""

from __future__ import annotations

import re
from typing import List

from src.problem import TextElement, SegmentationProblem


def clean_transcript_line(line: str) -> str:
    """
    Limpia marcas de tiempo y etiquetas de una línea de transcripción.
    Soporta formatos:
      - 00:00:00.240
      - 00:00:03
      - [00:03] o (00:03)
      - Etiquetas como [Música], [Aplausos], (Risas)
    """
    # Eliminar marcas de tiempo al principio (con o sin corchetes/paréntesis)
    line = re.sub(
        r'^\s*[\(\[\]]?\s*\d{1,2}:\d{2}(?::\d{2})?(\.\d{3})?\s*[\)\]]?\s*',
        '',
        line
    )
    # Eliminar marcas de tiempo sueltas en cualquier parte del texto
    line = re.sub(
        r'\b\d{1,2}:\d{2}(?::\d{2})?(\.\d{3})?\b',
        '',
        line
    )
    # Eliminar etiquetas típicas de subtítulos como [Música], [Aplausos], etc.
    line = re.sub(r'\[[^\]]+\]|\([^\)]+\)', '', line)
    
    # Limpiar espacios múltiples y bordes
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def parse_transcript_text(text: str, target_word_count: int = 120) -> List[str]:
    """
    Parsea el texto completo de una transcripción.
    Elimina comentarios (#), limpia marcas de tiempo y agrupa líneas
    consecutivas hasta aproximarse al número objetivo de palabras.
    
    Args:
        text              : Contenido del archivo de transcripción
        target_word_count : Cantidad de palabras objetivo por párrafo (default 120)
        
    Returns:
        Lista de párrafos normalizados.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Saltar comentarios y líneas vacías
        if not line or line.startswith('#'):
            continue
            
        cleaned = clean_transcript_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)
            
    # Agrupar las líneas limpias en párrafos del tamaño deseado
    paragraphs = []
    current_paragraph_lines = []
    current_word_count = 0
    
    for line in cleaned_lines:
        words_in_line = len(line.split())
        if current_word_count > 0 and current_word_count + words_in_line > target_word_count:
            # Guardar párrafo actual
            paragraphs.append(' '.join(current_paragraph_lines))
            current_paragraph_lines = [line]
            current_word_count = words_in_line
        else:
            current_paragraph_lines.append(line)
            current_word_count += words_in_line
            
    # Añadir el último párrafo restante si existe
    if current_paragraph_lines:
        paragraphs.append(' '.join(current_paragraph_lines))
        
    return paragraphs


def convert_transcript_to_problem(
    text: str,
    name: str,
    target_word_count: int = 120,
    lambda_pen: float = 0.50,
    min_seg: int = 2,
) -> SegmentationProblem:
    """
    Convierte el texto de una transcripción en un objeto SegmentationProblem listo.
    
    Args:
        text              : Contenido del archivo de transcripción
        name              : Nombre identificador para la instancia
        target_word_count : Cantidad de palabras promedio por párrafo
        lambda_pen        : Penalización de corte (default 0.50)
        min_seg           : Mínimo tamaño de segmento (default 2)
        
    Returns:
        Instancia de SegmentationProblem.
    """
    paragraphs = parse_transcript_text(text, target_word_count=target_word_count)
    
    elements = [
        TextElement(idx=i, text=p)
        for i, p in enumerate(paragraphs)
    ]
    
    return SegmentationProblem(
        elements=elements,
        name=name,
        lambda_pen=lambda_pen,
        k_max=None,
        min_seg=min_seg,
        true_cuts=None
    )
