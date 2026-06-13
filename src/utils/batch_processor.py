"""
utils/batch_processor.py — Procesamiento en lote para normalizar múltiples transcripciones.

Permite escanear un directorio, procesar archivos .txt locales, limpiarlos
y guardarlos como instancias JSON válidas en data/instances/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from src.instance import save_instance
from src.utils.transcript_parser import convert_transcript_to_problem


def process_single_transcript_file(
    file_path: str,
    output_dir: str = "data/instances",
    target_word_count: int = 120,
    lambda_pen: float = 0.50,
    min_seg: int = 2,
) -> str:
    """
    Procesa un único archivo .txt local de transcripción y lo guarda como JSON.
    
    Returns:
        Ruta del archivo JSON generado.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"El archivo de entrada no existe: {file_path}")
        
    # Nombre de la instancia derivado del nombre del archivo (limpiando caracteres extraños)
    instance_name = input_path.stem
    instance_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in instance_name)
    instance_name = instance_name.strip('_')
    
    # Leer el archivo de texto plano
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        
    # Convertir a SegmentationProblem
    problem = convert_transcript_to_problem(
        text=text,
        name=instance_name,
        target_word_count=target_word_count,
        lambda_pen=lambda_pen,
        min_seg=min_seg
    )
    
    # Asegurar directorio de salida y guardar
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{instance_name}.json"
    
    save_instance(problem, str(out_path))
    return str(out_path)


def batch_process_transcripts(
    input_dir: str,
    output_dir: str = "data/instances",
    target_word_count: int = 120,
    lambda_pen: float = 0.50,
    min_seg: int = 2,
) -> List[str]:
    """
    Escanea un directorio en busca de archivos .txt, los procesa y los exporta a JSON.
    
    Returns:
        Lista de rutas de archivos JSON generados con éxito.
    """
    in_dir = Path(input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"El directorio de entrada no existe: {input_dir}")
        
    generated_files = []
    
    # Encontrar todos los archivos .txt en el directorio de entrada (sin recursión profunda)
    for file_path in in_dir.glob("*.txt"):
        try:
            out_path = process_single_transcript_file(
                file_path=str(file_path),
                output_dir=output_dir,
                target_word_count=target_word_count,
                lambda_pen=lambda_pen,
                min_seg=min_seg
            )
            generated_files.append(out_path)
        except Exception as e:
            print(f"[batch] Error al procesar '{file_path.name}': {e}")
            
    return generated_files
