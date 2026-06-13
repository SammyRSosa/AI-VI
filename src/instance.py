"""
instance.py — Carga y generación de instancias del problema de segmentación.

Fuentes de datos:
    1. Project Gutenberg (libros de dominio público).
    2. Instancias sintéticas con ground truth conocido (para evaluar correctitud).
    3. Carga/guardado en formato JSON.
"""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from src.problem import SegmentationProblem, TextElement


# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

GUTENBERG_BASE = "https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

# Libros sugeridos (dominio público, en inglés o español, con estructura temática clara):
# 1342  → Pride and Prejudice (Austen) — inglés, narrativa, ≈500 párrafos
# 2701  → Moby Dick (Melville)         — inglés, narrativa, ≈600 párrafos
# 84    → Frankenstein (Shelley)       — inglés, narrativa, ≈300 párrafos
# 11    → Alice's Adventures           — inglés, narrativa, ≈200 párrafos
GUTENBERG_BOOKS = {
    "pride_and_prejudice": 1342,
    "moby_dick": 2701,
    "frankenstein": 84,
    "alice": 11,
}


# ──────────────────────────────────────────────────────────────────────────────
# Descarga y limpieza de Project Gutenberg
# ──────────────────────────────────────────────────────────────────────────────

def _download_gutenberg_text(book_id: int, cache_dir: str = "data/gutenberg_cache") -> str:
    """
    Descarga el texto plano de un libro de Project Gutenberg.
    Guarda en caché local para no re-descargar.

    Returns:
        Texto completo como string.
    """
    cache_path = Path(cache_dir) / f"pg{book_id}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = GUTENBERG_BASE.format(book_id=book_id)
    print(f"[instance] Descargando libro {book_id} desde {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text
    cache_path.write_text(text, encoding="utf-8")
    print(f"[instance] Guardado en caché: {cache_path}")
    return text


def _clean_gutenberg(raw_text: str) -> str:
    """
    Elimina cabeceras/pies de Gutenberg y normaliza el texto.
    """
    # Buscar el inicio real del texto (después del encabezado de Gutenberg)
    start_patterns = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK",
        r"Produced by",
    ]
    start_idx = 0
    for pat in start_patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            start_idx = m.end()
            break

    # Buscar el final real del texto
    end_patterns = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
        r"End of Project Gutenberg",
    ]
    end_idx = len(raw_text)
    for pat in end_patterns:
        m = re.search(pat, raw_text, re.IGNORECASE)
        if m:
            end_idx = m.start()
            break

    text = raw_text[start_idx:end_idx]
    # Normalizar caracteres unicode
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def _split_into_paragraphs(text: str, min_words: int = 20) -> List[str]:
    """
    Divide el texto en párrafos por líneas en blanco.
    Filtra párrafos muy cortos (títulos de capítulo, etc.).
    """
    # Dividir por doble salto de línea (párrafo)
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw_paragraphs:
        # Limpiar espacios y saltos internos
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        # Filtrar párrafos muy cortos
        word_count = len(p.split())
        if word_count < min_words:
            continue
        paragraphs.append(p)
    return paragraphs


# ──────────────────────────────────────────────────────────────────────────────
# API pública: carga desde Gutenberg
# ──────────────────────────────────────────────────────────────────────────────

def load_gutenberg(
    book_name: str,
    max_elements: Optional[int] = None,
    min_words_per_para: int = 20,
    cache_dir: str = "data/gutenberg_cache",
) -> List[TextElement]:
    """
    Carga un libro de Project Gutenberg y devuelve su lista de TextElement.

    Args:
        book_name       : clave del libro (ver GUTENBERG_BOOKS) o un book_id int.
        max_elements    : si se especifica, toma solo los primeros N párrafos.
        min_words_per_para: mínimo de palabras para considerar un párrafo válido.
        cache_dir       : directorio de caché local.

    Returns:
        Lista de TextElement (sin embeddings; los embeds se calculan aparte).
    """
    if isinstance(book_name, str):
        if book_name not in GUTENBERG_BOOKS:
            raise ValueError(
                f"Libro '{book_name}' no encontrado. Disponibles: {list(GUTENBERG_BOOKS.keys())}"
            )
        book_id = GUTENBERG_BOOKS[book_name]
    else:
        book_id = int(book_name)

    raw = _download_gutenberg_text(book_id, cache_dir)
    clean = _clean_gutenberg(raw)
    paragraphs = _split_into_paragraphs(clean, min_words=min_words_per_para)

    if max_elements is not None:
        paragraphs = paragraphs[:max_elements]

    elements = [TextElement(idx=i, text=p) for i, p in enumerate(paragraphs)]
    print(f"[instance] '{book_name}': {len(elements)} párrafos cargados.")
    return elements


# ──────────────────────────────────────────────────────────────────────────────
# Instancias sintéticas con ground truth
# ──────────────────────────────────────────────────────────────────────────────

# Plantillas de párrafos por tema para instancias sintéticas.
# Cada lista contiene fragmentos de texto temáticamente coherentes.
_TOPIC_TEMPLATES = {
    "astronomia": [
        "Las estrellas son esferas de plasma que generan energía mediante la fusión nuclear en su núcleo.",
        "La Vía Láctea es una galaxia espiral barrada que contiene entre 200 y 400 mil millones de estrellas.",
        "Los agujeros negros son regiones del espacio-tiempo donde la gravedad es tan intensa que ni la luz puede escapar.",
        "Los planetas del sistema solar orbitan el Sol en trayectorias elípticas según las leyes de Kepler.",
        "Los cometas son cuerpos celestes formados por hielo y roca que desarrollan una coma al acercarse al Sol.",
        "La nebulosa es una nube interestelar de polvo, hidrógeno, helio y otros gases ionizados.",
        "Los pulsares son estrellas de neutrones que emiten radiación electromagnética periódica con gran precisión.",
    ],
    "gastronomia": [
        "La cocina mediterránea se caracteriza por el uso del aceite de oliva, las verduras frescas y las legumbres.",
        "El proceso de fermentación del pan transforma los azúcares en dióxido de carbono y alcohol mediante levaduras.",
        "Las especias como la canela, el cardamomo y el azafrán han sido artículos de comercio desde la antigüedad.",
        "La técnica del sous vide consiste en cocinar alimentos envasados al vacío en agua a temperatura controlada.",
        "Los quesos artesanales se producen mediante la coagulación de la leche usando cuajo y cultivos bacterianos.",
        "El umami es el quinto sabor básico, presente en alimentos ricos en glutamato como el tomate y el queso.",
        "El chocolate proviene del cacao, cultivado principalmente en África Occidental y América Central.",
    ],
    "historia": [
        "La Revolución Francesa de 1789 supuso el fin del Antiguo Régimen y el ascenso de los valores ilustrados.",
        "El Imperio Romano alcanzó su máxima extensión bajo el gobierno del emperador Trajano en el siglo II d.C.",
        "La Ruta de la Seda conectó Asia, Oriente Próximo y Europa facilitando el intercambio comercial y cultural.",
        "La Primera Guerra Mundial comenzó en 1914 tras el asesinato del archiduque Francisco Fernando en Sarajevo.",
        "La invención de la imprenta por Gutenberg en el siglo XV democratizó el acceso al conocimiento escrito.",
        "La Revolución Industrial transformó las estructuras económicas y sociales de Europa a partir del siglo XVIII.",
        "La caída del muro de Berlín en 1989 marcó el inicio del fin de la Guerra Fría y la reunificación alemana.",
    ],
    "informatica": [
        "Los algoritmos de búsqueda binaria reducen la complejidad de búsqueda en listas ordenadas a O(log n).",
        "La programación orientada a objetos organiza el código en clases que encapsulan datos y comportamiento.",
        "Las redes neuronales artificiales están inspiradas en la estructura del cerebro biológico.",
        "La complejidad computacional estudia los recursos necesarios para resolver problemas computacionales.",
        "Los compiladores traducen el código fuente de alto nivel a código máquina ejecutable por el procesador.",
        "La criptografía asimétrica usa pares de claves pública y privada para garantizar la seguridad de las comunicaciones.",
        "Las bases de datos relacionales organizan la información en tablas vinculadas mediante claves foráneas.",
    ],
    "biologia": [
        "La fotosíntesis es el proceso por el cual las plantas convierten la luz solar en glucosa y oxígeno.",
        "El ADN es una molécula de doble hélice que contiene la información genética de todos los seres vivos.",
        "La mitosis es el proceso de división celular que produce dos células hijas con el mismo número de cromosomas.",
        "Los ecosistemas son sistemas formados por comunidades de organismos y su entorno físico y químico.",
        "La evolución por selección natural es el mecanismo propuesto por Darwin para explicar la diversidad biológica.",
        "Los virus son entidades biológicas acelulares que requieren células huésped para replicarse.",
        "Las enzimas son proteínas que actúan como catalizadores en las reacciones bioquímicas de los organismos.",
    ],
}


_TRANSITION_FILLERS = [
    "Cambiando de tema por un momento.",
    "Ahora bien, pasemos a considerar otro punto de vista.",
    "Como se ha venido mencionando en las secciones anteriores.",
    "Veamos esto con más detalle a continuación.",
    "Dejando a un lado este aspecto, hay otro factor importante.",
    "Volviendo a retomar el hilo principal de nuestra discusión.",
    "En otro orden de ideas, cabe señalar lo siguiente.",
    "A continuación se presentan algunos aspectos complementarios.",
]


def generate_synthetic(
    n: int,
    k_true: int,
    topics: Optional[List[str]] = None,
    seed: int = 42,
    noise_ratio: float = 0.1,
    noise_type: str = "topic_swap",  # "topic_swap" o "filler_injection"
) -> SegmentationProblem:
    """
    Genera una instancia sintética con cortes verdaderos conocidos.

    Args:
        n           : número total de elementos en la secuencia
        k_true      : número de cortes verdaderos (= k_true+1 segmentos)
        topics      : lista de temas a usar (se toman k_true+1 aleatorios si None)
        seed        : semilla aleatoria
        noise_ratio : fracción de elementos que se asignan a un tema incorrecto (ruido)
        noise_type  : tipo de ruido a inyectar ("topic_swap" o "filler_injection")

    Returns:
        SegmentationProblem con true_cuts conocidos.
    """
    rng = random.Random(seed)
    all_topics = list(_TOPIC_TEMPLATES.keys())

    if topics is None:
        if k_true + 1 > len(all_topics):
            topics = [rng.choice(all_topics) for _ in range(k_true + 1)]
        else:
            topics = rng.sample(all_topics, k_true + 1)

    # Dividir n elementos en k_true+1 segmentos aproximadamente iguales
    seg_sizes = _distribute(n, k_true + 1, rng)
    cuts = []
    acc = 0
    for size in seg_sizes[:-1]:
        acc += size
        cuts.append(acc - 1)  # último índice del segmento

    elements: List[TextElement] = []
    idx = 0
    for seg_idx, (topic, size) in enumerate(zip(topics, seg_sizes)):
        templates = _TOPIC_TEMPLATES[topic]
        for _ in range(size):
            text = rng.choice(templates)
            
            # Ocasionalmente insertar ruido
            if rng.random() < noise_ratio:
                if noise_type == "topic_swap" and len(all_topics) > 1:
                    noise_topic = rng.choice([t for t in all_topics if t != topic])
                    text = rng.choice(_TOPIC_TEMPLATES[noise_topic])
                elif noise_type == "filler_injection":
                    filler = rng.choice(_TRANSITION_FILLERS)
                    if rng.choice([True, False]):
                        text = f"{filler} {text}"
                    else:
                        text = f"{text} {filler}"
            
            elements.append(TextElement(idx=idx, text=text))
            idx += 1

    name = f"synthetic_n{n}_k{k_true}_seed{seed}_{noise_type}_noise{noise_ratio}"
    return SegmentationProblem(
        elements=elements,
        name=name,
        true_cuts=cuts,
        lambda_pen=0.1,
    )


def _distribute(n: int, k: int, rng: random.Random) -> List[int]:
    """Distribuye n en k grupos de tamaño similar (mínimo 1 cada uno)."""
    if k > n:
        raise ValueError(f"No se puede distribuir {n} elementos en {k} grupos.")
    base = n // k
    remainder = n % k
    sizes = [base + (1 if i < remainder else 0) for i in range(k)]
    rng.shuffle(sizes)
    return sizes


# ──────────────────────────────────────────────────────────────────────────────
# Generación de instancias estándar (para los experimentos)
# ──────────────────────────────────────────────────────────────────────────────

def generate_standard_instances(
    output_dir: str = "data/instances",
    seed: int = 42,
    noise_ratio: float = 0.1,
    noise_type: str = "topic_swap",
) -> None:
    """
    Genera y guarda el conjunto estándar de instancias sintéticas para los experimentos.

    Tamaños:
        small  : n=15,  k=3  → 5 instancias
        medium : n=60,  k=7  → 5 instancias
        large  : n=250, k=12 → 5 instancias
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs = [
        ("small",  15,  3),
        ("medium", 60,  7),
        ("large",  250, 12),
    ]

    rng = random.Random(seed)
    for size_name, n, k in configs:
        for rep in range(1, 6):
            inst_seed = rng.randint(0, 10_000)
            problem = generate_synthetic(
                n=n,
                k_true=k,
                seed=inst_seed,
                noise_ratio=noise_ratio,
                noise_type=noise_type,
            )
            problem.name = f"{size_name}_{rep:02d}"
            save_path = output_path / f"{size_name}_{rep:02d}.json"
            problem.save(str(save_path))
            print(f"[instance] Guardado: {save_path} (n={n}, k_true={k}, noise={noise_ratio}, type={noise_type})")


def load_wikipedia_as_instance(
    page_title: str,
    language: str = "es",
) -> SegmentationProblem:
    """
    Descarga un artículo de Wikipedia en español (o el idioma especificado),
    lo divide en párrafos y utiliza las cabeceras (h2, h3) para definir las
    fronteras de cortes verdaderas (ground truth).
    """
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "text",
        "redirects": 1,
    }
    headers = {
        "User-Agent": "OptimalSegmentationBot/1.0 (academic IA project; contact: student@example.com)"
    }
    print(f"[instance] Descargando artículo de Wikipedia: '{page_title}' (idioma: {language}) ...")
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise ValueError(f"Error al descargar página de Wikipedia: {data['error'].get('info', 'Desconocido')}")

    html_content = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html_content, "html.parser")

    # Eliminar elementos irrelevantes (infoboxes, referencias, tablas, navboxes)
    for tag in soup(["table", "div", "style", "script"]):
        if tag.name == "table" or (tag.name == "div" and any(c in tag.get("class", []) for c in ["navbox", "reflist", "infobox"])):
            tag.decompose()

    body_div = soup.find(class_="mw-parser-output")
    if not body_div:
        body_div = soup

    elements: List[TextElement] = []
    true_cuts: List[int] = []
    current_element_idx = 0

    # Iterar por los elementos hijos del cuerpo de la página
    for child in body_div.children:
        if child.name == "p":
            text = child.get_text().strip()
            # Limpiar notas al pie (e.g. [1], [2])
            text = re.sub(r"\[\d+\]", "", text)
            # Reemplazar saltos de línea y espacios múltiples
            text = re.sub(r"\s+", " ", text).strip()
            
            # Solo considerar párrafos con longitud significativa
            if len(text.split()) >= 15:
                elements.append(TextElement(idx=current_element_idx, text=text))
                current_element_idx += 1
                
        # Detectar cabecera directamente o dentro de un div contenedor
        header_tag = None
        if child.name in ["h2", "h3"]:
            header_tag = child
        elif child.name == "div" and any(c in child.get("class", []) for c in ["mw-heading", "mw-heading2", "mw-heading3"]):
            header_tag = child.find(["h2", "h3"])

        if header_tag is not None:
            # Una cabecera marca el inicio de una nueva sección.
            # El último elemento agregado (si existe) es un límite/corte.
            if current_element_idx > 0:
                boundary_cut = current_element_idx - 1
                # Ignorar ciertas secciones típicas de Wikipedia al final del artículo
                header_text = header_tag.get_text().lower()
                if any(x in header_text for x in ["referencias", "enlaces externos", "bibliografía", "véase también", "notas", "references", "external links"]):
                    # No agregamos más cortes a partir de aquí
                    break
                if boundary_cut not in true_cuts:
                    true_cuts.append(boundary_cut)

    # Eliminar el último corte si coincide con el final de los elementos
    while true_cuts and true_cuts[-1] >= len(elements) - 1:
        true_cuts.pop()

    name = f"wikipedia_{page_title.lower().replace(' ', '_')}"
    print(f"[instance] Instancia de Wikipedia creada: {len(elements)} párrafos, {len(true_cuts)} cortes.")
    return SegmentationProblem(
        elements=elements,
        name=name,
        true_cuts=true_cuts,
        lambda_pen=0.15,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades de carga/guardado
# ──────────────────────────────────────────────────────────────────────────────

def load_instance(path: str) -> SegmentationProblem:
    """Carga una instancia desde un archivo JSON."""
    return SegmentationProblem.load(path)


def save_instance(problem: SegmentationProblem, path: str) -> None:
    """Guarda una instancia en formato JSON."""
    problem.save(path)
    print(f"[instance] Instancia '{problem.name}' guardada en {path}")


def list_instances(directory: str = "data/instances") -> List[str]:
    """Lista todas las instancias JSON en un directorio."""
    p = Path(directory)
    return sorted(str(f) for f in p.glob("*.json"))
