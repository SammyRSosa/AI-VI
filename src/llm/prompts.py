"""
llm/prompts.py — Plantillas de prompts para el evaluador LLM.

Patrón de integración utilizado: #2 — LLM como Evaluador (Juez).
El LLM recibe un segmento de texto y devuelve un score de coherencia [0.0, 1.0]
junto con una breve justificación.

El prompt está diseñado con las siguientes decisiones de ingeniería:
    1. Rol explícito ("Eres un evaluador...") para orientar el comportamiento.
    2. Delimitadores <SEGMENT>...</SEGMENT> para aislar el contenido variable.
    3. Salida forzada en JSON → facilita el parsing determinístico.
    4. Escala definida con anclas semánticas (0.0, 0.5, 1.0) para reducir
       varianza en las respuestas.
    5. Instrucción de brevedad en "razon" para ahorrar tokens.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Prompt principal: evaluación de coherencia de un segmento
# ──────────────────────────────────────────────────────────────────────────────

COHERENCE_PROMPT = """\
Eres un evaluador experto en análisis de coherencia textual. \
Tu tarea es evaluar qué tan coherente y temáticamente unificado es el fragmento de texto que se te proporcionará.

Criterios de evaluación:
- Un texto es COHERENTE si todos sus párrafos o frases tratan sobre el mismo tema o contribuyen a un único hilo narrativo.
- Un texto es INCOHERENTE si mezcla temas distintos y no relacionados entre sí.

<SEGMENT>
{segment_text}
</SEGMENT>

Responde ÚNICAMENTE con un objeto JSON con exactamente esta estructura (sin texto adicional):
{{
  "score": <número decimal entre 0.0 y 1.0>,
  "razon": "<una frase concisa que justifique el score>"
}}

Escala de referencia:
  1.0 → El segmento trata un único tema de forma completamente unificada.
  0.7 → El segmento tiene un tema principal claro con pequeñas digresiones.
  0.5 → El segmento mezcla dos temas parcialmente relacionados.
  0.3 → El segmento mezcla temas distintos con algún hilo conductor débil.
  0.0 → El segmento mezcla temas completamente distintos y no relacionados.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt de comparación: ¿es mejor el corte en A o en B? (para two_phase)
# ──────────────────────────────────────────────────────────────────────────────

BOUNDARY_COMPARISON_PROMPT = """\
Eres un evaluador experto en segmentación temática de texto.
Se te presentarán dos opciones de cómo dividir un conjunto de párrafos en dos segmentos.

Opción A — Corte después del párrafo {cut_a}:
  Segmento izquierdo: {left_a}
  Segmento derecho: {right_a}

Opción B — Corte después del párrafo {cut_b}:
  Segmento izquierdo: {left_b}
  Segmento derecho: {right_b}

¿Cuál opción produce una división temática más natural y limpia?

Responde ÚNICAMENTE con un objeto JSON con exactamente esta estructura (sin texto adicional):
{{
  "mejor_opcion": "A" | "B",
  "razon": "<una frase concisa que justifique la elección>"
}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de formateo de prompts
# ──────────────────────────────────────────────────────────────────────────────

def format_coherence_prompt(segment_text: str) -> str:
    """Formatea el prompt de coherencia con el texto del segmento."""
    # Truncar si es muy largo para no quemar tokens (máx. ~1500 palabras)
    words = segment_text.split()
    if len(words) > 1500:
        truncated = " ".join(words[:1500])
        segment_text = truncated + "\n[... texto truncado por longitud ...]"
    return COHERENCE_PROMPT.format(segment_text=segment_text)


def format_boundary_comparison_prompt(
    cut_a: int,
    left_a: str,
    right_a: str,
    cut_b: int,
    left_b: str,
    right_b: str,
) -> str:
    """Formatea el prompt de comparación de cortes."""

    def _snippet(text: str, max_words: int = 60) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " [...]"

    return BOUNDARY_COMPARISON_PROMPT.format(
        cut_a=cut_a,
        left_a=_snippet(left_a),
        right_a=_snippet(right_a),
        cut_b=cut_b,
        left_b=_snippet(left_b),
        right_b=_snippet(right_b),
    )
