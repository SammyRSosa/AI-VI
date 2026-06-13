# Segmentación Óptima de Contenido — Proyecto Final IA

> **Tema 6**: Dado una secuencia de elementos de texto, dividirla en segmentos contiguos que maximicen la coherencia interna usando Programación Dinámica + LLM como evaluador semántico.

---

## Descripción

Este proyecto implementa un sistema de **segmentación óptima de texto** que combina:
- **Programación Dinámica** (O(n²)) para encontrar el conjunto óptimo de cortes.
- **LLM como evaluador de coherencia semántica** (Patrón Evaluador, rol #2).
- **Estrategia two-phase** para minimizar las llamadas al LLM: embeddings coseno para una segmentación inicial, LLM solo en los cortes ambiguos.

### Tres configuraciones comparadas

| Config | Solver | Métrica coherencia | LLM calls |
|--------|--------|--------------------|-----------|
| A — Baseline | DP coseno | Similitud coseno | 0 |
| B — DP+LLM full | DP + LLM | Score LLM todos los rangos | O(n²) |
| C — Two-phase | DP + LLM | Coseno → LLM en ambiguos | O(k·w²) |

---

## Instalación

### 1. Clonar / descargar el proyecto

```bash
cd tema6-segmentacion-optima
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar el LLM

Copiar `.env.example` a `.env` y editar la API key:

```bash
copy .env.example .env
```

Luego editar `.env`:
```
LLM_PROVIDER=gemini
LLM_MODEL=gemini/gemini-1.5-flash
GOOGLE_API_KEY=tu_api_key_real_aqui
```

**Proveedores soportados:**
- `gemini/gemini-1.5-flash` — Google Gemini (gratuito)
- `groq/llama3-70b-8192` — Groq con Llama 3 (gratuito)
- `gpt-4o-mini` — OpenAI (de pago)
- `claude-haiku-20240307` — Anthropic (de pago)

---

## Uso rápido

### Generar instancias de prueba

```bash
python src/main.py generate-instances
```

Genera 15 instancias JSON en `data/instances/` (5 small, 5 medium, 5 large).

### Ejecutar el solver en una instancia

```bash
# Baseline (sin LLM)
python src/main.py run --instance data/instances/small_01.json --config baseline_cosine

# Two-phase (recomendado)
python src/main.py run --instance data/instances/medium_01.json --config two_phase

# DP+LLM completo (lento para instancias grandes)
python src/main.py run --instance data/instances/small_01.json --config dp_llm_full
```

### Ejecutar el diseño experimental completo (45 corridas)

```bash
python src/main.py run-experiments
```

Los resultados se guardan incrementalmente en `data/results/results_all.csv`.

### Ver resumen de resultados

```bash
python src/main.py summary
```

---

## Interfaz Gráfica (Frontend Streamlit)

El proyecto incluye un dashboard interactivo web premium desarrollado en **Streamlit** que permite:
- **Visualización en tiempo real:** Ver los párrafos agrupados en cards de colores según su segmento temático asignado de forma óptima.
- **Importación directa de Wikipedia:** Descargar artículos de Wikipedia en cualquier idioma directamente desde la interfaz para segmentarlos al instante.
- **Calibración interactiva:** Ajustar de forma dinámica la penalización de cortes $\lambda$ y la ventana de revisión $w$ para ver inmediatamente cómo afectan a la cantidad e intra-coherencia de los segmentos resultantes.
- **Gráficas técnicas:** Visualizar el heatmap de similitud coseno de los embeddings y la distribución de longitudes de segmentos.

Para iniciar la aplicación, ejecuta el siguiente comando:
```bash
streamlit run src/app.py
```

### Ejecutar tests

```bash
# Todos los tests (no requieren LLM real, usan mocks)
pytest tests/ -v

# Solo tests de DP
pytest tests/test_dp.py -v

# Solo tests del cliente LLM
pytest tests/test_llm_client.py -v
```

---

## Estructura del proyecto

```
tema6-segmentacion-optima/
├── README.md
├── requirements.txt
├── .env.example
├── documentacion/
│   └── informe_tecnico.md          ← Informe técnico completo
├── src/
│   ├── problem.py                  ← Definición formal del problema
│   ├── instance.py                 ← Generador de instancias sintéticas
│   ├── solver/
│   │   ├── baseline.py             ← Config A: DP coseno
│   │   ├── dp_llm.py               ← Config B: DP + LLM (todos los rangos)
│   │   └── two_phase.py            ← Config C: Embeddings + LLM en ambiguos
│   ├── llm/
│   │   ├── client.py               ← Wrapper LLM con caché y reintentos
│   │   ├── cache.py                ← Caché en disco (SHA-256)
│   │   └── prompts.py              ← Prompts literales
│   ├── evaluation/
│   │   ├── metrics.py              ← Métricas de evaluación
│   │   └── experiments.py          ← Orquestador de 45 corridas
│   └── main.py                     ← CLI
├── data/
│   ├── instances/                  ← JSONs de instancias
│   ├── results/                    ← CSVs de resultados
│   └── llm_cache/                  ← Caché de respuestas LLM
├── notebooks/
│   └── analysis.ipynb              ← Análisis y visualizaciones
└── tests/
    ├── test_dp.py                  ← Tests del algoritmo DP
    └── test_llm_client.py          ← Tests del cliente LLM (con mocks)
```

---

## Configuración del LLM

El sistema usa **litellm** para abstraer el proveedor. Los parámetros clave:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `LLM_MODEL` | Modelo (formato litellm) | `gemini/gemini-1.5-flash` |
| `LLM_TEMPERATURE` | Temperatura (0.0 = determinístico) | `0.0` |
| `LLM_MAX_TOKENS` | Tokens máximos en respuesta | `256` |
| `LLM_CACHE_DIR` | Directorio del caché | `data/llm_cache` |

Los prompts literales están en [`src/llm/prompts.py`](src/llm/prompts.py).

---

## Reproducibilidad

Los resultados del informe son totalmente reproducibles:
1. Las instancias se generan con semilla fija (`RANDOM_SEED=42`).
2. El LLM usa temperatura 0.0.
3. El caché en `data/llm_cache/` guarda todas las respuestas → re-ejecutar los experimentos es instantáneo.

---

## Informe técnico

Ver [`documentacion/informe_tecnico.md`](documentacion/informe_tecnico.md) para la documentación completa del proyecto incluyendo:
- Modelado formal del problema.
- Pseudocódigo y análisis de complejidad del algoritmo DP.
- Descripción del dataset y proceso de generación.
- Rol del LLM y diseño de prompts.
- Metodología experimental y resultados.

---

*Proyecto Final — Inteligencia Artificial 2025-2026*
